import torch
import os
import time
from utils import log_metrics, log_model, log_summary


def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, checkpoint_dir='checkpoints'):
    """
    Main training loop for the model
    """
    # Preparation steps
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_fn.to(device)

    # Early stopping configuration
    early_stopping_config = cfg.get('early_stopping', {'enabled': False})
    early_stopping_enabled = early_stopping_config.get('enabled', False)
    patience = early_stopping_config.get('patience', 10)
    best_val_loss = float('inf')
    patience_counter = 0

    # Track training information
    best_epoch = -1
    early_stopped = False
    start_time = time.time()

    # Training loop
    num_epochs = cfg.get('training', {}).get('num_epochs', 100)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            images, targets = batch_data # batch_data is a tuple (images, targets): from collate_fn
            images = images.to(device)
            predictions = model(images)

            # Compute loss
            batch_dict = {
                'images': images,
                'bboxes': [t['boxes'] for t in targets],
                'labels': [t['labels'] for t in targets]
            }
            loss = loss_fn(batch_dict, predictions)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Track loss
            epoch_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)

        # Validation code
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch_data in val_loader:
                images, targets = batch_data
                images = images.to(device)
                predictions = model(images)

                batch_dict = {
                    'images': images,
                    'bboxes': [t['boxes'] for t in targets],
                    'labels': [t['labels'] for t in targets]
                }
                loss = loss_fn(batch_dict, predictions)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics to wandb
        log_metrics({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': current_lr,
            'epoch': epoch
        })

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # Early stopping check
        if early_stopping_enabled:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                # Save best model
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stopped = True
                    break

        # Save checkpoint
        save_freq = cfg.get('training', {}).get('save_freq', 15)
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)

    # Calculate training time
    training_time = time.time() - start_time
    stop_epoch = epoch

    # Log summary metrics
    summary = {
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss,
        'best_val_loss': best_val_loss,
        'early_stopped': early_stopped,
        'best_epoch': best_epoch,
        'stop_epoch': stop_epoch,
        'total_epoch': num_epochs,
        'training_time': training_time,
        'nb_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    log_summary(summary)

    # Save the final model
    if best_model_path:
        log_model(best_model_path, aliases=['best_model'])

    return model, summary
