import torch
import os
import time
from utils.wandb_integration import log_metrics, log_model, log_summary
from torchvision import tv_tensors


# ==============================
# Helper Functions
# ==============================

def train_epoch(model, train_loader, loss_fn, optimizer, device, accumulation_steps, current_epoch=0, max_epochs=100):
    """Run one epoch of training"""
    model.train()
    epoch_loss = 0.0
    # Add tracking for loss components
    loss_components = {
        'bbox_loss': 0.0,
        'conf_loss': 0.0,
        'match_rate': 0.0
    }

    # Set current epoch in loss function if it has the method
    if hasattr(loss_fn, 'set_epoch_info'):
        loss_fn.set_epoch_info(current_epoch, max_epochs)

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)

        with torch.set_grad_enabled(True):
            predictions = model(images)

            # Move bounding boxes to the same device as the model
            bboxes = []
            for t in targets:
                if 'bboxes' in t and t['bboxes'] is not None:
                    t['bboxes'] = t['bboxes'].to(device)
                else:
                    # Provide a default empty bounding box tensor
                    t['bboxes'] = tv_tensors.BoundingBoxes(
                        torch.zeros((0, 4), device=device),
                        format="xywh",
                        canvas_size=(images.shape[2], images.shape[3])
                    )
                bboxes.append(t['bboxes'])

            batch_dict = {
                'images': images,
                'bboxes': bboxes,
                'labels': [t['labels'] for t in targets]
            }
            loss, loss_dict = loss_fn(batch_dict, predictions)
            loss = loss / accumulation_steps

            loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps

            # Accumulate loss components
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key].item() * accumulation_steps

    print(f'Average loss for epoch: {epoch_loss / len(train_loader)}')

    # Average the loss components
    for key in loss_components:
        loss_components[key] /= len(train_loader)

    return epoch_loss / len(train_loader), loss_components


def validate(model, val_loader, loss_fn, device):
    """Run validation"""
    model.eval()
    val_loss = 0.0
    # Add tracking for loss components
    loss_components = {
        'bbox_loss': 0.0,
        'conf_loss': 0.0,
        'match_rate': 0.0
    }

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            predictions = model(images)

            # Move bounding boxes to the same device as the model
            bboxes = []
            for t in targets:
                # Move bounding boxes to the same device
                if 'bboxes' in t and t['bboxes'] is not None:
                    t['bboxes'] = t['bboxes'].to(device)
                bboxes.append(t['bboxes'])

            batch_dict = {
                'images': images,
                'bboxes': bboxes,
                'labels': [t['labels'] for t in targets]
            }
            loss, loss_dict = loss_fn(batch_dict, predictions)
            val_loss += loss.item()

            # Accumulate loss components
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key].item()

    # Average the loss components
    for key in loss_components:
        loss_components[key] /= len(val_loader)

    return val_loss / len(val_loader), loss_components


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


# =============================
# Core Training Function
# =============================

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
    early_stopped = False
    patience = early_stopping_config.get('patience', 10)
    best_val_loss = float('inf')
    patience_counter = 0

    # Moving average for validation loss
    ema_alpha = 0.9
    val_loss_ema = None  # Initialize EMA tracker

    # Training parameters
    num_epochs = cfg.get('training', {}).get('num_epochs', 100)
    best_epoch = -1

    accumulation_steps = cfg.get('training', {}).get(
        'gradient_accumulation_steps', 8)
    accumulation_steps = max(1, accumulation_steps)

    save_freq = cfg.get('training', {}).get('save_freq', 15)
    best_model_path = None
    start_time = time.time()

    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()  # Clear GPU memory
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Training...")

        # ----- Training phase -----
        avg_train_loss, train_loss_components = train_epoch(
            model, train_loader, loss_fn, optimizer, device, accumulation_steps, epoch, num_epochs)

        # ----- Validation phase -----
        avg_val_loss, val_loss_components = validate(model, val_loader, loss_fn, device)

        # ----- Logging -----
        if val_loss_ema is None:
            val_loss_ema = avg_val_loss  # Initialize with first value
        else:
            val_loss_ema = ema_alpha * val_loss_ema + (1 - ema_alpha) * avg_val_loss

        metrics = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_loss_ema': val_loss_ema,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch
        }

        metrics.update({f'train_{key}': value for key, value in train_loss_components.items()})
        metrics.update({f'val_{key}': value for key, value in val_loss_components.items()})
        log_metrics(metrics)    # wandb logging

        # ----- Scheduler step -----
        if scheduler is not None:
            scheduler.step()

        # ----- Early stopping -----
        if early_stopping_enabled:
            if val_loss_ema < best_val_loss:
                best_val_loss = val_loss_ema
                best_epoch = epoch
                patience_counter = 0
                # Save best model
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
                save_checkpoint(model, optimizer, epoch,
                                avg_val_loss, best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stopped = True
                    break

        # ----- Checkpointing -----
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, epoch,
                            avg_train_loss, checkpoint_path)

    # ----- End of training -----
    # Calculate training time
    training_time = time.time() - start_time
    stop_epoch = epoch

    # Log summary metrics
    summary = {
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss,
        'final_val_loss_ema': val_loss_ema,
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
