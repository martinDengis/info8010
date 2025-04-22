import torch
import os

def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, checkpoint_dir='checkpoints'):
    """
    Main training loop for the model
    """
    # Preparation steps
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_fn.to(device)

    # Training loop
    num_epochs = cfg.NUM_EPOCHS
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            images = batch['images'].to(device)
            predictions = model(images)

            # Compute loss
            loss = loss_fn(batch, predictions)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track loss
            epoch_loss += loss.item()

        # Validation code
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                images = batch['images'].to(device)
                predictions = model(images)
                loss = loss_fn(batch, predictions)
                val_loss += loss.item()

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # Save checkpoint
        save_freq = cfg.get('save_freq', 15)
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
