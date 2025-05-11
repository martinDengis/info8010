import torch
import os
import time
from utils.wandb_integration import log_metrics, log_model, log_summary, log_ap_values, log_precision_recall_curve
from utils.loss_utils import get_bboxes, evaluate_detection_performance


# ==============================
# Helper Functions
# ==============================

def train_epoch(model, train_loader, loss_fn, optimizer, acc_steps, device):
    """Run one epoch of training"""
    model.train()
    epoch_loss = 0.0

    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        with torch.set_grad_enabled(True):
            preds = model(images)

            loss = loss_fn(preds, targets)
            loss = loss / acc_steps
            loss.backward()

            if ((batch_idx + 1) % acc_steps == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * acc_steps

    avg_train_loss = epoch_loss / len(train_loader)
    print(f'Average loss for epoch: {avg_train_loss}')
    return avg_train_loss


def validate(model, val_loader, loss_fn, val_loss_ema, device, ema_alpha=0.9):
    """Run validation"""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)

            loss = loss_fn(preds, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    # EMA Validation Loss
    if val_loss_ema is None:
        val_loss_ema = avg_val_loss  # Initialize with first value
    else:
        val_loss_ema = ema_alpha * val_loss_ema + (1 - ema_alpha) * avg_val_loss

    return avg_val_loss, val_loss_ema

def detection_performance(model, train_loader, thresholds, device):
    pred_boxes, target_boxes = get_bboxes(
        train_loader, model, iou_threshold=0.5, threshold=0.4, device=device
    )

    # Single call to evaluate all metrics
    eval_results = evaluate_detection_performance(
        pred_boxes, target_boxes, thresholds, box_format="midpoint"
    )

    # Extract results
    mAP = eval_results['mAP']
    ap_dict = eval_results['ap_values']
    detailed_metrics = eval_results.get('detailed_metrics', None)
    return mAP, ap_dict, detailed_metrics


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
    estopping_config = cfg.get('early_stopping', {})
    early_stopping_enabled = estopping_config.get('enabled', False)
    patience = estopping_config.get('patience', 10)
    patience_counter = 0
    early_stopped = False
    best_val_loss = float('inf')

    # Init Val EMA Tracker
    val_loss_ema = None

    # Initialize mAP thresholds
    thresholds = [t/100 for t in range(50, 100, 5)]

    # Training parameters
    training_config = cfg.get('training', {})
    num_epochs = training_config.get('num_epochs', 100)
    best_epoch = -1

    acc_steps = training_config.get('grad_acc_steps', 8)
    acc_steps = max(1, acc_steps)

    save_freq = training_config.get('save_freq', 15)
    best_model_path = None
    start_time = time.time()

    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()  # Clear GPU memory
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Training...")

        # ----- Detection Performance Evaluation -----
        res = detection_performance(model, train_loader, thresholds, device)
        mAP, ap_dict, detailed_metrics = res
        print(f"Train mAP@[.5:.05:.95]: {mAP}")

        # ----- Training phase -----
        avg_train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            acc_steps,
            device,
        )

        # ----- Validation phase -----
        avg_val_loss, val_loss_ema = validate(
            model,
            val_loader,
            loss_fn,
            val_loss_ema,
            device,
        )

        # ----- Scheduler step -----
        if scheduler is not None:
            scheduler.step()

        # ----- Logging -----
        log_metrics({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_loss_ema': val_loss_ema,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'mAP': mAP,
        }, epoch)
        log_ap_values(ap_dict, epoch)
        log_precision_recall_curve(detailed_metrics['precision_recall_curve'], epoch)

        # ----- Early stopping -----
        if val_loss_ema < best_val_loss:
            best_val_loss = val_loss_ema
            best_epoch = epoch
            patience_counter = 0
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch,
                            avg_val_loss, best_model_path)

            if early_stopping_enabled:
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
    training_time = (time.time() - start_time) / 60.0  # training time in minutes
    stop_epoch = epoch

    # Log summary metrics
    summary = {
        'best_val_loss': best_val_loss,
        'early_stopped': early_stopped,
        'best_epoch': best_epoch,
        'stop_epoch': stop_epoch,
        'total_epoch': num_epochs,
        'training_time_minutes': training_time,
        'nb_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    log_summary(summary)

    # Save the final model
    if best_model_path:
        log_model(best_model_path, aliases=['best_model'])

    return model, summary

