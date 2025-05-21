"""
Training script for the bib number detection model.
"""
import os
import argparse
import yaml
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from model import BibNet
from loss import BibNetLoss, Evaluator
from dataloader import create_dataloaders


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch=None):
    """
    Train model for one epoch.

    Args:
        model (nn.Module): The model to train
        dataloader (DataLoader): Training data loader
        optimizer (Optimizer): The optimizer
        criterion (nn.Module): Loss function
        device (torch.device): Device to use

    Returns:
        dict: Dictionary with losses
    """
    model.train()
    epoch_loss = 0
    loss_components_sum = {
        "obj_loss": 0,
        "noobj_loss": 0,
        "bbox_loss": 0,
    }

    for images, targets in dataloader:
        # Move to device
        # input should be a tensor batch, not a list of tensors
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()                                 # Reset gradients
        predictions = model(images)                           # Forward pass
        loss, loss_components = loss_fn(predictions, targets) # Calculate loss
        loss.backward()                                       # Backward pass
        optimizer.step()                                      # Update weights

        # Update metrics
        epoch_loss += loss.item()
        for k, v in loss_components.items():
            loss_components_sum[k] += v.item()

    # Calculate average losses
    num_batches = len(dataloader)
    epoch_loss /= num_batches
    for k in loss_components_sum:
        loss_components_sum[k] /= num_batches

    return {
        "loss": epoch_loss,
        **loss_components_sum,
        "epoch": epoch,
    }


def evaluate(model, dataloader, device, evaluator, loss_fn=None, epoch=None):
    """Evaluate model on validation set with loss and AP metrics."""
    model.eval()
    predictions = []
    targets_list = []
    total_val_loss = 0
    loss_components_sum = {
        "obj_loss": 0,
        "noobj_loss": 0,
        "bbox_loss": 0,
    }

    with torch.no_grad():
        for images, targets in dataloader:
            # Move to device
            images = images.to(device)
            targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
            targets_list.extend(targets)

            # Get predictions:
            # `model(images)` for loss calculation, and
            # `model.predict(images)` for evaluation (transformed predictions)
            outputs = model(images)
            predictions_batch = model.predict(images)
            predictions.extend(predictions_batch)

            # Calculate val loss if loss_fn provided
            if loss_fn is not None:
                val_loss, loss_components = loss_fn(outputs, targets_device)
                total_val_loss += val_loss.item()
                for k, v in loss_components.items():
                    loss_components_sum[k] += v.item()

    # Convert targets to device for evaluation
    targets_list = [{k: v.to(device) for k, v in t.items()} for t in targets_list]

    # Calculate AP metrics
    metrics = evaluator.evaluate(predictions, targets_list)

    # Add val loss to metrics if calculated
    if loss_fn is not None:
        num_batches = len(dataloader)
        metrics["val_loss"] = total_val_loss / num_batches
        for k in loss_components_sum:
            metrics[f"val_{k}"] = loss_components_sum[k] / num_batches

    metrics["epoch"] = epoch

    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, checkpoint_dir):
    """
    Save checkpoint.

    Args:
        model (nn.Module): The model
        optimizer (Optimizer): The optimizer
        scheduler (LRScheduler): The learning rate scheduler
        epoch (int): Current epoch
        loss (float): Current loss
        metrics (dict): Current metrics
        checkpoint_dir (str): Directory to save checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
        "metrics": metrics,
    }, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """
    Load checkpoint.

    Args:
        model (nn.Module): The model
        optimizer (Optimizer): The optimizer
        scheduler (LRScheduler): The learning rate scheduler
        checkpoint_path (str): Path to checkpoint
        device (torch.device): Device to use

    Returns:
        tuple: (epoch, loss, metrics)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    metrics = checkpoint["metrics"]

    print(f"Loaded checkpoint from epoch {epoch}")

    return epoch, loss, metrics


def main(config):
    """
    Main training function.

    Args:
        config (dict): Configuration dictionary
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Create data loaders
    train_loader, valid_loader, test_loader = create_dataloaders(
        data_dir=config["data_dir"],
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # Create model
    model = BibNet(
        img_size=config["img_size"],
        backbone_features=config["backbone_features"],
    ).to(device)

    # Create criterion
    criterion = BibNetLoss(
        img_size=config["img_size"],
        grid_sizes=config["grid_sizes"],
        lambda_obj=config["lambda_obj"],
        lambda_noobj=config["lambda_noobj"],
        lambda_coord=config["lambda_coord"],
    ).to(device)

    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # Create LR scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["min_lr"],
    )

    # Create evaluator
    evaluator = Evaluator()

    # Initialize variables
    start_epoch = 0
    best_map = 0.0

    # Load checkpoint if specified
    if config["resume"] and os.path.exists(config["resume"]):
        start_epoch, _, _ = load_checkpoint(
            model, optimizer, scheduler, config["resume"], device
        )
        start_epoch += 1

    # Training loop
    for epoch in range(start_epoch, config["epochs"]):
        if (epoch == 0) or (epoch + 1) % 10 == 0:
            print(f"\n{'=' * 20} Epoch {epoch + 1}/{config['epochs']} {'=' * 20}")

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch=epoch)
        print(f"Trained epoch {epoch + 1} - Loss: {train_metrics['loss']:.4f}")

        # Update learning rate
        scheduler.step()

        # Evaluate
        eval_metrics = evaluate(model, valid_loader, device, evaluator, loss_fn=criterion, epoch=epoch)
        print(f"Evaluated epoch {epoch + 1} - mAP: {eval_metrics.get('mAP', 0):.4f}")

        # Print metrics every 10 epochs
        if (epoch == 0) or (epoch + 1) % 10 == 0:
            print("Training metrics:")
            for k, v in train_metrics.items():
                if isinstance(v, (float, int)):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

            print("Evaluation metrics:")
            for k, v in eval_metrics.items():
                if isinstance(v, (float, int)):
                    print(f"  {k}: {v:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or eval_metrics["mAP"] > best_map:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_metrics["loss"],
                eval_metrics, config["checkpoint_dir"]
            )

        # Save best model
        if eval_metrics["mAP"] > best_map:
            best_map = eval_metrics["mAP"]
            best_model_path = os.path.join(config["checkpoint_dir"], "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with mAP: {best_map:.4f}")

    # Test with best model
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate(model, test_loader, device, evaluator)

    print("Test metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, (float, int)):
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bib number detection model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
