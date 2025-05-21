"""
Script for using wandb integration with the training pipeline.
"""
import os
import argparse
import yaml
import torch
import wandb
from train import main as train_main

# Import wandb utilities
from wandb_utils import (
    init_wandb_run,
    create_sweep_config,
    start_sweep,
    log_metrics,
    save_model_artifact,
    load_sweep_config_from_file
)

def train_with_wandb(config=None):
    """
    Training function that wraps the main training
    function with wandb integration.

    Args:
        config: Configuration dictionary, potentially from wandb sweep
    """
    # Initialize wandb
    run = init_wandb_run(
        config=config,
        project_name="bibnet",
        run_name=None,
        group="bibnetv4-runs",
        tags=["object-detection", "yolo", "bib-number"],
    )

    # Monkey patch the train and evaluate functions to log metrics
    original_train_one_epoch = train_main.__globals__['train_one_epoch']
    original_evaluate = train_main.__globals__['evaluate']
    original_save_checkpoint = train_main.__globals__['save_checkpoint']

    def train_one_epoch_with_logging(*args, **kwargs):
        # Extract optimizer from args (3rd param of train_one_epoch)
        optimizer = args[2] if len(args) > 2 else kwargs.get('optimizer')

        metrics = original_train_one_epoch(*args, **kwargs)
        lr = optimizer.param_groups[0]['lr']
        metrics["learning_rate"] = lr

        # Log training metrics
        log_metrics(metrics, prefix="train")

        return metrics

    def evaluate_with_logging(*args, **kwargs):
        metrics = original_evaluate(*args, **kwargs)
        log_metrics(metrics, prefix="eval")
        return metrics

    def save_checkpoint_with_wandb(model, optimizer, scheduler, epoch, loss, metrics, checkpoint_dir):
        # Call original save_checkpoint
        original_save_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, checkpoint_dir)

        # Save model as wandb artifact if this is the best model
        if epoch % 10 == 0 or metrics.get("mAP", 0) > run.summary.get("best_map", 0):
            is_best = metrics.get("mAP", 0) > run.summary.get("best_map", 0)

            # Update best mAP if needed
            if is_best:
                run.summary["best_map"] = metrics.get("mAP", 0)

            # Save regular checkpoint
            save_model_artifact(
                model=model,
                run_id=run.id,
                artifact_name=f"model-epoch-{epoch}",
                metadata={
                    "epoch": epoch,
                    "loss": loss,
                    "metrics": metrics,
                    "is_best": is_best,
                },
            )

            # Save a separate "best" checkpoint when a new best model is found
            if is_best:
                save_model_artifact(
                    model=model,
                    run_id=run.id,
                    artifact_name="best-model",
                    metadata={
                        "epoch": epoch,
                        "loss": loss,
                        "metrics": metrics,
                        "is_best": True,
                    },
                )

    # Replace the functions with logging versions
    train_main.__globals__['train_one_epoch'] = train_one_epoch_with_logging
    train_main.__globals__['evaluate'] = evaluate_with_logging
    train_main.__globals__['save_checkpoint'] = save_checkpoint_with_wandb

    try:
        # Run the training
        train_main(run.config)
    finally:
        # Restore original functions
        train_main.__globals__['train_one_epoch'] = original_train_one_epoch
        train_main.__globals__['evaluate'] = original_evaluate
        train_main.__globals__['save_checkpoint'] = original_save_checkpoint

        # Finish the wandb run
        run.finish()


def main():
    """Main function to start wandb runs or sweeps."""
    parser = argparse.ArgumentParser(description="Train with wandb integration")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--sweep", action="store_true", help="Run as a wandb sweep")
    parser.add_argument("--count", type=int, default=5, help="Number of sweep runs to execute")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.sweep:
        # Create and start sweep
        sweep_config = load_sweep_config_from_file(args.config)
        sweep_id = start_sweep(
            sweep_config=sweep_config,
            train_function=train_with_wandb,
            count=args.count
        )
        print(f"Started sweep with ID: {sweep_id}")
    else:
        # Run a single training run with wandb
        train_with_wandb(config)


if __name__ == "__main__":
    main()
