import wandb
from config import *


def log_metrics(metrics, step=None):
    """
    Log metrics to wandb

    Args:
        metrics (dict): Dictionary of metrics to log
        step (int, optional): Step number for logging
    """
    wandb.log(metrics, step=step)


def log_summary(metrics, step=None):
    """
    Log summary metrics to wandb

    Args:
        metrics (dict): Dictionary of summary metrics to log
        step (int, optional): Step number for logging
    """
    keys = list(metrics.keys())
    values = list(metrics.values())
    for i in range(len(keys)):
        wandb.run.summary[keys[i]] = values[i]


def log_model(model_path, aliases=None):
    """
    Log model to wandb

    Args:
        model_path (str): Path to the model file
        aliases (list, optional): List of aliases for the model
    """
    wandb.save(model_path)
    if aliases:
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(model_path)
        for alias in aliases:
            wandb.log_artifact(artifact, aliases=[alias])


def finish_run():
    """Close the current wandb run"""
    wandb.finish()


def get_sweep_config():
    """
    Get a WandB sweep configuration for BibNet hyperparameter tuning.

    Returns:
        dict: A WandB sweep configuration
    """
    sweep_config = {
        "method": "bayes",
        "metric": {
            "name": "val_loss",
            "goal": "minimize"
        },
        "parameters": {
            # Model parameters
            "model.backbone_channels": {"values": [
                [32, 64, 128, 256],
                [64, 128, 256, 512],
                [128, 256, 512, 1024]
            ]},
            "model.neck_channels": {"values": [128, 256, 512]},
            "model.num_res_blocks": {"values": [
                [1, 2, 8, 8],
                [2, 4, 8, 8],
                [1, 2, 4, 4]
            ]},

            # Optimizer parameters
            "optimizer.type": {"values": ["adam", "radam"]},
            "optimizer.learning_rate": {"distribution": "log_uniform", "min": -5, "max": -2},  # 1e-5 to 1e-2
            "optimizer.weight_decay": {"distribution": "log_uniform", "min": -6, "max": -3}, # 1e-6 to 1e-3

            # Scheduler parameters
            "scheduler.type": {"values": ["step", "cosine"]},
            "scheduler.use_warmup": {"values": [True, False]},
            "scheduler.warmup_epochs": {"values": [3, 5, 10]},
            "scheduler.step_size": {"values": [15, 30, 45]},
            "scheduler.gamma": {"values": [0.1, 0.2, 0.5]},
            "scheduler.t_max": {"values": [50, 100, 150]},

            # Early stopping parameters
            "early_stopping.patience": {"values": [5, 10, 15]},

            # Training parameters
            "training.batch_size": {"values": [8, 16]},
            "training.num_epochs": {"values": [100, 150]},
        }
    }

    return sweep_config


def main():
    # Get default configuration
    cfg = get_cfg_defaults()
    entity = cfg["wandb"]["entity"]
    project = cfg["wandb"]["project"]
    group = cfg["wandb"]["group"]

    # Get sweep configuration
    sweep_config = get_sweep_config()

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=cfg["wandb"]["project"])

    # Define the training function for each sweep run
    def train_sweep():
        # Initialize a new wandb run
        with wandb.init(entity=entity, project=project, group=group) as run:
            # Update config with sweep parameters
            for key, value in run.config.items():
                if '.' in key:
                    # Handle nested config parameters
                    parts = key.split('.')
                    config_dict = cfg
                    for part in parts[:-1]:
                        # Handle array indices in the configuration path
                        if part.isdigit():
                            part = int(part)
                        config_dict = config_dict[part]

                    # Set the actual value
                    last_part = parts[-1]
                    if last_part.isdigit():
                        last_part = int(last_part)
                    config_dict[last_part] = value
                else:
                    # Handle top-level parameters
                    cfg[key] = value

            # Training logic
            from tools.train_net import train
            train(cfg)

    # Start the sweep agent
    wandb.agent(sweep_id, train_sweep, count=10)  # Run 10 trials


if __name__ == "__main__":
    main()
