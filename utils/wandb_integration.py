import wandb
from config import get_cfg_defaults


def log_metrics(metrics, step=None):
    """
    Log metrics to wandb if a run is active

    Args:
        metrics (dict): Dictionary of metrics to log
        step (int, optional): Step number for logging
    """
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_summary(metrics, step=None):
    """
    Log summary metrics to wandb if a run is active

    Args:
        metrics (dict): Dictionary of summary metrics to log
        step (int, optional): Step number for logging
    """
    if wandb.run is not None:
        keys = list(metrics.keys())
        values = list(metrics.values())
        for i in range(len(keys)):
            wandb.run.summary[keys[i]] = values[i]


def log_model(model_path, aliases=None):
    """
    Log model to wandb if a run is active

    Args:
        model_path (str): Path to the model file
        aliases (list, optional): List of aliases for the model
    """
    if wandb.run is not None:
        wandb.save(model_path)
        if aliases:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(model_path)
            for alias in aliases:
                wandb.log_artifact(artifact, aliases=[alias])


def finish_run():
    """Close the current wandb run if one exists"""
    if wandb.run is not None:
        wandb.finish()


def get_sweep_config(model_type='bibnet'):
    """
    Get a WandB sweep configuration for BibNet hyperparameter tuning.

    Args:
        model_type (str): Type of model to configure ('bibnet' or 'yolov1)

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
            # Shared parameters for both models
            "optimizer.type": {"values": ["adam", "radam"]},
            "optimizer.learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
            "optimizer.weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-3},

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
            # "training.batch_size": {"values": [8, 16, 32]},
            # "training.num_epochs": {"values": [100, 150]},
        }
    }

    # Model-specific parameters
    if model_type == 'bibnet':
        # BibNet-specific parameters
        sweep_config["parameters"].update({
            "model.p_blocks": {"values": [2, 3, 4, 5]},
            "model.c_blocks": {"values": [1, 2, 3]},
            # need to make sure each feature_channels has same lenght as corresponding p_blocks value
            "model.feature_channels": {"values": [
                [32, 64],                 # For p_blocks=2
                [32, 64, 128],            # For p_blocks=3
                [32, 64, 128, 256],       # For p_blocks=4
                [32, 64, 128, 256, 512],  # For p_blocks=5
            ]},
            "model.num_fc_layers": {"values": [1, 2, 3]},
            "model.hidden_size": {"values": [256, 512, 1024]}
        })

        # Custom constraint: Ensure feature_channels length matches p_blocks
        sweep_config["parameters"]["model.feature_channels"]["values"] = [
            channels for channels in sweep_config["parameters"]["model.feature_channels"]["values"]
            if len(channels) in sweep_config["parameters"]["model.p_blocks"]["values"]
        ]

    return sweep_config

def run_single_experiment(cfg, entity, project, group):
    """
    Launch a single wandb experiment with the current configuration

    Args:
        cfg (dict): Configuration dictionary
        entity (str): WandB entity name
        project (str): WandB project name
        group (str): Group name for this run
    """
    # Initialize a new wandb run
    with wandb.init(entity=entity, project=project, group=group, config=cfg) as run:
        # Training logic
        from tools.train_net import train
        train(cfg)

def main(model_type=None, run_sweep=False):
    # Get default configuration
    cfg = get_cfg_defaults()
    entity = cfg["wandb"]["entity"]
    project = cfg["wandb"]["project"]

    # Override model type if specified
    if model_type is not None:
        cfg["model"]["type"] = model_type

    # Model type from config
    model_type = cfg["model"]["type"]

    # Set group based on model_type
    group = f"{model_type}-runs"
    cfg["wandb"]["group"] = group

    if run_sweep:
        # Get sweep configuration
        sweep_config = get_sweep_config(model_type)

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
    else:
        # Run a single experiment with the current configuration
        run_single_experiment(cfg, entity, project, group)



if __name__ == "__main__":
    main()
