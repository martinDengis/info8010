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


def log_ap_values(ap_dict, epoch):
    """Log AP values at different thresholds as a wandb Table"""
    if wandb.run is not None:
        # Create table data with thresholds and AP values
        data = [[float(k.split('@')[1]), v] for k, v in ap_dict.items()]
        table = wandb.Table(data=data, columns=["IoU Threshold", "AP"])
        thresholds = [row[0] for row in data]
        ap_values = [row[1] for row in data]
        wandb.log({
            "AP_values_table": table,
            "AP_values_plot": wandb.plot.line(
                table,
                "IoU Threshold",
                "AP",
                title="AP Values at Different IoU Thresholds"
            )
        }, step=epoch)


def log_precision_recall_curve(curve_data, epoch=None):
    """
    Log precision-recall curve data to wandb as a line plot

    Args:
        curve_data (dict): Dictionary containing 'precision' and 'recall' numpy arrays
        epoch (int, optional): Current epoch for logging
    """
    if wandb.run is not None:
        # Create line plot data
        data = [[x, y]
                for x, y in zip(curve_data["recall"], curve_data["precision"])]

        # Create a wandb Table with the data
        table = wandb.Table(data=data, columns=["Recall", "Precision"])

        # Log the precision-recall curve as a line chart
        wandb.log({
            "precision_recall_curve": wandb.plot.line(
                table, "Recall", "Precision",
                title="Precision-Recall Curve (IoU=0.5)"
            )
        }, step=epoch)

        # Additionally, log raw data for custom visualizations
        wandb.log({
            "precision_recall_data": {
                "precision": curve_data["precision"],
                "recall": curve_data["recall"]
            }
        }, step=epoch)


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
            "name": "mAP",
            "goal": "maximize"
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

                    # Fix feature_channels to match p_blocks
                    if "model" in cfg and "p_blocks" in cfg["model"] and "feature_channels" in cfg["model"]:
                        p_blocks = cfg["model"]["p_blocks"]
                        feature_channels = cfg["model"]["feature_channels"]

                        # Check if we need to adjust feature_channels
                        if len(feature_channels) != p_blocks:
                            # Create appropriate feature_channels for this p_blocks value
                            if p_blocks == 2:
                                cfg["model"]["feature_channels"] = [32, 64]
                            elif p_blocks == 3:
                                cfg["model"]["feature_channels"] = [32, 64, 128]
                            elif p_blocks == 4:
                                cfg["model"]["feature_channels"] = [
                                    32, 64, 128, 256]
                            elif p_blocks == 5:
                                cfg["model"]["feature_channels"] = [
                                    32, 64, 128, 256, 512]
                            else:
                                # Fallback - create array with duplicated values
                                cfg["model"]["feature_channels"] = [
                                    64] * p_blocks

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
