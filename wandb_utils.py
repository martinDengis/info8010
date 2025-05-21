"""
Weights & Biases (wandb) integration utilities for bib number detection model.
This file provides functions for tracking, hyperparameter sweeps, and visualization.
"""
import os
import yaml
import torch
import wandb
from typing import Dict, Any, List, Optional, Union


def init_wandb_run(
    config: Dict[str, Any],
    project_name: str = 'bibnet',
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    notes: Optional[str] = None,
    mode: str = 'online',
):
    """
    Initialize a new wandb run for training or evaluation.

    Args:
        config: Dictionary containing configuration parameters
        project_name: Name of the wandb project
        run_name: Optional custom name for this specific run
        tags: Optional list of tags to categorize the run
        group: Optional group to organize multiple runs together
        notes: Optional notes about this run
        mode: wandb mode ('online', 'offline', or 'disabled')

    Returns:
        wandb.Run: Initialized wandb run object
    """
    # Initialize wandb
    run = wandb.init(
        project=project_name,
        config=config,
        name=run_name,
        tags=tags,
        group=group,
        notes=notes,
        mode=mode,
    )

    # Define artifacts directory for this run
    os.makedirs('wandb_artifacts', exist_ok=True)

    return run


def create_sweep_config(
    method: str = 'bayes',
    metric: Dict[str, str] = {'name': 'eval/mAP', 'goal': 'maximize'},
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a sweep configuration for hyperparameter optimization.

    Args:
        method: Search method ('grid', 'random', or 'bayes')
        metric: Dictionary specifying the metric to optimize
        parameters: Dictionary of parameters to sweep over

    Returns:
        Dict[str, Any]: Sweep configuration dict
    """
    # Default parameters based on config.yaml if none provided
    if parameters is None:
        parameters = {
            'batch_size': {'values': [4, 8, 16]},
            'lr': {'min': 0.00001, 'max': 0.001, 'distribution': 'log_uniform_values'},
            'weight_decay': {'min': 0.0001, 'max': 0.001, 'distribution': 'log_uniform_values'},
            'lambda_obj': {'min': 1.0, 'max': 10.0},
            'lambda_noobj': {'min': 0.1, 'max': 2.0},
            'lambda_coord': {'min': 1.0, 'max': 10.0},
        }

    # Create sweep configuration
    sweep_config = {
        'method': method,
        'metric': metric,
        'parameters': parameters,
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10,
        },
    }

    return sweep_config


def start_sweep(
    sweep_config: Dict[str, Any],
    train_function: callable,
    project_name: str = 'bibnet',
    entity: Optional[str] = None,
    count: int = 5,
) -> str:
    """
    Initialize and start a hyperparameter sweep.

    Args:
        sweep_config: Sweep configuration dict
        train_function: Function that will be called to execute a training run
        project_name: Name of the wandb project
        entity: Optional wandb username or team name
        count: Number of runs to execute

    Returns:
        str: The sweep ID
    """
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)

    # wrapper function to pass to the agent
    def train_agent():
        train_function(wandb.config)

    # Start the sweep agent
    wandb.agent(sweep_id, function=train_agent, count=count)

    return sweep_id


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = '',
    commit: bool = True,
) -> None:
    """
    Log metrics to wandb during training or evaluation.

    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number (epoch or iteration)
        prefix: Optional prefix to add to metric names (e.g., 'train/', 'eval/')
        commit: Whether to immediately commit the metrics to wandb
    """
    # Add prefix to metric names if provided
    if prefix and not prefix.endswith('/'):
        prefix = f'{prefix}/'

    # Create a copy of metrics to edit
    metrics_to_log = metrics.copy()

    # Create a new dictionary with prefixed keys
    prefixed_metrics = {f'{prefix}{k}': v for k, v in metrics_to_log.items()}

    # Log metrics to wandb
    wandb.log(prefixed_metrics, step=step, commit=commit)


def save_model_artifact(
    model: torch.nn.Module,
    run_id: str,
    artifact_name: str = 'model',
    metadata: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[str] = None,
) -> wandb.Artifact:
    """
    Save model as a wandb artifact.

    Args:
        model: PyTorch model to save
        run_id: ID of the current wandb run
        artifact_name: Name for the artifact
        metadata: Optional metadata to attach to the artifact
        checkpoint_path: Optional path to save the model checkpoint

    Returns:
        wandb.Artifact: The created artifact
    """
    # Create a wandb artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type='model',
        metadata=metadata,
    )

    # Save the model state dict
    if checkpoint_path is None:
        checkpoint_path = os.path.join('wandb_artifacts', f'{run_id}/{artifact_name}.pth')

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    # Add the model file to the artifact and log it
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)

    return artifact


def load_model_artifact(
    artifact_path: str,
    model: torch.nn.Module,
    device: Union[str, torch.device] = 'cpu',
) -> torch.nn.Module:
    """
    Load a model from a wandb artifact.

    Args:
        artifact_path: Path to the artifact in format "entity/project/artifact_name:version"
        model: PyTorch model instance to load weights into
        device: Device to load the model to

    Returns:
        torch.nn.Module: The model with loaded weights
    """
    # Download the artifact
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()

    # Find model checkpoint file (usually the only file in the artifact)
    model_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pth')]
    if not model_files:
        raise ValueError(f"No model files found in artifact directory {artifact_dir}")

    checkpoint_path = os.path.join(artifact_dir, model_files[0])

    # Load the model weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    return model


def load_sweep_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load sweep configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dict[str, Any]: Sweep configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert the config to wandb sweep format
    sweep_config = {
        'method': config.get('method', 'bayes'),
        'metric': config.get('metric', {'name': 'eval/mAP', 'goal': 'maximize'}),
        'parameters': config.get('parameters', {}),
    }

    # Handle early termination if present
    if 'early_terminate' in config:
        sweep_config['early_terminate'] = config['early_terminate']

    if sweep_config["parameters"] is None:
        # Add default parameter ranges
        # Batch size
        sweep_config['parameters']['batch_size'] = {
            'values': [4, 8, 16]
        }

        # Learning rate
        sweep_config['parameters']['lr'] = {
            'min': config.get('min_lr', 0.00001),
            'max': config.get('lr', 0.001) * 2,
            'distribution': 'log_uniform_values'
        }

        # Weight decay
        sweep_config['parameters']['weight_decay'] = {
            'min': config.get('weight_decay', 0.0005) / 10,
            'max': config.get('weight_decay', 0.0005) * 2,
            'distribution': 'log_uniform_values'
        }

        # Loss lambdas
        sweep_config['parameters']['lambda_obj'] = {
            'min': config.get('lambda_obj', 5.0) / 2,
            'max': config.get('lambda_obj', 5.0) * 2
        }

        sweep_config['parameters']['lambda_noobj'] = {
            'min': config.get('lambda_noobj', 0.5) / 2,
            'max': config.get('lambda_noobj', 1.0) * 2
        }

        sweep_config['parameters']['lambda_coord'] = {
            'min': config.get('lambda_coord', 5.0) / 2,
            'max': config.get('lambda_coord', 5.0) * 2
        }

    # Add any fixed parameters from the config file
    if 'fixed_parameters' in config:
        for name, value in config['fixed_parameters'].items():
            if name not in sweep_config['parameters']:
                # Add as a fixed parameter (not swept)
                sweep_config['parameters'][name] = {'value': value}

    return sweep_config
