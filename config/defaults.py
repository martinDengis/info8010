"""
Default configuration for BibNet model and training.
Uses plain Python dictionaries for easier integration with wandb.
"""

def get_cfg_defaults():
    """
    Get a dictionary with default values for BibNet project.

    Returns:
        dict: A dictionary with default configuration values
    """
    cfg = {
        # Model config
        "model": {
            "input_channels": 3,
            "backbone_channels": [64, 128, 256, 512],
            "neck_channels": 256,
            "num_res_blocks": [1, 2, 8, 8],
            "num_coords": 4
        },

        # Optimizer config
        "optimizer": {
            "type": "adam",  # Options: adam, radam, adamw
            "learning_rate": 1e-3,
            "weight_decay": 5e-4
        },

        # Scheduler config
        "scheduler": {
            "use_scheduler": True,
            "type": "step",  # Options: step, cosine
            "use_warmup": False,
            "warmup_epochs": 5,
                # Step scheduler parameters
            "step_size": 30,
            "gamma": 0.1,
                # Cosine scheduler parameters
            "t_max": 100
        },

        # Input config
        "input": {
            "size": (640, 640)
        },

        # Training config
        "batch_size": 16,
        "num_epochs": 100,
        "save_freq": 15,

        # wandb config
        "wandb": {
            "project": "bibnet",
            "entity": "your_entity",
            "name": None,
            "group": "model-v1",
        },
    }

    return cfg


def update_cfg(cfg, updates):
    """
    Update configuration with new values, handling nested dictionaries.

    Args:
        cfg (dict): Original configuration dictionary
        updates (dict): Dictionary with values to update

    Returns:
        dict: Updated configuration dictionary
    """
    for k, v in updates.items():
        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
            # Recursively update nested dictionaries
            cfg[k] = update_cfg(cfg[k], v)
        else:
            # Direct update for non-dictionary values
            cfg[k] = v
    return cfg
