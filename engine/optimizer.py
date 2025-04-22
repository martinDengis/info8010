import torch.optim as optim


def build_optimizer(cfg, model):
    """Creates the model optimizer"""

    # Default parameters for Adam
    lr = cfg.get('optimizer', {}).get('learning_rate', 1e-3)
    weight_decay = cfg.get('optimizer', {}).get('weight_decay', 5e-4)

    # Create Adam optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    return optimizer


def setup_scheduler(cfg, optimizer):
    """Creates the model learning rate scheduler

    Args:
        cfg: Configuration dictionary
        optimizer: The optimizer to schedule

    Returns:
        The learning rate scheduler or None if not configured
    """
    if not cfg.get('scheduler', {}).get('use_scheduler', True):
        return None

    scheduler_type = cfg.get('scheduler', {}).get('type', 'step')

    if scheduler_type == 'step':
        step_size = cfg.get('scheduler', {}).get('step_size', 30)
        gamma = cfg.get('scheduler', {}).get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        t_max = cfg.get('scheduler', {}).get('t_max', 100)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max)
    else:
        return None
