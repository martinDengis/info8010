from torch.optim.lr_scheduler import LinearLR, SequentialLR
import torch.optim as optim


def build_optimizer(cfg, model):
    """Creates the model optimizer"""

    # Default parameters for Adam
    optimizer_cfg = cfg.get('optimizer', {})
    lr = optimizer_cfg.get('learning_rate', 1e-3)
    weight_decay = optimizer_cfg.get('weight_decay', 5e-4)

    # Create optimizer
    optim_type = optimizer_cfg.get('type', 'adam')
    if optim_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optim_type == 'radam':
        optimizer = optim.RAdam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optim_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")

    return optimizer


def setup_scheduler(cfg, optimizer):
    """Creates the model learning rate scheduler with optional warmup"""
    scheduler_cfg = cfg.get('scheduler', {})
    if not scheduler_cfg.get('use_scheduler', True):
        return None

    # Init scheduler parameters
    scheduler_type = scheduler_cfg.get('type', 'step')
    use_warmup = scheduler_cfg.get('use_warmup', False)
    warmup_epochs = scheduler_cfg.get('warmup_epochs', 5)

    # main scheduler
    main_scheduler = None
    if scheduler_type == 'step':
        step_size = scheduler_cfg.get('step_size', 30)
        gamma = scheduler_cfg.get('gamma', 0.1)
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        t_max = scheduler_cfg.get('t_max', 100)
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max)
    else:
        return None

    if not use_warmup:
        return main_scheduler

    # warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,  # Start at 10% of the initial learning rate
        end_factor=1.0,    # End at 100% of the initial learning rate
        total_iters=warmup_epochs
    )

    # combine schedulers
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
