from data import get_data_loaders
from engine import *
from models import *
from pathlib import Path
import os


def train(cfg):
    # Determine which model to build based on the model type in config
    model_type = cfg.get('model', {}).get('type', 'bibnet')
    
    if model_type == 'bibnet':
        model = build_bibnet(cfg)
    elif model_type == 'bibc3net':
        model = build_bibc3net(cfg)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'bibnet' or 'bibc3net'.")

    data_dir = os.path.join(Path(__file__).parent.parent, 'data')
    batch_size = cfg.get('training', {}).get('batch_size', 16)
    train_loader, val_loader, _ = get_data_loaders(data_dir, batch_size)

    optimizer = build_optimizer(cfg, model)
    scheduler = setup_scheduler(cfg, optimizer)

    # Get loss weights from config, with defaults
    loss_cfg = cfg.get('loss', {})
    ciou_weight = loss_cfg.get('ciou_weight', 1.0)
    l1_weight = loss_cfg.get('l1_weight', 0.5)

    loss_fn = BboxLoss(
        ciou_weight=ciou_weight,
        l1_weight=l1_weight,
    )

    do_train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
    )
