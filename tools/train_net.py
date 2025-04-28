from data import get_data_loaders
from engine import do_train, build_optimizer, setup_scheduler
from models import build_bibnet, build_bibc3net
from engine import BboxLoss
from pathlib import Path
import os


def train(cfg):
    # Determine which model to build based on the model type in config
    model_type = cfg.get('model', {}).get('type', 'bibc3net')

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

    # Get loss parameters from config, with defaults
    loss_cfg = cfg.get('loss', {})
    lambda_bbox = loss_cfg.get('lambda_bbox', 1.0)
    lambda_conf = loss_cfg.get('lambda_conf', 1.0)
    lambda_coverage = loss_cfg.get('lambda_coverage', 1.0)
    iou_threshold = loss_cfg.get('iou_threshold', 0.1)
    conf_loss_type = loss_cfg.get('conf_loss_type', 'focal')
    focal_alpha = loss_cfg.get('focal_alpha', 0.25)
    focal_gamma = loss_cfg.get('focal_gamma', 2.0)

    loss_fn = BboxLoss(
        lambda_bbox=lambda_bbox,
        lambda_conf=lambda_conf,
        lambda_coverage=lambda_coverage,
        iou_threshold=iou_threshold,
        conf_loss_type=conf_loss_type,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma
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
