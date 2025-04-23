from data import get_data_loaders
from engine import *
from models import *
from pathlib import Path
import os


def train(cfg):
    model = build_bibnet(cfg)

    data_dir = os.path.join(Path(__file__).parent.parent, 'data')
    batch_size = cfg.get('batch_size', 16)
    train_loader, val_loader, _ = get_data_loaders(data_dir, batch_size)

    optimizer = build_optimizer(cfg, model)
    scheduler = setup_scheduler(cfg, optimizer)

    loss_fn = BboxLoss(
        ciou_weight=1.0,
        l1_weight=0.5,
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
