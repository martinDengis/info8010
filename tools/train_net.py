from data.data_loaders import get_data_loaders
from engine.loss import BboxLoss
from engine.optimizer import build_optimizer, setup_scheduler
from engine.trainer import do_train
from models import build_bibnet, build_yolov1_model
from pathlib import Path
import os


def train(cfg):
    # Determine which model to build based on the model type in config
    model_type = cfg.get('model', {}).get('type', 'bibnet')
    print(model_type)

    if model_type == 'bibnet':
        model = build_bibnet(cfg)
    elif model_type == "yolov1":
        model = build_yolov1_model()
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'bibnet' or 'bibnet'.")

    data_dir = os.path.join(Path(__file__).parent.parent, 'data')
    batch_size = cfg.get('training', {}).get('batch_size', 16)
    train_loader, val_loader, _ = get_data_loaders(data_dir, batch_size)

    optimizer = build_optimizer(cfg, model)
    scheduler = setup_scheduler(cfg, optimizer)

    loss_fn = BboxLoss(split_size=7, num_boxes=2, num_classes=1)

    do_train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
    )
