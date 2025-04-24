import os
import json
from pathlib import Path
import torchvision.transforms as T
from .transforms import AutoOrient, ResizeWithPadding, AutoContrast, Cutout, AddGaussianNoise


def build_transforms(cfg=None, is_train=True, data_dir=None):
    """
    Build transformation pipeline for bib number detection.

    Args:
        cfg (dict, optional): Configuration dictionary.
        is_train (bool): Whether to build training transforms (with augmentations).
        data_dir (str, optional): Path to data directory for loading stats.

    Returns:
        callable: A transformation pipeline.
    """
    transform_list = []

    # ----- Image Preprocessing -----
    # Auto-orient, auto-contrast and resize with padding
    transform_list.append(AutoOrient())
    transform_list.append(AutoContrast())

    target_size = (512, 512)  # Default size
    if cfg and 'input' in cfg and 'size' in cfg['input']:
        target_size = cfg['input']['size']
    transform_list.append(ResizeWithPadding(target_size))

    # ----- Data Augmentation (for training) -----
    """
    - RandomHorizontalFlip: Randomly flip the image horizontally.
    - ColorJitter: Randomly change brightness, contrast, and saturation.
    - RandomAdjustSharpness: Randomly adjust the sharpness of the image.
    - RandomAffine: Randomly apply affine transformations (rotation + translation) to force the model not to recognize the bib number only in the center of the image.
    - RandomPerspective: Randomly apply perspective transformations to simulate different camera angles.
    - RandomApply: Randomly apply Gaussian blur to simulate motion or focus issues.
    - RandomGrayscale: Occasionally convert the image to grayscale to improve robustness to color variation.
    - Cutout: Randomly occlude parts of the image to simulate occlusion (arms, other runners, etc.).
    - AddGaussianNoise: Add Gaussian noise to the image to simulate sensor noise.
    """
    if is_train:
        transform_list.extend([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            # T.RandomGrayscale(p=0.1), # may not be necessary -> test it
            Cutout(n_holes=2, length=(30, 80), p=0.3),
            AddGaussianNoise(mean=0., std=0.1, p=0.3)
        ])

    # Convert to tensor
    transform_list.append(T.ToTensor())

    # Add normalization
    # Default ImageNet normalization as fallback
    imagenet_norm = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if data_dir is not None:
        # Try to load dataset-specific normalization
        stats_file = Path(data_dir) / "dataset" / "dataset_stats.json"

        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            mean, std = stats["mean"], stats["std"]
            transform_list.append(T.Normalize(mean=mean, std=std))
            print(f"Using dataset-specific normalization from {stats_file}")
        else:
            transform_list.append(imagenet_norm)
            print("No dataset_stats.json found. Using ImageNet normalization.")
    else:
        transform_list.append(imagenet_norm)

    return T.Compose(transform_list)


def build_train_transforms(cfg=None, data_dir=None):
    """Build training transforms with data augmentation."""
    return build_transforms(cfg, is_train=True, data_dir=data_dir)


def build_test_transforms(cfg=None, data_dir=None):
    """Build test transforms without data augmentation."""
    return build_transforms(cfg, is_train=False, data_dir=data_dir)
