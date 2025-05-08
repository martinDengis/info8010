import os
import json
from pathlib import Path
import torch
from torchvision.transforms import v2
from data.transform.transforms import ResizeWithPadding

class ComposeWithBBox:
    """
    Compose multiple transforms for images and bounding boxes.
    Bounding boxes should get transformed geometrically in the same way as the image.
    Other transforms (e.g., normalization) are only applied to the image.
    """
    def __init__(self, geo_transforms=[], image_transforms=[]):
        """
        Initialize the ComposeWithBBox class.
        Args:
            geo_transforms (list): List of transforms that should be applied to both image and bbox.
            image_transforms (list): List of transforms that should be applied only to the image.
        """
        self.geo_transforms = geo_transforms
        self.image_transforms = v2.Compose(image_transforms)

    def __call__(self, image, target):
        # Apply geometry-based transforms to both image and bbox
        for t in self.geo_transforms:
            image, target = t(image, target)
        # Apply image-only transforms
        image = self.image_transforms(image)
        return image, target


def build_transforms(is_train=True, data_dir=None):
    """
    Build transformation pipeline for bib number detection.

    Args:
        cfg (dict, optional): Configuration dictionary.
        is_train (bool): Whether to build training transforms (with augmentations).
        data_dir (str, optional): Path to data directory for loading stats.

    Returns:
        callable: A transformation pipeline.
    """
    geo_transforms = []
    image_transforms = []

    # ----- Data Augmentation (for training) -----
    if is_train:
        geo_transforms.extend([
            v2.RandomResizedCrop(size=(512, 512), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            v2.RandomPerspective(distortion_scale=0.2, p=0.5),
        ])
        image_transforms.extend([
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            v2.RandomGrayscale(p=0.2),
        ])

    # Convert to tensor
    image_transforms.append(v2.ToImage())
    image_transforms.append(v2.ToDtype(torch.float32, scale=True))

    # Add normalization
    # Default ImageNet normalization as fallback
    imagenet_norm = v2.Normalize(
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
            image_transforms.append(v2.Normalize(mean=mean, std=std))
            print(f"Using dataset-specific normalization from {stats_file}")
        else:
            image_transforms.append(imagenet_norm)
            print("No dataset_stats.json found. Using ImageNet normalization.")
    else:
        image_transforms.append(imagenet_norm)

    transform = ComposeWithBBox(
        geo_transforms=geo_transforms,
        image_transforms=image_transforms
    )
    return transform


def build_train_transforms(data_dir=None):
    """Build training transforms with data augmentation."""
    return build_transforms(is_train=True, data_dir=data_dir)


def build_test_transforms(data_dir=None):
    """Build test transforms without data augmentation."""
    return build_transforms(is_train=False, data_dir=data_dir)
