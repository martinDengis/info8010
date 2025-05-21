"""
DataLoader module for the bib number detection dataset.
"""
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import BibNumberDataset


def coco_to_yolo(bbox, img_size):
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    return [    # clipping as safe guard
        max(0.0, min(1.0, x_center / img_size)),
        max(0.0, min(1.0, y_center / img_size)),
        max(0.0, min(1.0, width / img_size)),
        max(0.0, min(1.0, height / img_size))
    ]


class TransformWrapper:
    """
    Wrapper class for albumentations transforms to make them compatible with PyTorch datasets.
    """

    def __init__(self, transform, img_size=416):
        self.transform = transform
        self.img_size = img_size

    def __call__(self, img, target):
        # Convert PIL image to numpy array
        # Img has shape (H, W, C) and is in [0, 255] range
        img_np = np.array(img)

        # Get bounding boxes
        boxes = target["boxes"].numpy() if len(
            target["boxes"]) > 0 else np.zeros((0, 4))

        # Pre-process boxes to ensure they're valid for albumentations
        if len(boxes) > 0:
            # For COCO format: x_min, y_min, width, height
            img_width, img_height = img_np.shape[1], img_np.shape[0]

            # Ensure coordinates are within image bounds
            boxes[:, 0] = np.clip(boxes[:, 0], 0, img_width - 1e-5)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, img_height - 1e-5)

            # Ensure width and height don't go outside image
            # and stating point thereof are newly calculated x_min, y_min
            boxes[:, 2] = np.clip(boxes[:, 2], 0, img_width - boxes[:, 0])
            boxes[:, 3] = np.clip(boxes[:, 3], 0, img_height - boxes[:, 1])

        # Apply transformations
        transformed = self.transform(
            image=img_np, bboxes=boxes, labels=target["labels"].numpy())

        # Deconstruct transformed
        img_tensor = transformed["image"]
        boxes = transformed["bboxes"]
        labels = torch.tensor(transformed["labels"], dtype=torch.int64) if len(transformed["labels"]) > 0 else torch.zeros(0, dtype=torch.int64)

        # Convert COCO format [x_min, y_min, width, height]
        # to YOLO format [x_center, y_center, width, height],
        # and normalize to [0, 1] range
        if len(boxes) > 0:
            yolo_boxes = np.array([coco_to_yolo(bbox, self.img_size) for bbox in boxes])
            boxes = torch.tensor(yolo_boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        # Update target
        target["boxes"] = boxes
        target["labels"] = labels

        return img_tensor, target


def get_transforms(img_size=416, is_train=True, mean=None, std=None):
    """
    Get image and bounding box transformations.

    Args:
        img_size (int): Size to resize images to
        is_train (bool): Whether to apply training augmentations

    Returns:
        TransformWrapper: Transform wrapper instance
    """
    # ImageNet fallback mean and std
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        # Check if mean and std are already in [0,1] range
        if np.any(mean > 1.0) or np.any(std > 1.0):
            mean = mean / 255.0
            std = std / 255.0

    # NB: BboxParams format is 'coco' from the dataset
    bbox_params = A.BboxParams(
        format='coco',
        label_fields=['labels'],
        check_each_transform=True,  # Check after each transform
        min_area=0.0,  # Allow any box area
        min_visibility=0.0,  # Allow any visibility
    )

    if is_train:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.AtLeastOneBBoxRandomCrop(height=img_size, width=img_size, p=0.3),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
            A.ToGray(p=0.1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], bbox_params=bbox_params)
    else:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], bbox_params=bbox_params)

    return TransformWrapper(transform, img_size)


def create_dataloaders(data_dir, img_size=416, batch_size=8, num_workers=4):
    """
    Create DataLoaders for train, validation and test sets.

    Args:
        data_dir (str): Path to the dataset directory
        img_size (int): Size to resize images to
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    # Create transforms
    mean, std = compute_dataset_stats(img_size=img_size)
    transforms = {
        'train': get_transforms(img_size=img_size, is_train=True, mean=mean, std=std),
        'valid': get_transforms(img_size=img_size, is_train=False, mean=mean, std=std),
        'test': get_transforms(img_size=img_size, is_train=False, mean=mean, std=std),
    }

    # Create datasets
    datasets = {}
    for split in ['train', 'valid', 'test']:
        datasets[split] = BibNumberDataset(
            root_dir=data_dir,
            split=split,
            transform=transforms[split],
            img_size=img_size
        )

    # Create dataloaders
    loaders = {}
    for split in ['train', 'valid', 'test']:
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return loaders['train'], loaders['valid'], loaders['test']


def collate_fn(batch):
    """
    Custom collate function for batching samples with variable number of objects.

    Args:
        batch (list): List of (image, target) tuples

    Returns:
        tuple: (images, targets) where images is a tensor of shape [batch_size, C, H, W]
               and targets is a list of dictionaries
    """
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def compute_dataset_stats(data_dir='data', img_size=416, batch_size=16, num_workers=4):
    """
    Compute mean and standard deviation of the dataset.

    Args:
        data_dir (str): Path to the dataset directory
        img_size (int): Size to resize images to
        batch_size (int): Batch size for data loading
        num_workers (int): Number of workers for data loading

    Returns:
        tuple: (mean, std) as numpy arrays with shape [3]
    """
    # Create a simplified transform without normalization
    bbox_params = A.BboxParams(
        format='coco',
        label_fields=['labels'],
        check_each_transform=True,
        min_area=0.0,
        min_visibility=0.0,
    )

    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        ToTensorV2(),
    ], bbox_params=bbox_params)

    transform_wrapper = TransformWrapper(transform, img_size)

    # Create dataset
    dataset = BibNumberDataset(
        root_dir=data_dir,
        split='train',  # Only using training set for stats
        transform=transform_wrapper,
        img_size=img_size
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    # Initialize variables
    pixel_sum = torch.zeros(3)
    pixel_sq_sum = torch.zeros(3)
    num_pixels = 0

    print("Computing dataset statistics...")

    # Iterate through the dataset
    for images, _ in dataloader:
        # Convert to float32 for better precision
        images = images.float()
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)

        # Sum over H*W dimensions
        pixel_sum += images.sum(dim=[0, 2])
        pixel_sq_sum += (images ** 2).sum(dim=[0, 2])
        num_pixels += images.size(0) * images.size(2)

    # Calculate mean and std
    mean = pixel_sum / num_pixels
    var = torch.maximum(pixel_sq_sum / num_pixels -
                        (mean ** 2), torch.tensor(1e-6))
    std = torch.sqrt(var)

    # Convert to numpy for easier handling
    mean_np = mean.numpy()
    std_np = std.numpy()

    print(f"Dataset mean: {mean_np}")
    print(f"Dataset std: {std_np}")

    return mean_np, std_np
