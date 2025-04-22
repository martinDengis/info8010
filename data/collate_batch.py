import torch

def collate_fn(batch):
    """
    Custom collate function to handle variable-sized targets.

    Args:
        batch: List of (image, target) tuples

    Returns:
        Tuple of (images, targets) where images is a tensor and targets is a list
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images, dim=0)

    return images, targets


def collate_fn_loaders(batch):
    """Custom collate function that only collects the images and ignores targets"""
    images = torch.stack([item[0] for item in batch])
    return images, None