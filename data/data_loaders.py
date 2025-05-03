from torch.utils.data import DataLoader
from pathlib import Path
from data.bibnet_dataset import BibNetDataset
from data.transform.build import build_train_transforms, build_test_transforms


def build_dataset(data_dir, mode, force_reload=False, transform=None):
    """
    Build a BibNetDataset for the specified mode.

    Args:
        data_dir (str): Path to the data directory
        mode (str): Dataset split ('train', 'valid', or 'test')
        force_reload (bool): Whether to force reload the dataset
        transform (callable, optional): Transform to apply to the dataset

    Returns:
        BibNetDataset: The built dataset
    """
    print(f"Building {mode} dataset...")
    dataset = BibNetDataset(
        mode=mode,
        split_size=7,
        num_boxes=2,
        num_classes=1,
        transform=transform,
        data_dir=data_dir,
        force_reload=force_reload,
    )

    print(f"Built {mode} dataset with {len(dataset)} samples.")

    return dataset


def get_data_loaders(data_dir, batch_size):
    """Create data loaders for training, validation, and testing datasets."""
    data_dir = Path(data_dir)

    # Define the transforms (need to be defined here for DataLoader)
    train_transform = build_train_transforms(data_dir=data_dir)
    test_transform = build_test_transforms(data_dir=data_dir)

    # Create or get datasets
    train_dataset = build_dataset(
        data_dir,
        mode="train",
        transform=train_transform
    )
    val_dataset = build_dataset(
        data_dir,
        mode="valid",
        transform=test_transform
    )
    test_dataset = build_dataset(
        data_dir,
        mode="test",
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    return train_loader, val_loader, test_loader
