from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

import h5py
import json
import numpy as np
import os
import torch
import torchvision.transforms as transforms


class BibNetDataset(Dataset):
    """Dataset for bib number detection using COCO format annotations."""

    def __init__(self, data_dir, mode="train", transform=None, force_reload=False):
        """
        Initialize the BibNetDataset.

        Args:
            data_dir (str): Path to the root data directory
            mode (str): One of 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on images
            force_reload (bool): If True, regenerate the H5 file even if it exists
        """
        self.data_dir = Path(data_dir)
        self.mode = mode

        # Use custom transform if provided, otherwise create a default transform
        if transform is not None:
            self.transform = transform
        else:
            # Check if dataset_stats.json exists
            stats_file = self.data_dir / "dataset_stats.json"

            if os.path.exists(stats_file):
                # Load statistics and apply normalization
                with open(stats_file, 'r') as f:
                    stats = json.load(f)

                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=stats["mean"], std=stats["std"])
                ])
                print(f"Using dataset-specific normalization from {stats_file}")
            else:
                # No normalization if stats file doesn't exist
                self.transform = transforms.Compose([
                    transforms.ToTensor()
                ])
                print("No dataset_stats.json found. Using ToTensor without normalization.")

        self.dataset_dir = self.data_dir / "dataset" / "v1" / mode
        self.h5_file_path = self.data_dir / f"bib_{mode}.h5"

        # Load or create H5 file
        if not os.path.exists(self.h5_file_path) or force_reload:
            self._create_h5_dataset()

        with h5py.File(self.h5_file_path, 'r') as h5f:
            self.num_samples = h5f["images"].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as h5f:
            # Load image
            image = torch.from_numpy(h5f["images"][idx])

            # Load bounding boxes and labels
            bboxes = torch.from_numpy(h5f["bboxes"][idx])
            labels = torch.from_numpy(h5f["labels"][idx])

            # Get number of valid boxes
            num_boxes = int(h5f["num_boxes"][idx])

            # Filter out padding
            if num_boxes > 0:
                bboxes = bboxes[:num_boxes]
                labels = labels[:num_boxes]

            # Create target dict
            target = {
                "boxes": bboxes,
                "labels": labels,
                "image_id": torch.tensor([idx]),
                "orig_size": torch.tensor(h5f["orig_sizes"][idx]),
            }

            return image, target

    def _create_h5_dataset(self):
        """Create an H5 file with processed images and annotations."""
        print(f"Creating H5 dataset for {self.mode} mode...")

        annotation_file = self.dataset_dir / "_annotations.coco.json"
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # mapping from image_id to file_name
        image_id_to_file = {img["id"]: img["file_name"]
                            for img in coco_data["images"]}

        # Group annotations by image_id
        annotations_by_image = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)

        # Calculate max number of boxes per image for padding
        max_boxes = max([len(anns) for anns in annotations_by_image.values(
        )]) if annotations_by_image else 0

        # Get all image IDs
        image_ids = [img["id"] for img in coco_data["images"]]
        num_images = len(image_ids)

        with h5py.File(self.h5_file_path, 'w') as h5f:
            # Create datasets
            image_dset = h5f.create_dataset(
                "images", (num_images, 3, 640, 640), dtype=np.float32)
            bbox_dset = h5f.create_dataset(
                "bboxes", (num_images, max_boxes, 4), dtype=np.float32)
            label_dset = h5f.create_dataset(
                "labels", (num_images, max_boxes), dtype=np.int64)
            num_boxes_dset = h5f.create_dataset(
                "num_boxes", (num_images,), dtype=np.int32)
            orig_sizes_dset = h5f.create_dataset(
                "orig_sizes", (num_images, 2), dtype=np.int32)

            # Process each (image, annotations) pairs
            for idx, image_id in enumerate(tqdm(image_ids, desc=f"Processing {self.mode} images")):
                image_filename = image_id_to_file[image_id]
                image_path = self.dataset_dir / image_filename

                img = Image.open(image_path).convert("RGB")
                img_tensor = self.transform(img)
                image_dset[idx] = img_tensor.numpy()

                width, height = img.size
                orig_sizes_dset[idx] = np.array([height, width])

                anns = annotations_by_image.get(image_id, [])
                num_boxes_dset[idx] = len(anns)

                # Fill in bounding boxes and labels
                boxes = np.zeros((max_boxes, 4), dtype=np.float32)
                labels = np.zeros(max_boxes, dtype=np.int64)
                for box_idx, ann in enumerate(anns):
                    # COCO bbox format is [x, y, width, height]
                    # Convert to [x1, y1, x2, y2] format
                    x, y, w, h = ann["bbox"]
                    boxes[box_idx] = [x, y, x + w, y + h]
                    labels[box_idx] = ann["category_id"]

                bbox_dset[idx] = boxes
                label_dset[idx] = labels

        print(f"Successfully created H5 dataset at {self.h5_file_path}")

    @staticmethod
    def calculate_dataset_stats(data_dir, mode="train", batch_size=32):
        """
        Calculate the mean and standard deviation of the dataset.

        Args:
            data_dir (str): Path to the root data directory
            mode (str): One of 'train', 'val', or 'test'
            batch_size (int): Batch size for calculation

        Returns:
            tuple: (mean, std) tensors with channel-wise statistics
        """
        dataset = BibNetDataset(
            data_dir=data_dir,
            mode=mode,
            transform=transforms.Compose([transforms.ToTensor()]),
            force_reload=False
        )

        from torch.utils.data import DataLoader

        # Custom collate function that only collects the images and ignores targets
        def collate_fn(batch):
            images = torch.stack([item[0] for item in batch])
            return images, None

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)

        # Initialize channels sum and squared sum
        channels_sum = torch.zeros(3)
        channels_squared_sum = torch.zeros(3)
        num_batches = 0

        # Calculate running statistics
        print("Calculating dataset statistics...")
        for images, _ in tqdm(loader):
            # Images shape: [batch_size, 3, height, width]
            channels_sum += torch.mean(images, dim=[0, 2, 3]) * images.size(0)
            channels_squared_sum += torch.mean(images**2, dim=[0, 2, 3]) * images.size(0)
            num_batches += images.size(0)

        mean = channels_sum / num_batches
        std = torch.sqrt(channels_squared_sum / num_batches - mean**2)

        print(f"Dataset {mode} statistics:")
        print(f"Mean: {mean.tolist()}")
        print(f"Std: {std.tolist()}")

        return mean, std
