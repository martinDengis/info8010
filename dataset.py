"""
Dataset class for loading COCO-formatted bib number detection dataset.
"""
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class BibNumberDataset(Dataset):
    """
    Dataset class for loading COCO-formatted bib number detection dataset.

    Args:
        root_dir (str): Path to the dataset root directory
        split (str): Dataset split ('train', 'valid', or 'test')
        transform (callable, optional): Optional transform to be applied to the images
        img_size (int): Size to resize images to (square)
    """
    def __init__(self, root_dir, split="train", transform=None, img_size=416):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size

        # Load annotations
        self.annotations_file = os.path.join(root_dir, split, "_annotations.coco.json")
        with open(self.annotations_file, 'r') as f:
            self.coco_data = json.load(f)

        # Extract images and annotations
        self.images = self.coco_data["images"]
        self.annotations = self.coco_data["annotations"]

        # Create image_id to annotations mapping
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            # Only use category_id 1 (bib-number)
            if ann["category_id"] == 1:
                self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image info
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_file = img_info["file_name"]
        img_path = os.path.join(self.root_dir, self.split, img_file)

        # Load image
        image = Image.open(img_path).convert("RGB") # shape: (H, W, C)
        orig_width, orig_height = image.size

        # Create target
        target = {
            "boxes": [],
            "labels": [],
            "image_id": torch.tensor([img_id]),
            "orig_size": torch.tensor([orig_height, orig_width]),
        }

        # Load annotations for this image
        if img_id in self.img_to_anns:
            for ann in self.img_to_anns[img_id]:
                # Get bounding box in COCO format [x, y, width, height]
                # Conversion to YOLO format [x_center, y_center, width, height] and normalization is done in the transform (i.e., dataloader-level)
                box = ann["bbox"]
                target["boxes"].append(box)
                target["labels"].append(1)  # Class 1 for bib-number; not really used anywhere but keep it for reference

        # Convert to tensors
        if len(target["boxes"]) > 0:
            target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float32)
            target["labels"] = torch.tensor(target["labels"], dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros(0, dtype=torch.int64)

        # Apply transformations
        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def get_img_height_and_width(self, idx):
        """Get height and width of an image."""
        img_info = self.images[idx]
        return img_info["height"], img_info["width"]
