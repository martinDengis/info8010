from pathlib import Path
import h5py
import json
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.io import decode_image
from data.transform.build import build_train_transforms, build_test_transforms


class BibNetDataset(Dataset):
    def __init__(self, mode="train", split_size=7, num_boxes=2, num_classes=1, transform=None, data_dir=None, force_reload=False, stats_mode=False):
        super().__init__()
        self.mode = mode
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.stats_mode = stats_mode

        # Set data directory
        if data_dir is None:
            self.root_data_dir = Path("data")
        else:
            self.root_data_dir = Path(data_dir)

        self.dataset_dir = self.root_data_dir / "dataset" / mode
        self.data_dir = self.root_data_dir / 'dataset' / f'bib_{mode}.h5'

        # Set transform
        self.transform = transform
        if self.transform is None:
            if mode == "train":
                self.transform = build_train_transforms(data_dir=self.root_data_dir)
            else:
                self.transform = build_test_transforms(data_dir=self.root_data_dir)

        # Load or create H5 file
        if not os.path.exists(self.data_dir) or force_reload:
            self._create_h5_dataset()

        with h5py.File(self.data_dir, 'r') as h5f:
            self.num_samples = h5f["images"].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.data_dir, 'r') as h5f:
            # Load image
            image = torch.tensor(h5f["images"][idx], dtype=torch.float32)   # shape is [C, H, W]
            img_height, img_width = image.shape[-2:]

            # Load bounding boxes and labels
            bboxes = torch.from_numpy(h5f["bboxes"][idx])
            labels = torch.from_numpy(h5f["labels"][idx])

            # Get number of valid boxes
            num_boxes = int(h5f["num_boxes"][idx])

            # Filter out padding
            if num_boxes > 0:
                bboxes = bboxes[:num_boxes]
                labels = labels[:num_boxes]

                # Convert bboxes to tv_tensors.BoundingBoxes
                bboxes = tv_tensors.BoundingBoxes(
                    bboxes,
                    format="xywh",
                    canvas_size=(img_height, img_width)
                )

            # Apply transformations
            if self.transform is not None and not self.stats_mode:
                image, bboxes = self.transform(image, bboxes)

            img_height, img_width = image.shape[-2:] # updated after transform

            label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
            for box_idx, box in enumerate(bboxes):
                class_label = int(labels[box_idx].item() - 1)  # Convert to 0-based index
                x, y, width, height = box

                # Convert from COCO top-left format to midpoint format
                x += width / 2
                y += height / 2

                # Normalize x, y, width, height to [0, 1]
                x /= img_width
                y /= img_height
                width /= img_width
                height /= img_height

                # i,j represents the cell row and cell column
                i, j = int(self.S * y), int(self.S * x)
                x_cell, y_cell = self.S * x - j, self.S * y - i

                # Edge case on i,j
                i = min(i, self.S - 1)
                j = min(j, self.S - 1)

                """
                Calculating the width and height of cell of bounding box,
                relative to the cell is done by the following, with
                width as the example:

                width_pixels = (width*self.image_width)
                cell_pixels = (self.image_width)

                Then to find the width relative to the cell is simply:
                width_pixels/cell_pixels, simplification leads to the
                formulas below.
                """
                width_cell, height_cell = (
                    width * self.S,
                    height * self.S,
                )

                # If no object already found for specific cell i,j
                # Note: This means we restrict to ONE object
                # per cell!
                if label_matrix[i, j, self.C] == 0:
                    # Set that there exists an object
                    label_matrix[i, j, self.C] = 1

                    # Box coordinates
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    label_matrix[i, j, self.C+1:self.C+5] = box_coordinates

                    # Set one hot encoding for class_label
                    label_matrix[i, j, class_label] = 1

            return image, label_matrix

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
        max_boxes = max([len(anns) for anns in annotations_by_image.values()
                        ]) if annotations_by_image else 0

        # Get all image IDs
        image_ids = [img["id"] for img in coco_data["images"]]
        num_images = len(image_ids)

        with h5py.File(self.data_dir, 'w') as h5f:
            # Create datasets
            image_dset = h5f.create_dataset(
                "images", (num_images, 3, 512, 512), dtype=np.float32)
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
                anns = annotations_by_image.get(image_id, [])
                if not anns:    # if no annotations for this image, skip it
                    continue

                image_filename = image_id_to_file[image_id]
                image_path = self.dataset_dir / image_filename

                # Decode image and apply transform
                org_img = decode_image(
                    str(image_path), mode="RGB")  # shape = [C, H, W]
                img = self.transform(org_img)
                image_dset[idx] = img

                height, width = org_img.shape[-2:]
                orig_sizes_dset[idx] = np.array([height, width])
                num_boxes_dset[idx] = len(anns)

                # Fill in bounding boxes and labels
                bboxes = np.zeros((max_boxes, 4), dtype=np.float32)
                labels = np.zeros(max_boxes, dtype=np.int64)
                for box_idx, ann in enumerate(anns):
                    # COCO bbox format is [x, y, width, height]
                    x, y, w, h = ann["bbox"]

                    # Resize the bbox the same way as the image
                    box = tv_tensors.BoundingBoxes(torch.tensor(
                        [x, y, w, h]), format="xywh", canvas_size=(width, height))
                    box = self.transform(box)  # Resize the bounding box

                    # Save the resized bounding box in xywh format
                    np_box = box.numpy()
                    bboxes[box_idx] = np_box
                    labels[box_idx] = ann["category_id"]

                bbox_dset[idx] = bboxes
                label_dset[idx] = labels

        print(f"Successfully created H5 dataset at {self.data_dir}")

    @staticmethod
    def calculate_dataset_stats(data_dir, mode="train", batch_size=32):
        """
        Calculate the mean and standard deviation of the dataset.

        Args:
            data_dir (str): Path to the root data directory
            mode (str): One of 'train', 'valid', or 'test'
            batch_size (int): Batch size for calculation

        Returns:
            tuple: (mean, std) tensors with channel-wise statistics
        """
        dataset = BibNetDataset(
            mode=mode,
            split_size=7,
            num_boxes=2,
            num_classes=1,
            transform=v2.Compose([  # essentially no transform
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
            ]),
            data_dir=data_dir,
            force_reload=False,
            stats_mode=True
        )
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Initialize channels sum and squared sum
        channels_sum = torch.zeros(3)
        channels_squared_sum = torch.zeros(3)
        num_batches = 0

        # Calculate running statistics
        for images, _ in tqdm(loader, desc=f"Processing {mode} batches"):
            # Images shape: [batch_size, 3, height, width]
            channels_sum += torch.mean(images, dim=[0, 2, 3]) * images.size(0)
            channels_squared_sum += torch.mean(images ** 2, dim=[0, 2, 3]) * images.size(0)
            num_batches += images.size(0)

        mean = channels_sum / num_batches
        std = torch.sqrt(channels_squared_sum / num_batches - mean**2)

        print(f"Dataset {mode} statistics:")
        print(f"Mean: {mean.tolist()}")
        print(f"Std: {std.tolist()}")

        return mean, std

