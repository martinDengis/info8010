import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pytest
from PIL import Image

# Add the project root to the path so we can import from utils
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import draw_bounding_boxes
from utils.plot_utils import plot_img
from data.transform.build import build_transforms, ComposeWithBBox

def create_sample_image_and_bbox():
    """Create a sample image and bounding box for testing."""
    # Create a 300x400 black image
    image = torch.zeros(3, 300, 400, dtype=torch.uint8)

    # Add colored rectangle in the center (x=150, y=100, w=100, h=80)
    image[:, 100:180, 150:250] = torch.tensor([255, 0, 0]).view(3, 1, 1)

    # Create bounding box in xywh format (center_x, center_y, width, height)
    bbox = torch.tensor([[200, 140, 100, 80]], dtype=torch.float32)

    # Convert to BoundingBoxes format
    bbox = tv_tensors.BoundingBoxes(
        bbox,
        format=tv_tensors.BoundingBoxFormat.XYWH,
        canvas_size=(300, 400)
    )

    return image, bbox

def test_resize_transform():
    """Test if bounding boxes are properly transformed when resizing."""
    image, bbox = create_sample_image_and_bbox()

    # Create a resize transform
    resize_transform = v2.Resize(size=(150, 200), antialias=True)

    # Apply transform to both image and bbox
    transformed_image, transformed_bbox = resize_transform(image, bbox)

    # The bounding box should be scaled by 0.5
    expected_bbox = torch.tensor([[100, 70, 50, 40]], dtype=torch.float32)
    expected_bbox = tv_tensors.BoundingBoxes(
        expected_bbox,
        format=tv_tensors.BoundingBoxFormat.XYWH,
        canvas_size=(150, 200)
    )

    # Verify the transformed bbox
    assert transformed_bbox.shape == expected_bbox.shape
    # Use .as_subclass(torch.Tensor) instead of .tensor
    assert torch.allclose(transformed_bbox.as_subclass(torch.Tensor), expected_bbox.as_subclass(torch.Tensor), atol=1.0)
    assert transformed_bbox.canvas_size == expected_bbox.canvas_size

    # Visualize the results
    plot_img([(image, bbox), (transformed_image, transformed_bbox)],
             row_title=["Original", "Resized"])

def test_random_crop_transform():
    """Test if bounding boxes are properly transformed when applying random crop."""
    image, bbox = create_sample_image_and_bbox()

    # Create a deterministic crop transform for testing
    # Crop the image to (200, 300) starting at (50, 50)
    crop_transform = v2.CenterCrop(size=(200, 300))

    # Apply transform to both image and bbox
    transformed_image, transformed_bbox = crop_transform(image, bbox)

    # Visualize the results
    plot_img([(image, bbox), (transformed_image, transformed_bbox)],
             row_title=["Original", "Cropped"])

    # Verify the canvas size has changed
    assert transformed_bbox.canvas_size == (200, 300)

def test_composite_transforms():
    """Test if bounding boxes are properly transformed when applying multiple transforms."""
    image, bbox = create_sample_image_and_bbox()

    # Create a composite transform
    geo_transforms = [
        v2.Resize(size=(256, 256), antialias=True),
        v2.RandomHorizontalFlip(p=1.0),  # Always flip for testing
    ]

    image_transforms = [
        v2.ColorJitter(brightness=0.2),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) # = v2.ToTensor()
    ]

    transform = ComposeWithBBox(
        geo_transforms=geo_transforms,
        image_transforms=image_transforms
    )

    # Apply transforms
    transformed_image, transformed_bbox = transform(image, bbox)

    # After resize to 256x256 and horizontal flip, the bbox should be flipped
    # Original center_x = 200 in 400px width
    # After resize: 200 * (256/400) = 128
    # After flip: 256 - 128 = 128 (center is preserved)
    # Similarly for width and height: 100 * (256/400) = 64, 80 * (256/300) = ~68

    # Visualize the results (need to handle float tensor for transformed image)
    plot_img([(image, bbox), (transformed_image, transformed_bbox)],
             row_title=["Original", "Composite Transform"])

    # Verify the canvas size has changed
    assert transformed_bbox.canvas_size == (256, 256)

def test_real_transforms_pipeline():
    """Test if build_transforms properly handles bounding boxes."""
    image, bbox = create_sample_image_and_bbox()

    # Get the real transforms used in the project
    transforms = build_transforms(is_train=True)

    # Apply transforms
    transformed_image, transformed_bbox = transforms(image, bbox)

    # Visualize the results
    plot_img([(image, bbox), (transformed_image, transformed_bbox)],
             row_title=["Original", "Training Transforms"])

    # The canvas size should match the size in build_transforms
    assert transformed_bbox.canvas_size[0] <= 512 and transformed_bbox.canvas_size[1] <= 512

if __name__ == "__main__":
    print("Testing bbox transformations...")

    print("\nTest Case 1: Resize Transform")
    test_resize_transform()

    print("\nTest Case 2: Random Crop Transform")
    test_random_crop_transform()

    print("\nTest Case 3: Composite Transforms")
    test_composite_transforms()

    print("\nTest Case 4: Real Transforms Pipeline")
    test_real_transforms_pipeline()

    print("\nAll tests completed!")
