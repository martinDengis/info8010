import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms


def plot_tensor_as_img(t):
    """
    Plot a tensor as an image.

    Args:
        t (torch.Tensor): Input tensor of shape (C, H, W) or (H, W, C)
    """
    if t.dim() == 3:
        if t.size(0) == 3:  # C, H, W
            t = t.permute(1, 2, 0)  # H, W, C
        elif t.size(0) == 1:  # H, W
            t = t.squeeze(0)  # H, W
        else:
            raise ValueError("Tensor must be of shape (C, H, W) or (H, W)")

    plt.imshow(t.numpy())
    plt.axis('off')
    plt.show()


def plot_img_with_boxes(img, boxes, labels=None, save_path=None):
    """
    Plot an image with bounding boxes.

    Args:
        img (PIL.Image or np.ndarray): Input image.
        boxes (list of tuples): List of bounding boxes in the format [(x1, y1, x2, y2), ...].
        labels (list of str, optional): List of labels for each bounding box.
    """
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img)

    plt.imshow(img)
    ax = plt.gca()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if labels is not None:
            ax.text(x1, y1, labels[i], color='white', fontsize=12,
                    bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    return plt.gcf()
