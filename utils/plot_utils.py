import torch
import matplotlib.pyplot as plt
from torchvision import tv_tensors
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.v2 import functional as F


def plot_img(imgs, row_title=None, save_path=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            bboxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    bboxes = target.get("bboxes")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    bboxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if bboxes is not None:
                assert isinstance(bboxes, tv_tensors.BoundingBoxes), f"Expected BoundingBoxes, got {type(bboxes)}"
                # Convert bboxes to xyxy format for draw_bounding_boxes
                if bboxes.format == tv_tensors.BoundingBoxFormat.XYWH:
                    bboxes = F.convert_bounding_box_format(bboxes, new_format="xyxy")

                img = draw_bounding_boxes(img, bboxes, colors="yellow", width=3)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
