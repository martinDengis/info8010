import numbers
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import tv_tensors
from torchvision.tv_tensors import BoundingBoxes
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F, Transform


# class that inherits from Transform
# and pads the image to a square but keeps the aspect ratio
class ResizeWithPadding(Transform):
    def __init__(self, size, fill=0, padding_mode="constant"):
        super().__init__()
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def _get_params(self, flat_inputs):
        return {}

    def forward(self, *inputs):
        if len(inputs) == 1:
            # If only one input is provided, check if img or boxes
            if isinstance(inputs[0], tv_tensors.BoundingBoxes):
                return self._forward_boxes(inputs[0])
            elif isinstance(inputs[0], (torch.Tensor, Image.Image)):
                return self._forward_img(inputs[0])
            else:
                raise ValueError(f"Unsupported input type: {type(inputs[0])}")
        else:
            img, boxes = inputs
            return self._forward_img(img), self._forward_boxes(boxes)

    def _forward_img(self, img):
        # Get the original size of the image
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
            device = img.device
        else:  # PIL Image
            w, h = img.size
            device = None

        # Calculate the new size while keeping the aspect ratio
        if isinstance(self.size, (list, tuple)):
            target_size = (self.size[0], self.size[1])
        else:
            target_size = (self.size, self.size)

        # Calculate the scale factor to resize the image
        scale_factor = min(target_size[1] / h, target_size[0] / w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # Calculate the padding offsets
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2

        # Store these for the bounding box transformation
        self.scale_factor = scale_factor
        self.offsets = (x_offset, y_offset)

        if isinstance(img, torch.Tensor):
            # Resize the tensor
            img_resized = F.resize(img, [new_h, new_w], antialias=True)
            # Pad the tensor
            padding = [x_offset, y_offset, target_size[0] - new_w - x_offset, target_size[1] - new_h - y_offset]
            return F.pad(img_resized, padding, fill=self.fill, padding_mode=self.padding_mode)
        else:
            # Resize the PIL image
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)
            # Create a new image with the target size and fill with the specified color
            img_padded = Image.new(img.mode, target_size, self.fill)
            # Paste the resized image into the padded image
            img_padded.paste(img_resized, (x_offset, y_offset))
            return img_padded

    def _forward_boxes(self, boxes):
        # Convert to float for safe math
        data = boxes.as_subclass(torch.Tensor).float()

        if boxes.format == tv_tensors.BoundingBoxFormat.XYWH:
            x, y, w, h = data.unbind(-1)
            x = x * self.scale_factor + self.offsets[0]
            y = y * self.scale_factor + self.offsets[1]
            w = w * self.scale_factor
            h = h * self.scale_factor
            new_data = torch.stack([x, y, w, h], dim=-1)
        else:  # XYXY
            x0, y0, x1, y1 = data.unbind(-1)
            x0 = x0 * self.scale_factor + self.offsets[0]
            y0 = y0 * self.scale_factor + self.offsets[1]
            x1 = x1 * self.scale_factor + self.offsets[0]
            y1 = y1 * self.scale_factor + self.offsets[1]
            new_data = torch.stack([x0, y0, x1, y1], dim=-1)

        # Rewrap with new canvas size and original format
        if isinstance(self.size, (list, tuple)):
            canvas_size = (self.size[0], self.size[1])  # (H, W)
        else:
            canvas_size = (self.size, self.size)

        return BoundingBoxes(new_data, format=boxes.format, canvas_size=canvas_size)
