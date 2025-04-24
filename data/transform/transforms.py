import torch
from PIL import Image, ImageOps
import torchvision.transforms.functional as F


class AutoOrient(object):
    """
    Auto-orient image based on EXIF data.
    """

    def __call__(self, img):
        if isinstance(img, Image.Image):
            return ImageOps.exif_transpose(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AutoContrast(object):
    """
    Apply automatic contrast enhancement to the image.

    Args:
        cutoff (float): Percentage of pixels to cut off from histogram. Default is 0.
        ignore (list): List of pixel values to ignore. Default is None.
    """

    def __init__(self, cutoff=0, ignore=None):
        self.cutoff = cutoff
        self.ignore = ignore

    def __call__(self, img):
        if isinstance(img, Image.Image):
            return ImageOps.autocontrast(img, cutoff=self.cutoff, ignore=self.ignore)
        return img

    def __repr__(self):
        return self.__class__.__name__ + f'(cutoff={self.cutoff}, ignore={self.ignore})'


class ResizeWithPadding(object):
    """
    Resize the input image to the specified size while maintaining aspect ratio,
    padding with black (zeros) as needed.

    Args:
        size (tuple): Target size (height, width)
        fill (int or tuple): Fill value for padding. Default is 0 (black).
        padding_mode (str): Type of padding. Default is 'constant'.
    """

    def __init__(self, size, fill=0, padding_mode='constant'):
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            return img

        width, height = img.size
        target_height, target_width = self.size

        # Calculate scaling factor
        scale = min(target_height / height, target_width / width)
        new_height, new_width = int(height * scale), int(width * scale)

        # Resize while preserving aspect ratio
        resized_img = img.resize((new_width, new_height), Image.BILINEAR)

        # Create a new black image with the target size
        padded_img = Image.new('RGB', (target_width, target_height), self.fill)

        # Paste the resized image onto the padded image, centered
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        padded_img.paste(resized_img, (paste_x, paste_y))

        return padded_img

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.size}, fill={self.fill}, padding_mode={self.padding_mode})'


class AddGaussianNoise(object):
    """
    Add Gaussian noise to image.

    Args:
        mean (float): Mean of the Gaussian noise
        std (float): Standard deviation of the Gaussian noise
        p (float): Probability of applying this transform
    """

    def __init__(self, mean=0., std=1., p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            if isinstance(img, torch.Tensor):
                noise = torch.randn_like(img) * self.std + self.mean
                return torch.clamp(img + noise, 0, 1)
            else:
                # convert to tensor, add noise, convert back
                img_tensor = F.to_tensor(img)
                noise = torch.randn_like(img_tensor) * self.std + self.mean
                noisy_img = torch.clamp(img_tensor + noise, 0, 1)
                return F.to_pil_image(noisy_img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std}, p={self.p})'


class Cutout(object):
    """
    Apply cutout augmentation to images.

    This creates random rectangular masks on the image, simulating occlusions that
    might occur in race photos (e.g., when parts of bibs are covered by arms, other
    runners, or race elements).

    Args:
        n_holes (int): Number of holes/patches to cut out
        length (int or tuple): Length of the holes. If tuple (min_length, max_length),
                              randomly samples length for each hole.
        p (float): Probability of applying cutout
    """

    def __init__(self, n_holes=1, length=50, p=0.5):
        self.n_holes = n_holes
        self.length = length if isinstance(length, tuple) else (length, length)
        self.p = p

    def __call__(self, img):
        if torch.rand(1) >= self.p:
            return img

        if isinstance(img, Image.Image):
            img_tensor = F.to_tensor(img)
            h, w = img_tensor.shape[1], img_tensor.shape[2]
            mask = torch.ones_like(img_tensor)

            for _ in range(self.n_holes):
                # Sample random length if it's a tuple
                if isinstance(self.length, tuple):
                    length = torch.randint(
                        self.length[0], self.length[1] + 1, (1,)).item()
                else:
                    length = self.length

                # Random pos
                y = torch.randint(0, h, (1,))
                x = torch.randint(0, w, (1,))

                # Calculate box boundaries
                y1 = torch.clamp(y - length // 2, 0, h)
                y2 = torch.clamp(y + length // 2, 0, h)
                x1 = torch.clamp(x - length // 2, 0, w)
                x2 = torch.clamp(x + length // 2, 0, w)

                # Apply mask
                mask[:, y1:y2, x1:x2] = 0

            # Apply the mask
            img_tensor = img_tensor * mask

            return F.to_pil_image(img_tensor)
        elif isinstance(img, torch.Tensor):
            h, w = img.shape[1], img.shape[2]
            mask = torch.ones_like(img)

            for _ in range(self.n_holes):
                # Sample random length
                if isinstance(self.length, tuple):
                    length = torch.randint(
                        self.length[0], self.length[1] + 1, (1,)).item()
                else:
                    length = self.length

                # Random position
                y = torch.randint(0, h, (1,))
                x = torch.randint(0, w, (1,))

                # Calculate box boundaries
                y1 = torch.clamp(y - length // 2, 0, h)
                y2 = torch.clamp(y + length // 2, 0, h)
                x1 = torch.clamp(x - length // 2, 0, w)
                x2 = torch.clamp(x + length // 2, 0, w)

                # Apply mask
                mask[:, y1:y2, x1:x2] = 0

            # Apply the mask
            return img * mask
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + f'(n_holes={self.n_holes}, length={self.length}, p={self.p})'
