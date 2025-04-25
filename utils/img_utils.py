import torch
import numpy as np
from torchvision import tv_tensors
from torchvision.transforms.v2 import ToPILImage

# ----- Tensor to X -----

def img_tensor2np(tensor):
    """From [C, H, W] to [H, W, C] format"""
    return tensor.permute(1, 2, 0).contiguous().cpu().numpy()


def img_tensor2pil(tensor):
    """From [C, H, W] to PIL (width, height) format"""
    return ToPILImage()(tensor.cpu())


# ----- NP to X -----

def img_np2tensor(np_img):
    """From [H, W, C] to [C, H, W] format"""
    return torch.from_numpy(np_img).permute(2, 0, 1).contiguous()


def img_np2pil(np_img):
    """From [H, W, C] to PIL (width, height) format"""
    return ToPILImage()(np_img)


# ----- PIL to X -----

def img_pil2tensor(pil_img):
    """From PIL image to [C, H, W] format"""
    return tv_tensors.Image(pil_img)


def img_pil2np(pil_img):
    """From PIL image to [H, W, C] format"""
    return np.array(pil_img)
