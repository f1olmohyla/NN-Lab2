"""
Region warping and preprocessing transforms
Section 2.1 & Appendix A: Warp regions to 227x227 with context padding
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple
from torchvision import transforms


class RegionWarper:

    def __init__(
        self,
        output_size: Tuple[int, int] = (227, 227),
        context_padding: int = 16,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):

        self.output_size = output_size
        self.context_padding = context_padding
        self.mean = mean
        self.std = std

        self.normalize = transforms.Normalize(mean=mean, std=std)

    def warp_region(
        self,
        image: Image.Image,
        bbox: np.ndarray
    ) -> torch.Tensor:

        x1, y1, x2, y2 = bbox
        img_w, img_h = image.size

        box_w = x2 - x1
        box_h = y2 - y1

        pad_w = (self.context_padding / self.output_size[0]) * box_w
        pad_h = (self.context_padding / self.output_size[1]) * box_h

        x1_padded = max(0, x1 - pad_w)
        y1_padded = max(0, y1 - pad_h)
        x2_padded = min(img_w, x2 + pad_w)
        y2_padded = min(img_h, y2 + pad_h)

        crop = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))

        warped = crop.resize(self.output_size, Image.BILINEAR)

        tensor = transforms.ToTensor()(warped)
        normalized = self.normalize(tensor)

        return normalized

    def __call__(self, image: Image.Image, bbox: np.ndarray) -> torch.Tensor:
        return self.warp_region(image, bbox)


def get_alexnet_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]     # ImageNet std
        )
    ])
