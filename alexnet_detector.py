"""AlexNet-based object detector.

This module keeps the classical AlexNet convolutional stem as a feature extractor
and connects it to a standard Faster R-CNN detection head (RPN + ROI heads) so
that the network can predict multiple aircraft bounding boxes and objectness
scores per 2560×2560 satellite tile.

Usage:
    from alexnet_detector import build_alexnet_detector
    model = build_alexnet_detector(num_classes=3)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Tuple

import torch
from torch import nn
from torchvision.models import AlexNet_Weights, alexnet
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


class AlexNetFeatureBackbone(nn.Module):
    """Wrap the AlexNet feature extractor so it can be plugged into detection heads."""

    def __init__(
        self,
        weights: AlexNet_Weights | None = AlexNet_Weights.IMAGENET1K_V1,
        trainable_layers: int = 2,
    ) -> None:
        super().__init__()
        base = alexnet(weights=weights)

        # Freeze earlier layers if desired to keep ImageNet priors intact.
        if trainable_layers < 5:
            for name, param in base.features.named_parameters():
                block_idx = int(name.split(".")[0])
                if block_idx < len(base.features) - trainable_layers:
                    param.requires_grad = False

        self.body = base.features
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict([("feat", self.body(x))])


def build_alexnet_detector(
    num_classes: int,
    *,
    pretrained_backbone: bool = True,
    trainable_layers: int = 2,
    anchor_sizes: Iterable[int] = (32, 64, 128, 256, 512),
    aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
) -> FasterRCNN:
    """Create a Faster R-CNN style detector backed by AlexNet features."""

    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    backbone = AlexNetFeatureBackbone(weights=weights, trainable_layers=trainable_layers)

    anchor_generator = AnchorGenerator(
        sizes=(tuple(anchor_sizes),),
        aspect_ratios=(aspect_ratios,),
    )

    roi_pooler = MultiScaleRoIAlign(featmap_names=["feat"], output_size=7, sampling_ratio=2)

    # AlexNet's default normalization ensures compatibility with ImageNet weights.
    weight_transforms = (
        AlexNet_Weights.IMAGENET1K_V1.transforms()
        if pretrained_backbone
        else AlexNet_Weights.IMAGENET1K_V1.transforms()
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        image_mean=weight_transforms.mean,
        image_std=weight_transforms.std,
    )
    return model


__all__ = ["AlexNetFeatureBackbone", "build_alexnet_detector"]

