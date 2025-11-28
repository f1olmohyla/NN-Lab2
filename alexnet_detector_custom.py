"""AlexNet detector without torchvision pretraining.

This module recreates the original AlexNet backbone described by Krizhevsky et al.
in the NeurIPS 2012 paper "ImageNet Classification with Deep Convolutional Neural Networks"
and attaches a Faster R-CNN detection head on top. All layers are initialized from scratch
following the classical architecture (five convolutional stages + three fully-connected
layers), so no torchvision weights are loaded.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.constant_(module.bias, 0.0)


class ClassicAlexNetBackbone(nn.Module):
    """Hand-crafted AlexNet backbone with five convolutional stages."""

    def __init__(self, trainable_layers: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                ),
                nn.Sequential(
                    nn.Conv2d(96, 256, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 384, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(384, 384, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                ),
            ]
        )
        self.out_channels = 256
        self.apply(_init_weights)

        total_blocks = len(self.blocks)
        for idx, block in enumerate(self.blocks):
            if idx < total_blocks - trainable_layers:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
        return OrderedDict([("feat", x)])


def _load_classifier_backbone_weights(backbone: ClassicAlexNetBackbone, weights_path: Path) -> None:
    """Load convolutional weights from a classifier checkpoint into the backbone."""

    state_dict = torch.load(weights_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError("Classifier checkpoint must be a plain state_dict dictionary.")

    prefix = "features."
    backbone_state = {}
    for key, tensor in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            backbone_state[new_key] = tensor

    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if missing:
        raise RuntimeError(f"Missing backbone parameters when loading classifier weights: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected parameters in classifier weights: {unexpected}")


def build_alexnet_detector(
    num_classes: int,
    *,
    trainable_layers: int = 2,
    anchor_sizes: Iterable[int] = (32, 64, 128, 256, 512),
    aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    classifier_weights: str | Path | None = None,
    freeze_backbone: bool = False,
) -> FasterRCNN:
    """Create a Faster R-CNN detector using the scratch-built AlexNet backbone."""

    backbone = ClassicAlexNetBackbone(trainable_layers=trainable_layers)

    if classifier_weights is not None:
        _load_classifier_backbone_weights(backbone, Path(classifier_weights))

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    anchor_generator = AnchorGenerator(
        sizes=(tuple(anchor_sizes),),
        aspect_ratios=(aspect_ratios,),
    )

    roi_pooler = MultiScaleRoIAlign(featmap_names=["feat"], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        image_mean=IMAGENET_MEAN,
        image_std=IMAGENET_STD,
    )
    return model


__all__ = ["ClassicAlexNetBackbone", "build_alexnet_detector"]
