"""Standalone AlexNet classifier for Airbus aircraft dataset training.

This module replicates the classic AlexNet convolutional stack (five conv
stages followed by three fully connected layers) and exposes a simple
classification head without any detection-specific components.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


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


class ClassicAlexNetFeatureExtractor(nn.Module):
    """Hand-crafted AlexNet convolutional feature extractor."""

    def __init__(self, trainable_layers: int = 5) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ClassicAlexNetClassifier(nn.Module):
    """AlexNet classifier head built on the scratch feature extractor."""

    def __init__(
        self,
        num_classes: int = 2,
        *,
        dropout: float = 0.5,
        trainable_layers: int = 5,
    ) -> None:
        super().__init__()
        self.features = ClassicAlexNetFeatureExtractor(trainable_layers=trainable_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        flattened_size = self.features.out_channels * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(flattened_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_alexnet_classifier(
    num_classes: int = 2,
    *,
    dropout: float = 0.5,
    trainable_layers: int = 5,
) -> ClassicAlexNetClassifier:
    """Factory helper to instantiate the classifier with custom settings."""

    return ClassicAlexNetClassifier(
        num_classes=num_classes,
        dropout=dropout,
        trainable_layers=trainable_layers,
    )


__all__ = [
    "ClassicAlexNetClassifier",
    "ClassicAlexNetFeatureExtractor",
    "build_alexnet_classifier",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]

