"""
CNN Feature Extractor (AlexNet)
Section 2.1: "We extract a 4096-dimensional feature vector from each region proposal"
"""

import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights
from typing import Optional, Literal


class AlexNetFeatureExtractor(nn.Module):
    """
    Architecture: 5 conv layers + 2 fc layers → extract fc6 or fc7 features
    """

    def __init__(
        self,
        pretrained: bool = True,
        feature_layer: Literal["fc6", "fc7"] = "fc7",
        num_classes: int = 1
    ):
        super().__init__()

        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        self.alexnet = alexnet(weights=weights)

        self.feature_layer = feature_layer
        self.num_classes = num_classes

        self.features = self.alexnet.features  # 5 conv layers
        self.avgpool = self.alexnet.avgpool

        # Fully connected layers
        # fc6: classifier[1] (4096 outputs)
        # fc7: classifier[4] (4096 outputs)
        self.fc6 = nn.Sequential(
            nn.Dropout(),
            self.alexnet.classifier[1],  # Linear(9216, 4096)
            nn.ReLU(inplace=True),
        )

        self.fc7 = nn.Sequential(
            nn.Dropout(),
            self.alexnet.classifier[4],  # Linear(4096, 4096)
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(4096, num_classes + 1)

        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:

        # Conv layers
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # FC6
        x = self.fc6(x)

        if self.feature_layer == "fc6":
            return x

        # FC7
        x = self.fc7(x)
        return x

    def forward(self, x: torch.Tensor, extract_features: bool = False):
        features = self.extract_features(x)

        if extract_features:
            return features

        # Classification (for fine-tuning)
        logits = self.classifier(features)
        return logits

    def freeze_layers(self, num_layers: int = 0):
        if num_layers == 0:
            return

        layers = list(self.features.children())[:num_layers]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False


def load_pretrained_alexnet(feature_layer: str = "fc7") -> AlexNetFeatureExtractor:
    return AlexNetFeatureExtractor(
        pretrained=True,
        feature_layer=feature_layer,
        num_classes=1
    )
