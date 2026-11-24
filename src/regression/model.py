"""Medical Image Regression Model Architectures.
Model design: CNN Backbone + Global Average Pooling (GAP) + Linear Output Head.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models

from .config import RegressionConfig, ModelConfig


class CustomMedicalCNN(nn.Module):
    """Custom Convolutional Neural Network with Global Average Pooling (GAP) and Linear regression head.

    Architecture
    ------------
    Input (B, C, H, W)
      │
      ▼
    [Conv2d(32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)]
      │
      ▼
    [Conv2d(64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)]
      │
      ▼
    [Conv2d(128, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)]
      │
      ▼
    [Conv2d(256, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)]
      │
      ▼
    Global Average Pooling: AdaptiveAvgPool2d((1, 1)) -> (B, 256, 1, 1)
      │
      ▼
    Flatten: (B, 256)
      │
      ▼
    Dropout(p) -> Linear(256, 64) -> ReLU -> Linear(64, 1)
      │
      ▼
    Output (B, 1) Continuous Target
    """

    def __init__(self, in_channels: int = 3, dropout_rate: float = 0.2) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Average Pooling collapses spatial (H, W) -> (1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Regression Head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.gap(feat)
        out = self.head(pooled)
        return out.squeeze(-1)  # Return shape (B,)


class PretrainedMedicalCNN(nn.Module):
    """Transfer Learning CNN Backbone (ResNet family) with GAP and Linear Output for Regression."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 3,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
        elif backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base_model = models.resnet34(weights=weights)
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Adapt input conv if grayscale (1 channel)
        if in_channels != 3:
            old_conv = base_model.conv1
            base_model.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias,
            )

        # Backbone layers (all layers up to avgpool)
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )

        # Explicit Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        in_features = base_model.fc.in_features

        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Linear regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.gap(feat)
        out = self.head(pooled)
        return out.squeeze(-1)  # Return shape (B,)


def build_regression_model(config: RegressionConfig) -> nn.Module:
    """Build and return regression model according to configuration."""
    m_cfg = config.model
    d_cfg = config.data

    if m_cfg.backbone == "custom_cnn":
        return CustomMedicalCNN(
            in_channels=d_cfg.in_channels,
            dropout_rate=m_cfg.dropout_rate,
        )
    else:
        return PretrainedMedicalCNN(
            backbone_name=m_cfg.backbone,
            pretrained=m_cfg.pretrained,
            in_channels=d_cfg.in_channels,
            dropout_rate=m_cfg.dropout_rate,
            freeze_backbone=m_cfg.freeze_backbone,
        )
