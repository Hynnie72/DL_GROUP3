"""
densenet121.py - DenseNet-121 pre-trained model cho Intel Image Classification.

DenseNet-121: 121 lớp, dense connections (mỗi lớp nhận input từ TẤT CẢ lớp trước),
~8M tham số (nhỏ hơn ResNet18 và VGG16 nhiều nhưng hiệu quả tham số cao hơn).

Kiến trúc:
  features:
    conv0 → norm0 → relu0 → pool0
    → denseblock1 → transition1
    → denseblock2 → transition2
    → denseblock3 → transition3
    → denseblock4 → norm5
  → ReLU → AdaptiveAvgPool2d → Flatten
  → classifier: Linear(1024 → num_classes)

API mới: weights=models.DenseNet121_Weights.IMAGENET1K_V1
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

from src.utils import freeze_all, unfreeze_all, freeze_module, unfreeze_module, set_bn_eval


def build(
    num_classes: int = 6,
    freeze_mode: str = "finetune_last",
    pretrained: bool = True,
) -> nn.Module:
    """
    Xây dựng DenseNet-121 đã được adapt cho bài toán phân loại.

    Args:
        num_classes: Số lớp đầu ra (6 cho Intel dataset).
        freeze_mode: Chiến lược đóng băng tham số.
        pretrained: Nạp weights ImageNet1K V1.

    Returns:
        DenseNet-121 model đã adapt.
    """
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)

    # Thay lớp phân loại cuối: Linear(1024 → 1000) → Linear(1024 → num_classes)
    in_features = model.classifier.in_features  # 1024
    model.classifier = nn.Linear(in_features, num_classes)

    _apply_freeze_mode(model, freeze_mode)

    return model


def _apply_freeze_mode(model: nn.Module, freeze_mode: str) -> None:
    """
    Áp dụng chế độ đóng băng cho DenseNet-121.

    freeze_backbone: Đóng toàn bộ features, chỉ mở classifier.
    finetune_last:   Đóng conv0~denseblock3, mở denseblock4 + norm5 + classifier.
                     Lý do: denseblock4 là khối dense cuối cùng với 16 lớp dense,
                     chứa các đặc trưng mức cao nhất. norm5 là lớp BN cuối trước pool.
                     Mở cả 2 + classifier cho phép fine-tune phần học đặc trưng quan
                     trọng nhất mà không cần train toàn bộ mạng (tiết kiệm VRAM/thời gian).
    full_finetune:   Mở toàn bộ.
    """
    if freeze_mode == "freeze_backbone":
        freeze_module(model.features)
        unfreeze_module(model.classifier)

    elif freeze_mode == "finetune_last":
        freeze_all(model)
        unfreeze_module(model.features.denseblock4)
        unfreeze_module(model.features.norm5)
        unfreeze_module(model.classifier)

    elif freeze_mode == "full_finetune":
        unfreeze_all(model)

    else:
        raise ValueError(f"freeze_mode không hợp lệ: {freeze_mode}")


def set_train_mode(model: nn.Module, freeze_mode: str) -> None:
    """
    Gọi khi bắt đầu mỗi epoch. Đảm bảo BN đóng băng ở eval mode.
    DenseNet có rất nhiều BN (mỗi dense layer đều có BN).
    Khi freeze_backbone/finetune_last, BN trong phần đóng băng phải ở eval().
    """
    model.train()
    if freeze_mode in ("freeze_backbone", "finetune_last"):
        set_bn_eval(model)
