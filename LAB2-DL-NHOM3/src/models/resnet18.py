"""
resnet18.py - ResNet-18 pre-trained model cho Intel Image Classification.

ResNet-18: 18 lớp, skip connections, ~11.7M tham số.
Backbone: Conv1 → BN → ReLU → MaxPool → Layer1 → Layer2 → Layer3 → Layer4
Head: AdaptiveAvgPool → Flatten → FC(512 → num_classes)

API mới (PyTorch >= 0.13): weights=models.ResNet18_Weights.IMAGENET1K_V1
Thay thế pretrained=True đã bị deprecated (cùng kết quả, chỉ khác cách gọi).
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.utils import freeze_all, unfreeze_all, freeze_module, unfreeze_module, set_bn_eval


def build(
    num_classes: int = 6,
    freeze_mode: str = "finetune_last",
    pretrained: bool = True,
) -> nn.Module:
    """
    Xây dựng ResNet-18 đã được adapt cho bài toán phân loại.

    Args:
        num_classes: Số lớp đầu ra (6 cho Intel dataset).
        freeze_mode: Chiến lược đóng băng tham số:
            - 'freeze_backbone': Đóng toàn bộ, chỉ train lớp fc mới.
            - 'finetune_last': Mở layer4 + fc (fine-tune phần cuối).
            - 'full_finetune': Mở toàn bộ (fine-tune toàn mạng).
        pretrained: Nạp weights ImageNet1K V1 hay khởi tạo ngẫu nhiên.

    Returns:
        ResNet-18 model đã adapt.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Thay lớp phân loại cuối: fc(512 → 1000) → fc(512 → num_classes)
    in_features = model.fc.in_features  # 512
    model.fc = nn.Linear(in_features, num_classes)

    _apply_freeze_mode(model, freeze_mode)

    return model


def _apply_freeze_mode(model: nn.Module, freeze_mode: str) -> None:
    """
    Áp dụng chế độ đóng băng cho ResNet-18.

    freeze_backbone: Đóng toàn bộ backbone, chỉ mở fc mới.
    finetune_last:   Đóng conv1~layer3, mở layer4 + fc.
                     Lý do: layer4 học đặc trưng mức cao (texture phức tạp, shapes)
                     phù hợp nhất để fine-tune cho domain mới (cảnh quan tự nhiên).
    full_finetune:   Mở toàn bộ.
    """
    if freeze_mode == "freeze_backbone":
        freeze_all(model)
        unfreeze_module(model.fc)

    elif freeze_mode == "finetune_last":
        freeze_all(model)
        unfreeze_module(model.layer4)
        unfreeze_module(model.fc)

    elif freeze_mode == "full_finetune":
        unfreeze_all(model)

    else:
        raise ValueError(f"freeze_mode không hợp lệ: {freeze_mode}")


def set_train_mode(model: nn.Module, freeze_mode: str) -> None:
    """
    Gọi khi bắt đầu mỗi epoch để đảm bảo BatchNorm đóng băng ở eval mode.

    Lý do: model.train() sẽ set TẤT CẢ module (kể cả BN đóng băng) về train mode.
    Điều này khiến BN đóng băng cập nhật running stats với batch hiện tại,
    gây ra không ổn định khi batch size nhỏ. Ta cần override lại bằng set_bn_eval().
    """
    model.train()
    if freeze_mode in ("freeze_backbone", "finetune_last"):
        set_bn_eval(model)
