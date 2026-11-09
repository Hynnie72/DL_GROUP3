"""
models/__init__.py - Registry và factory function cho các model.
"""

from typing import Callable, Dict
import torch.nn as nn

from src.models.resnet18 import build as build_resnet18
from src.models.vgg16 import build as build_vgg16
from src.models.densenet121 import build as build_densenet121


# Registry: tên model → hàm build
MODEL_REGISTRY: Dict[str, Callable] = {
    "resnet18":    build_resnet18,
    "vgg16":       build_vgg16,
    "densenet121": build_densenet121,
}

VALID_FREEZE_MODES = {"freeze_backbone", "finetune_last", "full_finetune"}


def build_model(
    name: str,
    num_classes: int = 6,
    freeze_mode: str = "finetune_last",
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function: tạo model theo tên và cấu hình.

    Args:
        name: Tên model ('resnet18', 'vgg16', 'densenet121').
        num_classes: Số lớp phân loại (mặc định 6 cho Intel dataset).
        freeze_mode: Chế độ đóng băng ('freeze_backbone', 'finetune_last', 'full_finetune').
        pretrained: Nạp weights ImageNet hay không.

    Returns:
        Model đã được adapt và đặt freeze mode.

    Raises:
        ValueError: Nếu tên model hoặc freeze_mode không hợp lệ.
    """
    name = name.lower().strip()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' không được hỗ trợ. "
            f"Các model hợp lệ: {list(MODEL_REGISTRY.keys())}"
        )

    if freeze_mode not in VALID_FREEZE_MODES:
        raise ValueError(
            f"freeze_mode '{freeze_mode}' không hợp lệ. "
            f"Các mode hợp lệ: {VALID_FREEZE_MODES}"
        )

    return MODEL_REGISTRY[name](
        num_classes=num_classes,
        freeze_mode=freeze_mode,
        pretrained=pretrained,
    )


def list_models() -> list:
    """Liệt kê tên các model có sẵn."""
    return list(MODEL_REGISTRY.keys())
