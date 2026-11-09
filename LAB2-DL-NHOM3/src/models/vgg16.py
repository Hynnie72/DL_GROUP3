"""
vgg16.py - VGG-16 pre-trained model cho Intel Image Classification.

VGG-16: 16 lớp có tham số (13 conv + 3 FC), ~138M tham số.
Features (conv): 13 lớp Conv2d theo 5 khối, MaxPool
Avgpool: AdaptiveAvgPool2d(7, 7)
Classifier: Flatten → Linear(25088→4096) → ReLU → Dropout(0.5)
            → Linear(4096→4096) → ReLU → Dropout(0.5) → Linear(4096→num_classes)

Quyết định thiết kế: GIỮ NGUYÊN 2 lớp Dropout(p=0.5) trong classifier.
Lý do: Intel dataset có ~14k ảnh train, tương đối nhỏ so với ImageNet (1.2M).
Dropout giúp regularize, giảm overfitting. Các nghiên cứu cho thấy giữ Dropout
của VGG16 khi fine-tune trên dataset nhỏ/vừa cho kết quả tốt hơn bỏ đi.

Lưu ý hiệu năng: VGG16 có ~138M params và classifier rất lớn (25088→4096).
AMP (Automatic Mixed Precision) bắt buộc khi train trên GPU để tiết kiệm VRAM.
Nếu OOM với batch_size=32, giảm xuống 16 hoặc 8.

API mới: weights=models.VGG16_Weights.IMAGENET1K_V1 (thay pretrained=True).
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights

from src.utils import freeze_all, unfreeze_all, unfreeze_module, set_bn_eval


def build(
    num_classes: int = 6,
    freeze_mode: str = "finetune_last",
    pretrained: bool = True,
) -> nn.Module:
    """
    Xây dựng VGG-16 đã được adapt cho bài toán phân loại.

    Args:
        num_classes: Số lớp đầu ra (6 cho Intel dataset).
        freeze_mode: Chiến lược đóng băng tham số.
        pretrained: Nạp weights ImageNet1K V1.

    Returns:
        VGG-16 model đã adapt.
    """
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights)

    # Thay lớp phân loại cuối (index 6): Linear(4096→1000) → Linear(4096→num_classes)
    # Giữ nguyên Dropout(0.5) tại index 2 và 5 trong classifier
    in_features = model.classifier[6].in_features  # 4096
    model.classifier[6] = nn.Linear(in_features, num_classes)

    _apply_freeze_mode(model, freeze_mode)

    return model


def _apply_freeze_mode(model: nn.Module, freeze_mode: str) -> None:
    """
    Áp dụng chế độ đóng băng cho VGG-16.

    freeze_backbone: Đóng toàn bộ model (features + classifier), chỉ mở
                     classifier[6] (lớp Linear mới thay cho 1000 lớp ImageNet).
                     Giữ classifier[0..5] đóng băng để tránh train 119.5M tham số
                     FC không cần thiết khi dataset nhỏ.
    finetune_last:   Đóng features[0..23] (4 khối đầu), mở features[24:] (khối conv 5 cuối)
                     + toàn bộ classifier.
                     Lý do: VGG có 5 khối conv, mở khối 5 (features[24:]) chứa các
                     đặc trưng mức cao nhất trước pooling. Kết hợp với toàn bộ
                     classifier giúp adapt tốt cho domain mới (cảnh quan).
    full_finetune:   Mở toàn bộ.

    Cấu trúc VGG16 features (index):
      Block 1: [0]  Conv2d(3,64)   [1] ReLU  [2] Conv2d(64,64)  [3] ReLU  [4] MaxPool
      Block 2: [5]  Conv2d(64,128) [6] ReLU  [7] Conv2d(128,128)[8] ReLU  [9] MaxPool
      Block 3: [10] Conv2d(128,256)...[16] MaxPool
      Block 4: [17] Conv2d(256,512)...[23] MaxPool
      Block 5: [24] Conv2d(512,512)...[30] MaxPool  ← mở ở finetune_last
    """
    if freeze_mode == "freeze_backbone":
        freeze_all(model)
        unfreeze_module(model.classifier[6])

    elif freeze_mode == "finetune_last":
        freeze_all(model)
        # Mở Block 5 (features index 24 đến hết)
        for i in range(24, len(model.features)):
            unfreeze_module(model.features[i])
        # Mở toàn bộ classifier
        unfreeze_module(model.classifier)

    elif freeze_mode == "full_finetune":
        unfreeze_all(model)

    else:
        raise ValueError(f"freeze_mode không hợp lệ: {freeze_mode}")


def set_train_mode(model: nn.Module, freeze_mode: str) -> None:
    """
    Gọi khi bắt đầu mỗi epoch. Đảm bảo BN đóng băng ở eval mode.
    VGG16 không có BN mặc định (dùng vgg16, không phải vgg16_bn),
    nhưng vẫn gọi set_bn_eval() để an toàn nếu có BN nào được thêm vào.
    """
    model.train()
    if freeze_mode in ("freeze_backbone", "finetune_last"):
        set_bn_eval(model)
