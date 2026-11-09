# Model: Lab 2 - Transfer Learning (Intel Scene Classification)

| Truong        | Noi dung                        |
|---------------|---------------------------------|
| Module        | docs/model.md                   |
| Author        | Nhom 3 - LAB2-DL-NHOM3         |
| Created date  | 2026-10                         |
| Updated date  | 2026-11-09                      |

---

## Purpose

Phan loai anh phong canh Intel 6 lop (buildings, forest, glacier, mountain, sea, street)
bang Transfer Learning tu 3 CNN pretrained tren ImageNet-1K: ResNet18, VGG16, DenseNet121.

---

## Initial model

- **Architecture** : ResNet18 / VGG16 / DenseNet121 (pretrained ImageNet-1K)
  Head cu duoc thay bang `nn.Linear(feature_dim, 6)`
- **Input / Output**: Input `[B, 3, 224, 224]` RGB chuan hoa ImageNet
  Output logits `[B, 6]` (chua softmax)
- **Source files**  : `src/models/resnet18.py`, `vgg16.py`, `densenet121.py`, `__init__.py`

| Model       | Params  | Size FP32 | Feature dim | Top-1 ImageNet |
|-------------|--------:|----------:|------------:|---------------:|
| ResNet18    | 11.2 M  | 42.6 MB   | 512         | 69.8%          |
| VGG16       | 134.3 M | 512.3 MB  | 4096        | 71.6%          |
| DenseNet121 | 7.0 M   | 26.6 MB   | 1024        | 74.4%          |

---

## Trained model

- **Train desc**  : 9 thi nghiem P0 (E1-E4) — so sanh model, freeze mode, LR, batch size, optimizer
  Chi tiet: `docs/training.md` | `configs/experiments.yaml`
- **Saving path** : `outputs/<exp_name>/checkpoint_best.pth`
  Vi du: `outputs/resnet18__finetune_last__lr1e-3__bs32__sgd/checkpoint_best.pth`
- **Best result** : ResNet18 finetune_last lr=1e-3 bs=32 SGD
  test_acc=**0.9463**, F1=**0.9471**

---

## Freeze modes

| Mode             | Mo ta                                      | Trainable params (ResNet18) |
|------------------|--------------------------------------------|-----------------------------|
| freeze_backbone  | Chi train head (Linear layer)              | 3 078                       |
| finetune_last    | Train head + layer4 (block cuoi backbone)  | 8 396 806                   |
| full_finetune    | Train toan bo model                        | 11 179 590                  |

---

**Tham chieu:** `src/models/` | [training.md](training.md) | [evaluation.md](evaluation.md)
