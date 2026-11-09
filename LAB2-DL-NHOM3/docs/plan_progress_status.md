# Lab 2: Ke hoach & Tien do

---

## 1. Tong quan

| Hang muc | Trang thai |
|---|---|
| 7 file docs/ | Hoan thanh |
| Ma nguon src/ | Hoan thanh |
| Cau hinh configs/ | Hoan thanh (23 TN) |
| Smoke test 63/63 | Hoan thanh (26s tren CPU) |
| Notebook lab2_report.ipynb | Hoan thanh |
| 9/9 P0 thi nghiem | Hoan thanh (Kaggle T4) |
| Test metric | test_acc=0.9463, F1=0.9471 |

---

## 2. Ket qua 9 thi nghiem P0

| Hang | Experiment | Test Acc | F1 Macro | Train (min) |
|---:|---|---:|---:|---:|
| 1 | resnet18__finetune_last__lr1e-3__bs32__sgd | **0.9463** | **0.9471** | 23.2 |
| 2 | resnet18__finetune_last__lr1e-3__bs16__sgd | 0.9413 | 0.9422 | 12.0 |
| 3 | resnet18__finetune_last__lr1e-2__bs32__sgd | 0.9373 | 0.9382 | 12.3 |
| 4 | resnet18__finetune_last__lr1e-4__bs32__adam | 0.9337 | 0.9350 | 17.8 |
| 5 | resnet18__finetune_last__lr1e-3__bs64__sgd | 0.9323 | 0.9335 | 22.8 |
| 6 | resnet18__finetune_last__lr1e-4__bs32__sgd | 0.9203 | 0.9214 | 22.9 |
| 7 | densenet121__freeze_backbone__lr1e-3__bs32__sgd | 0.9183 | 0.9200 | 24.8 |
| 8 | vgg16__freeze_backbone__lr1e-3__bs32__sgd | 0.9170 | 0.9190 | 25.4 |
| 9 | resnet18__freeze_backbone__lr1e-3__bs32__sgd | 0.9067 | 0.9087 | 30.2 |

---

## 3. REQ Checklist

| REQ | Noi dung | Trang thai |
|---|---|---|
| REQ-01 | 3 model pretrained | PASS |
| REQ-02 | 2 freeze mode | PASS |
| REQ-03 | 3 learning rate | PASS |
| REQ-04 | 3 batch size | PASS |
| REQ-05 | SGD vs Adam | PASS |
| REQ-06 | TensorBoard logs | PASS |
| REQ-07 | Test set evaluation | PASS |
| REQ-08 | 7 file docs/ | PASS |
| REQ-09 | Smoke test < 2 min | PASS (26s) |
| REQ-10 | Notebook Run All | PASS (Kaggle) |
| REQ-11 | Resume mechanism | PASS |
| REQ-12 | Error analysis | PASS |
| REQ-13 | Demo seg_pred | PASS |

---

## 4. Viec con lai

| Viec | Ghi chu |
|---|---|
| Nop bai | Zip toan bo thu muc LAB2-DL-NHOM3 |
| Thuyet trinh | Dung artifact thuyet_trinh_FULL.md |

---

**Cap nhat lan cuoi:** 2026-11-09
