# Lab 2: Demo - Huong dan chay local va tren Kaggle

---

## 1. Yeu cau moi truong

| Thanh phan | Yeu cau |
|---|---|
| Python | 3.9+ |
| PyTorch | 2.x + CUDA (train that) hoac CPU (smoke) |
| Thu vien | pip install -r requirements.txt |
| GPU | Tesla T4 (Kaggle) hoac GPU CUDA >= 6GB |
| Dataset | Intel Image Classification (Kaggle puneet6060) |

---

## 2. Smoke test (local, khong can GPU, khong can dataset)

```bash
pip install -r requirements.txt
python tests/smoke_test.py
```

Ket qua mong doi: **63/63 dat**, duoi 2 phut tren CPU.
Ket qua thuc te: 63/63 dat trong 26s.

---

## 3. Chay tren Kaggle (da train xong)

9/9 P0 da train tren Kaggle (Tesla T4). De xem lai notebook voi output:

1. Vao Kaggle > notebook `lab2-nhom3`
2. Add Input: **Intel Image Classification** (`puneet6060`) + **lab2-src** (ma nguon)
3. Settings > Accelerator > **GPU T4 x2**
4. Nhan **Run All**

---

## 4. Chay bao cao tu dong (local)

```bash
cd d:\LAB2-DL-NHOM3
python scripts/py_report.py
```

Output:
- `outputs/experiment_summary.csv`
- `outputs/report_summary.json`
- `outputs/figures/cmp_*.png` (4 bieu do so sanh)
- `outputs/py_report.log`

---

## 5. TensorBoard

```bash
tensorboard --logdir d:\LAB2-DL-NHOM3\runs\
# Mo trinh duyet: http://localhost:6006
```

---

## 6. Ket qua nhanh

| Model | Test Acc | F1 Macro | Train time | Inference |
|---|---|---|---|---|
| ResNet18 finetune_last (BEST) | **0.9463** | **0.9471** | 23.2 min | 2.4 ms/img |
| DenseNet121 freeze_backbone | 0.9183 | 0.9200 | 24.8 min | 3.6 ms/img |
| VGG16 freeze_backbone | 0.9170 | 0.9190 | 25.4 min | 5.2 ms/img |

---

**Tham chieu:** [training.md](training.md) | [evaluation.md](evaluation.md)
