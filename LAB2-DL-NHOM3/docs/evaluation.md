# Lab 2: Evaluation — Metric, kết quả và phân tích

---

## 1. Metric đánh giá

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| Accuracy | TP+TN / tổng mẫu | Tỉ lệ dự đoán đúng toàn bộ |
| Macro F1 | TB không trọng số F1 từng lớp | Đánh giá công bằng mọi lớp, kể cả lớp ít mẫu |
| Precision từng lớp | TP / (TP+FP) | Tỉ lệ dự đoán đúng trong số dự đoán là lớp đó |
| Recall từng lớp | TP / (TP+FN) | Tỉ lệ bắt đúng trong số mẫu thật của lớp |
| F1 từng lớp | 2·P·R / (P+R) | Cân bằng precision và recall |
| Inference ms/img | Tổng thời gian / số ảnh | Tốc độ dự đoán thực tế (có I/O, không warm-up) |

- **Chọn model:** dựa trên `val_accuracy`; tie-break bằng `val_loss` thấp hơn.
- **Test set:** chỉ dùng một lần sau khi chốt checkpoint; không dùng để chỉnh hyperparameter.

---

## 2. Protocol đánh giá

1. Nạp `checkpoint_best.pth` (lưu lúc val_acc cao nhất trong train).
2. `model.eval()` + `torch.no_grad()`.
3. Chạy toàn bộ `seg_test` (khoảng 3 000 ảnh, batch 32).
4. Tính accuracy, macro F1, precision/recall/F1 từng lớp bằng `sklearn.metrics`.
5. Vẽ confusion matrix 6×6 (chuẩn hoá theo hàng → recall).
6. Tìm top cặp lớp nhầm nhiều nhất.
7. Ghi đầy đủ vào `metrics.json` và log TensorBoard HParams.

---

## 3. Kích thước model và inference

- **Kích thước model:** `model_size_mb()` tính tham số FP32; checkpoint thực tế lớn hơn vì có optimizer state.
- **Inference ms/img:** đo bằng `time.time()` quanh toàn bộ vòng lặp test, gồm đọc dữ liệu, chuyển ảnh lên GPU và đồng bộ ngầm — **không phải** benchmark thuần model, không có warm-up. Chỉ so sánh giữa các run cùng máy, cùng batch size, cùng loader.

| Model | Kích thước FP32 | Inference ms/img (Tesla T4) |
|---|---:|---:|
| ResNet18 | 42,6 MB | 2,4 |
| VGG16 | 512,3 MB | 5,2 |
| DenseNet121 | 26,6 MB | 3,6 |

---

## 4. Artifact của một lần evaluate

`run_full_evaluation()` cập nhật `metrics.json` với: `test_acc`, `test_f1`, `test_precision`, `test_recall`, `per_class_metrics`, `inference_ms_per_image`, `images_per_sec`, `model_size_mb`, `total_params`, `trainable_params`, `confusion_matrix`, `top_error_pairs`, `cm_path`.

File trong `outputs/{exp_name}/`:
- `checkpoint_best.pth` — model tốt nhất
- `checkpoint_latest.pth` — checkpoint cuối
- `metrics.json` — toàn bộ metric
- `confusion_matrix.png` — ma trận nhầm lẫn

TensorBoard event trong `runs/{exp_name}/`.

---

## 5. Kết quả 9 thí nghiệm P0

### E1 — So sánh model và freeze mode

| Thí nghiệm | Model | Freeze | Val Acc | Test Acc | F1 Macro | Trainable | Epochs | Train (h) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| E1-01 | ResNet18 | `freeze_backbone` | 0.9122 | 0.9067 | 0.9087 | 3 078 | 30 | 0,50 |
| E1-02 | ResNet18 | `finetune_last` | 0.9487 | **0.9463** | **0.9471** | 8 396 806 | 30 | 0,39 |
| E1-03 (P1) | ResNet18 | `full_finetune` | — | — | — | 11 179 590 | — | — |
| E1-04 | VGG16 | `freeze_backbone` | 0.9160 | 0.9170 | 0.9190 | 24 582 | 27 | 0,42 |
| E1-05 (P1) | VGG16 | `finetune_last` | — | — | — | 126 649 862 | — | — |
| E1-07 | DenseNet121 | `freeze_backbone` | 0.9217 | 0.9183 | 0.9200 | 6 150 | 30 | 0,41 |

### E2 — Learning rate (ResNet18, `finetune_last`, bs=32, SGD)

| LR | Val Acc | Test Acc | F1 Macro | Epochs | Nhận xét |
|---|---:|---:|---:|---:|---|
| 1e-2 | 0.9463 | 0.9373 | 0.9382 | 16 | Hội tụ nhanh, overfit nhẹ |
| **1e-3 (baseline)** | 0.9487 | **0.9463** | **0.9471** | 30 | Tốt nhất |
| 1e-4 | 0.9221 | 0.9203 | 0.9214 | 30 | Hội tụ chậm, chưa bão hoà |

### E3 — Batch size (ResNet18, `finetune_last`, lr=1e-3, SGD)

| Batch size | Val Acc | Test Acc | F1 Macro | Epochs | Nhận xét |
|---|---:|---:|---:|---:|---|
| 16 | 0.9444 | 0.9413 | 0.9422 | 15 | Hội tụ sớm, gradient ổn định |
| **32 (baseline)** | 0.9487 | **0.9463** | **0.9471** | 30 | Tốt nhất |
| 64 | 0.9421 | 0.9323 | 0.9335 | 28 | Kém hơn nhẹ |

### E4 — Optimizer (ResNet18, `finetune_last`, lr=1e-4, bs=32)

| Optimizer | Val Acc | Test Acc | F1 Macro | Epochs | Nhận xét |
|---|---:|---:|---:|---:|---|
| SGD | 0.9221 | 0.9203 | 0.9214 | 30 | Baseline E4 |
| Adam | 0.9425 | 0.9337 | 0.9350 | 23 | Hội tụ nhanh hơn SGD cùng lr |

---

## 6. Điều kiện chấp nhận kết quả

| Kiểm tra | Điều kiện |
|---|---|
| Artifact đầy đủ | Có `metrics.json`, `checkpoint_best.pth`, `confusion_matrix.png`, event file |
| Metrics hợp lệ | `test_acc`, `test_f1` trong [0, 1]; confusion matrix 6×6 |
| Khớp train/eval | `total_params`, `trainable_params` khớp bảng mục 3 |
| Tái lập | `exp_name` đúng naming, seed 42, cùng split |
| Không rò rỉ test | Test không dùng để chọn epoch/hyperparameter |

---

## 7. Kết luận

### So sánh 3 model (freeze_backbone, cùng lr=1e-3 bs=32 SGD)

| Model | Test Acc | F1 | Kích thước | ms/img |
|---|---:|---:|---:|---:|
| ResNet18 | 0.9067 | 0.9087 | 42,6 MB | 38,5 |
| VGG16 | 0.9170 | 0.9190 | 512,3 MB | 5,2 |
| DenseNet121 | **0.9183** | **0.9200** | **26,6 MB** | **3,6** |

**DenseNet121 hiệu quả nhất** ở freeze_backbone: accuracy cao nhất, model nhỏ nhất, inference nhanh nhất.

### Các nhận xét chính

- **Freeze vs Finetune:** `finetune_last` (0.9463) > `freeze_backbone` (0.9067) → **+3,96 pp**
- **Learning rate:** 1e-3 tối ưu; 1e-2 hội tụ nhanh nhưng kém; 1e-4 hội tụ chậm
- **Batch size:** bs=32 tốt nhất; bs=16 gần tương đương; bs=64 kém nhẹ
- **Optimizer:** SGD lr=1e-3 tốt nhất tổng thể; Adam hội tụ nhanh hơn ở cùng lr
- **Lớp khó nhất:** glacier ↔ mountain (39 và 36 nhầm); lớp dễ nhất: forest (F1=0.9937)

### Per-class F1 — model tốt nhất (ResNet18 finetune_last lr=1e-3 bs=32 SGD)

| Lớp | F1 |
|---|---:|
| buildings | 0.9347 |
| forest | **0.9937** |
| glacier | 0.9139 |
| mountain | 0.9181 |
| sea | 0.9757 |
| street | 0.9464 |

### Khuyến nghị

> **Tốt nhất:** ResNet18 `finetune_last`, lr=1e-3, bs=32, SGD → test_acc=**0.9463**, F1=**0.9471**, 42,6 MB, 0,39h train, 2,4 ms/img 
> **Tài nguyên hạn chế:** DenseNet121 `freeze_backbone` → 0.9183 acc, 26,6 MB, 3,6 ms/img, 0,41h train

---

**Tham chiếu:** `src/evaluate.py` · `src/utils.py` · `notebooks/lab2_report.ipynb` · [training.md](training.md) · [model.md](model.md) · [data.md](data.md)
