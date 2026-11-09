# Lab 2: Data — Intel Image Classification

---

## 1. Nguồn và bài toán

- **Tên:** Intel Image Classification
- **Nguồn:** Kaggle, tác giả `puneet6060` — <https://www.kaggle.com/datasets/puneet6060/intel-image-classification>
- **Định dạng:** JPEG, ảnh gốc khoảng 150×150; input model resize về 224×224
- **Bài toán:** Phân loại đa lớp — mỗi ảnh thuộc đúng một trong 6 lớp
- **Quy mô công bố:** khoảng 14 034 ảnh train, 3 000 ảnh test, 7 301 ảnh `seg_pred`. Số thực tế từng lớp do notebook đếm.

---

## 2. Sáu lớp

| Index | Nhãn | Mô tả |
|---:|---|---|
| 0 | `buildings` | Công trình, toà nhà, cảnh đô thị |
| 1 | `forest` | Rừng cây, thảm thực vật dày |
| 2 | `glacier` | Sông băng, bề mặt băng tuyết |
| 3 | `mountain` | Núi, địa hình cao |
| 4 | `sea` | Biển, đại dương, mặt nước rộng |
| 5 | `street` | Đường phố, vỉa hè |

- Index lấy từ `ImageFolder.class_to_idx`, tức sắp xếp alphabet theo tên thư mục.
- `class_to_idx` phải lưu cùng checkpoint và dùng lại khi load để tránh lệch nhãn.
- Các cặp dễ nhầm cần kiểm chứng bằng confusion matrix: `glacier` ↔ `mountain`, `buildings` ↔ `street`.

---

## 3. Cấu trúc thư mục

```
intel-image-classification/
seg_train/seg_train/<class>/*.jpg
seg_test/seg_test/<class>/*.jpg
seg_pred/seg_pred/*.jpg # không nhãn, chỉ demo
```

- Cấu trúc lồng hai cấp cùng tên. `find_dataset_root()` dò `/kaggle/input` bằng `os.walk` để tìm thư mục có `seg_train` và `seg_test`, fallback về `dataset.local_root` trong `configs/base.yaml`. Không hard-code đường dẫn.

---

## 4. Chia train / validation

| Thông số | Giá trị |
|---|---|
| Nguồn để chia | `seg_train` |
| Tỉ lệ | 85% train / 15% validation |
| Phương pháp | `train_test_split(stratify=labels)` |
| Seed | 42 |
| Test | Nguyên `seg_test`, không chia lại |
| Quy tắc | **Không dùng test để chọn model hoặc chỉnh hyperparameter** |

- Stratified split giữ tỉ lệ lớp gần đồng đều trong cả train và validation, tránh validation thiếu mẫu của một lớp.

---

## 5. Transforms

**Train:**
`RandomResizedCrop(224, scale=(0.7, 1.0))` → `RandomHorizontalFlip(p=0.5)` → `RandomRotation(10)` → `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)` → `ToTensor()` → `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` → `RandomErasing(p=0.1, scale=(0.02, 0.15))`

**Validation / Test:**
`Resize(256)` → `CenterCrop(224)` → `ToTensor()` → `Normalize` như trên

| Transform | Lý do |
|---|---|
| `RandomResizedCrop(224, scale=(0.7, 1.0))` | Ảnh gốc 150×150 phải phóng to lên 224; crop ngẫu nhiên tăng đa dạng góc nhìn |
| `RandomHorizontalFlip` | Augmentation đơn giản, phù hợp ảnh phong cảnh |
| `RandomRotation(10)` | Ảnh phong cảnh có thể nghiêng nhẹ; xoay ±10° tăng đa dạng mà không phá ngữ nghĩa |
| `ColorJitter` nhẹ | Tăng robustness với thay đổi ánh sáng; chỉ áp dụng cho train |
| `RandomErasing(p=0.1)` | Che ngẫu nhiên một vùng nhỏ, buộc model dựa vào nhiều vùng ảnh hơn |
| `Resize(256) + CenterCrop(224)` | Val/test deterministic, giữ tỉ lệ và cắt trung tâm nhất quán |
| `Normalize` ImageNet | Khớp phân phối đầu vào của model pretrained |

- `RandomErasing` đặt sau `Normalize` theo khuyến nghị của torchvision.
- Val/test không có augmentation để metric ổn định giữa các lần đánh giá.

---

## 6. DataLoader

| Thông số | Giá trị | Ghi chú |
|---|---|---|
| `batch_size` | 32 | E3 dùng 16 và 64 |
| `num_workers` | 2 | Đặt 0 khi debug trên Windows |
| `pin_memory` | `torch.cuda.is_available()` | Chỉ có tác dụng với GPU |
| `shuffle` | True (train), False (val/test) | |
| `drop_last` | False | Giữ batch cuối |

- `get_dataloaders()` trả `(train_loader, val_loader, test_loader, class_to_idx)`.
- Hàm chỉ đọc `seg_train` và `seg_test`; không tạo loader cho `seg_pred` để tránh dùng nhầm cho metric.

---

## 7. Thống kê dữ liệu (điền sau khi chạy notebook)

| Lop | Train | Validation | Test |
|---|---:|---:|---:|
| buildings | 2 191 | 387 | 437 |
| forest | 2 271 | 400 | 474 |
| glacier | 2 404 | 425 | 553 |
| mountain | 2 236 | 395 | 525 |
| sea | 2 280 | 403 | 510 |
| street | 2 278 | 402 | 501 |
| **Tong** | **13 660** | **2 412** | **3 000** |

---

## 8. Kiểm tra chất lượng dữ liệu

- Số ảnh mỗi lớp ở 3 split; tỉ lệ mất cân bằng `max / min`.
- Ảnh không đọc được (`UnidentifiedImageError`) hoặc file hỏng.
- Lưới ảnh mẫu sau khi denormalize (xác nhận transform đúng).

---

## 9. Tập `seg_pred` và artifact

- `seg_pred` không có nhãn, chỉ dùng demo; không tính accuracy/F1; dùng `val_transform`.
- `get_pred_loader()` giới hạn số ảnh demo để tránh tốn thời gian.
- Kết quả demo gồm ảnh, nhãn dự đoán và độ tin cậy; lưu vào `outputs/demo_predictions.png`.

---

**Tham chiếu:** `src/data.py` · `configs/base.yaml` · [model.md](model.md) · [training.md](training.md) · [evaluation.md](evaluation.md)
