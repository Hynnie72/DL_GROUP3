# Demo — Lab 3 Practice 3

## 1. Yêu cầu môi trường

| Yêu cầu | Phiên bản |
|---|---|
| Python | >= 3.10 |
| PyTorch | >= 2.0 |
| CUDA | Tùy chọn (CPU cũng chạy được, nhưng chậm hơn) |
| RAM | >= 8 GB |

## 2. Cài đặt thư viện

```powershell
# Tạo virtual environment (khuyến nghị)
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell

# Cài thư viện từ requirements.txt
pip install -r requirements.txt
```

Thư viện chính: `torch`, `transformers`, `datasets`, `evaluate`,
`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `accelerate`, `fpdf2`.

## 3. Chuẩn bị dữ liệu

```
D:\Lab3_DL\
└── data\
    └── IMDB Dataset.csv    <- 50,000 rows, ~66 MB
```

Dataset không cần xử lý thêm — chương trình tự đọc, split và tokenize.

## 4. Smoke Run — Bắt buộc trước khi chạy full

```powershell
python scripts/smoke_run.py
```

Kết quả mong đợi (24/24 checks PASSED):

```
=======================================================
  SMOKE RUN - Lab3_DL Pipeline Verification
=======================================================
  1. Configuration                    3/3  PASS
  2. Dataset Loading & Tokenization   6/6  PASS
  3. Model + Freeze                   3/3  PASS
  4. Forward Pass                     3/3  PASS
  5. Backward + Optimizer             3/3  PASS
  6. Checkpoint Save/Reload           1/1  PASS
  7. Exercise 1 - SentimentAnalyzer   5/5  PASS
=======================================================
  SMOKE RUN: ALL 24 CHECKS PASSED
=======================================================
```

## 5. Chạy Notebook báo cáo chính

```powershell
jupyter notebook Lab3_Practice3.ipynb
```

Notebook chứa đầy đủ cả 2 bài tập:
- **Part 1 — Exercise 1:** Sentiment Analysis (Steps 1-4)
- **Part 2 — Exercise 2:** Fine-tuning DistilBERT (Steps 1-7 + Freeze demo + Confusion Matrix + Inference demo)

## 6. Exercise 1 — Chạy CLI

### Phân tích câu mặc định

```powershell
python main.py --mode inference
```

### Phân tích câu tùy chỉnh

```powershell
python main.py --mode inference --text "This movie was absolutely terrible."
```

### Lưu kết quả ra reports/

```powershell
python main.py --mode inference --save
```

Kết quả mong đợi:

```
Sentence  : I really enjoyed learning how to use Hugging Face Transformers.
Tokens    : ['[CLS]', 'i', 'really', 'enjoyed', 'learning', ...]
Sentiment : POSITIVE
Confidence: 99.92%
NEGATIVE  : 0.08%
POSITIVE  : 99.92%
```

## 7. Exercise 2 — Evaluate model đã lưu (không train lại)

```powershell
python main.py --mode eval --model saved_model/
```

Lệnh này sẽ:
- Load model từ `saved_model/`
- Tokenize 50,000 mẫu IMDb và tái tạo test split (7,500 mẫu)
- Chạy inference trên toàn bộ test set
- In ra Accuracy, Precision, Recall, F1
- Sinh Confusion Matrix PNG -> `artifacts/figures/`
- Lưu metrics -> `reports/`

> Thời gian chạy: ~15 phút trên CPU, ~2 phút trên GPU.

## 8. Exercise 2 — Full Training (nếu muốn train lại)

```powershell
# Với freeze backbone (mặc định — chỉ train classifier head)
python main.py --mode train

# Full fine-tuning (không freeze)
python main.py --mode train --no-freeze

# Chạy nhanh 200 mẫu để kiểm tra pipeline
python main.py --mode train --smoke
```

> ⚠️ Full training trên CPU mất ~8-12 giờ. Khuyến nghị dùng GPU hoặc Google Colab.

## 9. Sinh PDF Report

```powershell
python scripts/generate_report.py
```

Output: `reports/YYYYMMDD_HHMMSS_lab3_report.pdf`

## 10. Đường dẫn kết quả

| Loại output | Đường dẫn |
|---|---|
| **Notebook báo cáo** | `Lab3_Practice3.ipynb` |
| **Model đã train** | `saved_model/` |
| **Test metrics JSON** | `reports/test_metrics.json` |
| **Training history CSV** | `reports/training_history.csv` |
| **Loss/Accuracy curves** | `artifacts/figures/YYYYMMDD_training_curves.png` |
| **Confusion Matrix** | `artifacts/figures/YYYYMMDD_confusion_matrix.png` |
| **PDF Report** | `reports/YYYYMMDD_lab3_report.pdf` |

## 11. Google Colab

```python
# Bước 1: Upload project lên Drive hoặc clone repo
# Bước 2: Cài thư viện
!pip install -r requirements.txt

# Bước 3: Upload IMDB Dataset.csv vào data/

# Bước 4: Smoke run
!python scripts/smoke_run.py

# Bước 5: Chạy notebook hoặc CLI
!python main.py --mode eval --model saved_model/

# Bước 6: Sinh báo cáo
!python scripts/generate_report.py
```
