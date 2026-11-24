# Data — Lab 3 Practice 3

## Exercise 1 — Input Data

Exercise 1 không dùng dataset. Input là một câu tiếng Anh đơn lẻ cung cấp thủ công.

| Attribute | Value |
|---|---|
| **Demo sentence** | `"I really enjoyed learning how to use Hugging Face Transformers."` |
| **Type** | Functional demo — không phải training/benchmark dataset |
| **Training data của model** | Stanford Sentiment Treebank v2 (SST-2) — được fine-tune bởi HuggingFace |
| **Model source** | `distilbert-base-uncased-finetuned-sst-2-english` trên HF Hub |

---

## Exercise 2 — IMDb Dataset

### Nguồn dữ liệu

| Attribute | Value |
|---|---|
| **File** | `data/IMDB Dataset.csv` |
| **Nguồn gốc** | [Kaggle — IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| **Kích thước file** | ~66 MB |
| **Tổng số mẫu** | 50,000 |
| **Số lớp** | 2 (POSITIVE, NEGATIVE) |

### Cấu trúc CSV

| Cột | Kiểu | Mô tả |
|---|---|---|
| `review` | string | Nội dung đánh giá phim tiếng Anh |
| `sentiment` | string | Nhãn gốc: `"positive"` hoặc `"negative"` |

### Phân phối nhãn

| Nhãn | Số mẫu | Tỷ lệ |
|---|---|---|
| positive -> 1 | 25,000 | 50.0% |
| negative -> 0 | 25,000 | 50.0% |

> Dataset cân bằng hoàn hảo — không cần class weighting hay oversampling.

### Train / Validation / Test Split

| Tập | Tỷ lệ | Số mẫu | Dùng để |
|---|---|---|---|
| **Train** | 70% | 34,999 | Cập nhật trọng số model |
| **Validation** | 15% | 7,501 | Theo dõi chất lượng trong quá trình training |
| **Test** | 15% | 7,500 | Đánh giá cuối cùng (chỉ dùng 1 lần duy nhất) |

```python
# Stratified split với seed cố định đảm bảo tái lập được kết quả
train_val, test = train_test_split(df, test_size=0.15, stratify=labels, random_state=42)
train, val      = train_test_split(train_val, test_size=0.15/0.85, stratify=..., random_state=42)
```

### Tiền xử lý dữ liệu

```
raw CSV (50,000 rows)
  │
  ├── dropna()                      loại bỏ dòng thiếu
  ├── strip().lower()               chuẩn hoá nhãn text
  ├── label encode                  negative -> 0 | positive -> 1
  └── rename: review -> text
        │
        └── DistilBERT WordPiece Tokenizer
              ├── truncation  = True
              ├── max_length  = 256
              ├── padding     = True (DataCollatorWithPadding)
              ├── return_tensors = None  (HF Dataset format)
              └── add_special_tokens: [CLS] ... [SEP]
```

### Input shape sau tokenization

| Tensor | Shape | dtype | Mô tả |
|---|---|---|---|
| `input_ids` | (batch_size, 256) | int64 | Token IDs từ vocabulary (30,522 tokens) |
| `attention_mask` | (batch_size, 256) | int64 | 1 = real token, 0 = padding |
| `labels` | (batch_size,) | int64 | 0 = NEGATIVE, 1 = POSITIVE |

### Data Augmentation

Không áp dụng. Lý do:
- Dataset đã cân bằng hoàn hảo (50/50).
- Kích thước đủ lớn (50,000 mẫu) cho bài toán binary classification.
- DistilBERT đã được pre-train trên corpus lớn — không cần augment thêm.

### Kiểm tra chất lượng dữ liệu

| Kiểm tra | Kết quả |
|---|---|
| Missing values | 0 (sau dropna) |
| Invalid labels | 0 (chỉ có "positive" / "negative") |
| Class imbalance | Không (50% / 50%) |
| Label distribution sau split | Giữ nguyên nhờ stratify=labels |

### DataLoader / Batch

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Batch size | 8 | Per device |
| Collator | `DataCollatorWithPadding` | Dynamic padding trong batch |
| Shuffle train | Có | HF Trainer mặc định |
| Shuffle val/test | Không | Giữ thứ tự để reproducible |
