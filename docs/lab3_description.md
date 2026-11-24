# Lab 3 — Practice 3: Get Started with Hugging Face

## 1. General Information

| Field | Value |
|---|---|
| **Project** | Lab 3 — Practice 3: Get Started with Hugging Face |
| **Course** | Machine Learning / Deep Learning |
| **Author** | *(sinh viên điền tên)* |
| **Created** | 2026-11-05 |
| **Updated** | 2026-11-10 |
| **Framework** | Hugging Face Transformers >= 4.40 · PyTorch >= 2.0 · Python >= 3.10 |
| **Notebook** | `Lab3_Practice3.ipynb` |

---

## 2. Purpose

Bài thực hành gồm hai bài tập hướng dẫn cách sử dụng thư viện Hugging Face Transformers trong bài toán xử lý ngôn ngữ tự nhiên:

**Exercise 1 — Sentiment Analysis (inference only)**
Sử dụng model `distilbert-base-uncased-finetuned-sst-2-english` đã được fine-tune sẵn trên Hugging Face Hub để phân loại cảm xúc câu tiếng Anh thành `POSITIVE` hoặc `NEGATIVE`. Không có quá trình training mới — chỉ load model và chạy inference.

**Exercise 2 — Fine-tuning DistilBERT trên IMDb**
Fine-tune `distilbert-base-uncased` trên tập IMDb (50,000 đánh giá phim) để xây dựng bộ phân loại nhị phân POSITIVE / NEGATIVE. Áp dụng kỹ thuật đóng băng backbone (freeze) — chỉ train classifier head (~592K tham số) thay vì toàn bộ 66.9M tham số.

---

## 3. Input & Output

### Exercise 1

| Hạng mục | Mô tả |
|---|---|
| **Input** | Câu tiếng Anh đơn lẻ (string) |
| **Tokenization** | DistilBERT WordPiece, special tokens `[CLS]`…`[SEP]`, max_length=512 |
| **Tensor shape** | `input_ids`: (1, L) · `attention_mask`: (1, L) — L <= 512 |
| **Output** | `SentimentResult(sentence, sentiment, confidence, tokens, probabilities)` |
| **Saving path** | `reports/inference_results.{json,csv}` |

### Exercise 2

| Hạng mục | Mô tả |
|---|---|
| **Input** | `data/IMDB Dataset.csv` — cột `review` (text), `sentiment` (label) |
| **Tensor shape** | `input_ids`: (B, 256) · `attention_mask`: (B, 256) · `labels`: (B,) |
| **Output** | Label `NEGATIVE (0)` hoặc `POSITIVE (1)` |
| **Model saving** | `saved_model/` |
| **Metrics saving** | `reports/test_metrics.json` · `reports/training_history.csv` |
| **Figures saving** | `artifacts/figures/YYYYMMDD_HHMMSS_*.png` |

---

## 4. Model Architecture

### Exercise 1

```
distilbert-base-uncased-finetuned-sst-2-english
├── distilbert (backbone — SST-2 fine-tuned, FROZEN during inference)
│   ├── embeddings: word(30522,768) + position(512,768) + LayerNorm
│   └── transformer: 6 layers x [MultiHeadAttention(12 heads) + FFN(768→3072→768)]
├── pre_classifier : Linear(768 → 768) + ReLU
└── classifier     : Linear(768 → 2)

Total parameters : ~66,955,010
Mode             : eval() + torch.no_grad()  — NO weight update
```

### Exercise 2

```
distilbert-base-uncased  (pre-trained, fine-tuned on IMDb)
│
├── distilbert (FROZEN — freeze_strategy = "backbone")
│   ├── embeddings                          [FROZEN]
│   └── transformer (6 layers)              [FROZEN]
│
├── pre_classifier : Linear(768 → 768)      [TRAINABLE]  589,824 + 768  params
└── classifier     : Linear(768 → 2)        [TRAINABLE]    1,536 +   2  params

Trainable : 592,130  (0.88%)
Frozen    : 66,362,880  (99.12%)
Total     : 66,955,010
```

---

## 5. Success Criteria

| Criterion | Target | Achieved |
|---|---|---|
| Ex1: model load và trả về nhãn hợp lệ | NEGATIVE / POSITIVE | ✅ |
| Ex1: confidence in [0, 1] | ✅ | ✅ 99.92% |
| Ex1: probabilities sum to 1 | ✅ | ✅ |
| Ex2: Test Accuracy | >= 90% | ✅ **91.57%** |
| Ex2: Test F1-score | >= 90% | ✅ **91.66%** |
| Ex2: Val Acc ~ Test Acc (no data leak) | Delta < 1% | ✅ Delta = 0.18% |
| Ex2: Freeze backbone applied | 592K trainable | ✅ Verified smoke run |

---

## 6. Cấu trúc thư mục

```
D:\Lab3_DL\
│
├── Lab3_Practice3.ipynb          ← Notebook báo cáo chính (Ex1 + Ex2)
├── main.py                       ← CLI entry point
├── requirements.txt              ← Danh sách thư viện
│
├── configs/
│   └── training_config.yaml     ← Toàn bộ hyperparameters
│
├── data/
│   └── IMDB Dataset.csv         ← 50,000 mẫu IMDb (66 MB)
│
├── docs/                        ← Tài liệu kỹ thuật
│   ├── lab3_description.md      ← Tổng quan dự án (file này)
│   ├── data.md                  ← Mô tả dataset
│   ├── model.md                 ← Kiến trúc model
│   ├── training.md              ← Pipeline training
│   ├── evaluation.md            ← Kết quả đánh giá
│   ├── demo.md                  ← Hướng dẫn chạy
│   └── plan_progress_status.md  ← Theo dõi tiến độ
│
├── src/                         ← Source code (Python package)
│   ├── __init__.py
│   ├── config.py                ← TrainingConfig, InferenceConfig
│   ├── text_processor.py        ← Text validation
│   ├── tokenizer.py             ← HF tokenizer wrapper (Ex1)
│   ├── sentiment_analyzer.py    ← SentimentAnalyzer class (Ex1)
│   ├── dataset.py               ← IMDbDataModule (Ex2)
│   ├── model.py                 ← TextClassificationModel + freeze (Ex2)
│   ├── trainer.py               ← FineTuningTrainer (Ex2)
│   ├── evaluator.py             ← ModelEvaluator + confusion matrix (Ex2)
│   └── result_saver.py          ← JSON/CSV/PNG saver
│
├── scripts/
│   ├── smoke_run.py             ← Pipeline verification (24 checks)
│   └── generate_report.py       ← PDF report builder (fpdf2)
│
├── saved_model/                 ← Fine-tuned DistilBERT (267 MB)
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── training_args.bin
│
├── artifacts/
│   └── figures/                 ← Hình ảnh, biểu đồ
│       ├── YYYYMMDD_training_curves.png
│       └── YYYYMMDD_confusion_matrix.png
│
├── reports/                     ← Kết quả số và báo cáo
│   ├── test_metrics.json        ← Acc, Precision, Recall, F1
│   ├── training_history.csv     ← Metrics theo epoch
│   └── YYYYMMDD_lab3_report.pdf ← PDF báo cáo tự động
│
├── checkpoints/                 ← Checkpoint trung gian (nếu có)
├── logs/                        ← TensorBoard / training logs
└── tests/                       ← Unit tests
```

---

## 7. How to Run

```
Step 1 — Cài thư viện
        pip install -r requirements.txt

Step 2 — Smoke Run (kiểm tra pipeline)
        python scripts/smoke_run.py

Step 3 — Chạy notebook báo cáo
        jupyter notebook Lab3_Practice3.ipynb

Step 4 — Hoặc chạy CLI riêng từng bài
        python main.py --mode inference          # Ex1
        python main.py --mode eval --model saved_model/   # Ex2 eval
        python main.py --mode train              # Ex2 train lại (nếu cần)

Step 5 — Sinh PDF Report
        python scripts/generate_report.py
```

> **Lưu ý:** `saved_model/` đã có sẵn trọng số từ lần fine-tune hoàn chỉnh (Test Acc 91.57%). **Không cần train lại** trừ khi muốn thử nghiệm hyperparameters khác.
