# Plan & Progress Status — Lab 3 Practice 3

> Cập nhật: 2026-11-10 | Người phụ trách: *(tên sinh viên)*

---

## Trạng thái tổng quan

| Giai đoạn | Trạng thái | Tiến độ |
|---|---|---|
| Phase 1 — Phân tích yêu cầu | ✅ DONE | 100% |
| Phase 2 — Thiết kế cấu trúc thư mục | ✅ DONE | 100% |
| Phase 3 — Viết tài liệu (7 file docs) | ✅ DONE | 100% |
| Phase 4 — Viết source code (src/) | ✅ DONE | 100% |
| Phase 5 — Smoke Run | ✅ DONE | 100% |
| Phase 6 — Evaluation (load saved model) | ✅ DONE | 100% |
| Phase 7 — Confusion Matrix + Training Curves | ✅ DONE | 100% |
| Phase 8 — Notebook báo cáo chính | ✅ DONE | 100% |
| Phase 9 — PDF Report | ✅ DONE | 100% |

---

## Chi tiết công việc

### Exercise 1 — Sentiment Analysis

| ID | Công việc | Trạng thái | Ghi chú |
|---|---|---|---|
| E1-01 | Cài `transformers` | ✅ DONE | requirements.txt |
| E1-02 | Load pre-trained model từ HF Hub | ✅ DONE | `SentimentAnalyzer.__init__()` |
| E1-03 | Tokenize câu mẫu | ✅ DONE | `analyzer.tokenize()` |
| E1-04 | Sentiment analysis + validation checks | ✅ DONE | `analyzer.analyze()` |
| E1-05 | Lưu kết quả JSON + CSV | ✅ DONE | `ResultSaver.save_inference()` |

### Exercise 2 — Fine-tuning DistilBERT

| ID | Công việc | Trạng thái | Ghi chú |
|---|---|---|---|
| E2-01 | Cài `transformers`, `datasets`, `evaluate` | ✅ DONE | requirements.txt |
| E2-02 | Load IMDb dataset (50K, split 70/15/15) | ✅ DONE | `IMDbDataModule.load_data()` |
| E2-03 | Load pretrained model + tokenizer | ✅ DONE | `TextClassificationModel.build()` |
| E2-04 | Preprocess dataset (tokenize, max_length=256) | ✅ DONE | `IMDbDataModule.preprocess()` |
| E2-05 | Define training arguments | ✅ DONE | `TrainingArguments` trong `FineTuningTrainer` |
| E2-06 | Tao Trainer + fine-tune (2 epochs) | ✅ DONE | `FineTuningTrainer.train()` |
| E2-07 | Evaluate fine-tuned model | ✅ DONE | Test Acc 91.57%, F1 91.66% |
| E2-08 | Freeze backbone | ✅ DONE | `TextClassificationModel._freeze_backbone()` |
| E2-09 | Luu model voi timestamp | ✅ DONE | `saved_model/` |
| E2-10 | Confusion Matrix PNG | ✅ DONE | `artifacts/figures/20261110_214819_confusion_matrix.png` |
| E2-11 | Training curves PNG | ✅ DONE | `artifacts/figures/20261110_214820_training_curves.png` |

### Notebook & Infrastructure

| ID | Công việc | Trạng thái | Ghi chú |
|---|---|---|---|
| INF-01 | Cau truc thu muc chuyen nghiep | ✅ DONE | Theo spec Phase 2 |
| INF-02 | `configs/training_config.yaml` | ✅ DONE | Toan bo hyperparams |
| INF-03 | `scripts/smoke_run.py` | ✅ DONE | 24/24 checks PASSED |
| INF-04 | `scripts/generate_report.py` | ✅ DONE | fpdf2 PDF builder |
| INF-05 | Docs — `lab3_description.md` | ✅ DONE | Co cau truc thu muc |
| INF-06 | Docs — `data.md` | ✅ DONE | |
| INF-07 | Docs — `model.md` | ✅ DONE | |
| INF-08 | Docs — `training.md` | ✅ DONE | |
| INF-09 | Docs — `evaluation.md` | ✅ DONE | |
| INF-10 | Docs — `demo.md` | ✅ DONE | |
| INF-11 | Docs — `plan_progress_status.md` | ✅ DONE | File nay |
| INF-12 | Smoke Run PASS | ✅ DONE | 24 checks — 2026-11-10 |
| INF-13 | PDF Report | ✅ DONE | `reports/20261110_215209_lab3_report.pdf` |
| INF-14 | Notebook bao cao chinh | ✅ DONE | `Lab3_Practice3.ipynb` (Ex1 + Ex2 gop) |

---

## Vấn đề đã ghi nhận

| ID | Vấn đề | Mức độ | Giải pháp |
|---|---|---|---|
| ISS-01 | `saved_model/` train voi `freeze_strategy="none"` (full fine-tuning) | Thap | Ghi chu trong docs — khong anh huong ket qua |
| ISS-02 | Best checkpoint (Epoch 1, val_loss thap nhat) khong luu rieng | Thap | Chi co last checkpoint — ghi nhan trong training.md |

---

## Ket qua cuoi cung

| Metric | Gia tri |
|---|---|
| Test Accuracy | **91.57%** (muc tieu >= 90%) |
| Test F1-score | **91.66%** (muc tieu >= 90%) |
| Val vs Test gap | **0.18%** (khong co data leakage) |
| Smoke Run | **24/24 checks PASSED** |
| Trainable params | **592,130 (0.88%)** — backbone frozen |

---

## Cong viec tiep theo

> 🎉 **Tat ca cong viec chinh da hoan thanh!**

### Tuy chon (khong bat buoc):
1. 🔲 Verify notebook bang `jupyter nbconvert --execute` (can internet + GPU)
2. 🔲 Nop bai: `Lab3_Practice3.ipynb` + thu muc du an
