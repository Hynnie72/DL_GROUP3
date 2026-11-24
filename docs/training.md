# Training — Lab 3 Practice 3

## Exercise 1 — Không có training

Exercise 1 chỉ thực hiện **inference**. Không có optimizer, loss, gradient hay weight update.

```
Load pre-trained model (distilbert-base-uncased-finetuned-sst-2-english)
        |
        v
model.eval()           ← tắt dropout, batchnorm chuyển sang eval mode
        |
        v
torch.no_grad()        ← tắt gradient computation, tiết kiệm bộ nhớ
        |
        v
Tokenize sentence      ← input_ids (1, L), attention_mask (1, L)
        |
        v
model.forward()        ← logits (1, 2)
        |
        v
F.softmax(logits)      ← probabilities [p_NEG, p_POS]
        |
        v
argmax -> label + confidence
```

---

## Exercise 2 — Fine-tuning Pipeline

### Luồng training tổng thể

```
IMDb CSV (50,000 rows)
  |
  |-- [1] Load & Split
  |       train_test_split(stratify, seed=42)
  |       -> train: 34,999 | val: 7,501 | test: 7,500
  |
  |-- [2] Tokenize
  |       DistilBERT WordPiece, max_length=256, truncation=True
  |       -> HuggingFace Dataset với input_ids, attention_mask, labels
  |
  |-- [3] Build Model
  |       load distilbert-base-uncased từ Hub
  |       _freeze_backbone() -> 592,130 trainable params
  |
  |-- [4] Training Arguments
  |       epochs=2, batch_size=8, lr=2e-5, seed=42
  |       save_strategy="epoch", eval_strategy="epoch"
  |
  |-- [5] Trainer.train()
  |       Epoch 1: forward -> CrossEntropy loss -> backward -> AdamW.step
  |       Epoch 2: forward -> CrossEntropy loss -> backward -> AdamW.step
  |       EpochMetricsCallback: tính train+val metrics sau mỗi epoch
  |
  |-- [6] Save
  |       saved_model/ <- model.safetensors + config + tokenizer
  |
  v-- [7] Evaluate on Test Set (1 lần duy nhất)
          Trainer.predict() -> logits -> argmax
          accuracy, precision, recall, F1, confusion matrix
```

### Hyperparameters

| Parameter | Value | Ghi chú |
|---|---|---|
| **Pretrained model** | `distilbert-base-uncased` | Backbone |
| **Epochs** | 2 | |
| **Batch size** | 8 | Per device |
| **Learning rate** | 2e-5 | |
| **Weight decay** | 0.01 | L2 regularization |
| **Max grad norm** | 1.0 | Gradient clipping |
| **Optimizer** | AdamW | HF Trainer default |
| **LR Scheduler** | Linear decay | HF Trainer default |
| **Seed** | 42 | random, numpy, torch, transformers |
| **Max seq length** | 256 | truncation |
| **Freeze strategy** | backbone | 592,130 trainable params |
| **Early stopping** | Không áp dụng | |

### Checkpoint Strategy

| Checkpoint | Lưu khi nào | Đường dẫn |
|---|---|---|
| **Epoch checkpoint** | Sau mỗi epoch | `checkpoints/checkpoint-XXXX/` |
| **Last model** | Sau training kết thúc | `saved_model/` |

> Best checkpoint (val_loss thấp nhất) là Epoch 1 (val_loss=0.2629). Muốn lưu best model cần thêm `load_best_model_at_end=True` trong TrainingArguments.

### Logging

| Item | Ghi vào đâu |
|---|---|
| Step loss | Console (logging_steps=100) |
| Per-epoch Train + Val metrics | Console (EpochMetricsCallback) |
| Training history | `reports/training_history.csv` |
| Test metrics | `reports/test_metrics.json` |
| Training curves PNG | `artifacts/figures/YYYYMMDD_training_curves.png` |
| Confusion Matrix PNG | `artifacts/figures/YYYYMMDD_confusion_matrix.png` |

### Smoke Run vs Normal Run

| | Smoke Run | Normal Run |
|---|---|---|
| **Dataset size** | 200 mẫu | 50,000 mẫu |
| **Epochs** | 1 | 2 |
| **Mục tiêu** | Kiểm tra pipeline không lỗi | Train đầy đủ |
| **Thời gian (CPU)** | ~2-3 phút | ~8-12 giờ |
| **Thời gian (GPU T4)** | ~30 giây | ~30-45 phút |
| **Lệnh** | `python scripts/smoke_run.py` | `python main.py --mode train` |

### Kết quả Training Thực tế (IMDb 50K, 2 epochs, GPU Colab T4)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:---:|---:|---:|---:|---:|
| 1 | 0.1547 | 94.42% | 0.2629 | 90.68% |
| 2 | 0.1045 | 97.33% | 0.3223 | 91.75% |

> Val Loss tăng từ Epoch 1 -> 2 (0.2629 -> 0.3223) là dấu hiệu overfitting nhẹ. Tuy nhiên Val Acc vẫn tăng nhẹ, và Test Acc (91.57%) rất gần Val Acc (91.75%) -> không có data leakage.
