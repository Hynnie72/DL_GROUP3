# Evaluation — Lab 3 Practice 3

## Exercise 1 — Functional Checks

Exercise 1 không có evaluation trên dataset. Chỉ thực hiện validation checks sau inference:

| Check | Điều kiện | Kết quả |
|---|---|---|
| Model labels | {NEGATIVE, POSITIVE} | ✅ |
| CLS token | `tokens[0] == "[CLS]"` | ✅ |
| SEP token | `tokens[-1] == "[SEP]"` | ✅ |
| Confidence range | `0.0 <= confidence <= 1.0` | ✅ |
| Probabilities sum | `|sum(p) - 1| < 1e-5` | ✅ |
| Output label | `sentiment in {NEGATIVE, POSITIVE}` | ✅ |

**Kết quả quan sát (demo sentence):**

```
Sentence   : I really enjoyed learning how to use Hugging Face Transformers.
Sentiment  : POSITIVE
Confidence : 99.92%
NEGATIVE   : 0.08%
POSITIVE   : 99.92%
```

> Đây là functional demo check, không phải statistical evaluation trên dataset.

---

## Exercise 2 — Model Evaluation

### Evaluation Pipeline

```
saved_model/ (distilbert fine-tuned trên IMDb)
        |
        v
Load model + tokenizer
        |
        v
IMDb CSV -> split test (7,500 mau, seed=42) -> tokenize
        |
        v
Trainer.predict(test_dataset)
        |
        |-- logits (7500, 2)
        |-- argmax -> predicted_labels (7500,)
        v
Compare vs true_labels (7500,)
        |
        |-- accuracy_score()
        |-- precision_recall_fscore_support()
        |-- confusion_matrix() -> PNG
        v
Luu: reports/test_metrics.json
     artifacts/figures/YYYYMMDD_confusion_matrix.png
```

### Metrics — Validation theo Epoch

| Epoch | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1 |
|:---:|---:|---:|---:|---:|---:|
| 1 | 0.2629 | 90.68% | 87.34% | 95.15% | 91.08% |
| 2 | 0.3223 | 91.75% | 91.21% | 92.40% | 91.80% |

### Metrics — Test Set (Final, chỉ chạy 1 lần)

| Metric | Value |
|---|---:|
| **Test Loss** | 0.3274 |
| **Test Accuracy** | **91.57%** |
| **Test Precision** | 90.77% |
| **Test Recall** | 92.56% |
| **Test F1-score** | **91.66%** |

> Test set gồm 7,500 mẫu, được tách ra trước khi training và chỉ dùng 1 lần duy nhất.

### Confusion Matrix (Test Set — 7,500 mẫu)

```
                     Predicted
                   NEG      POS
Actual   NEG  [  TN   |   FP  ]
         POS  [  FN   |   TP  ]
```

- **True Positive (TP):** Review thực sự POSITIVE, model dự đoán đúng POSITIVE
- **True Negative (TN):** Review thực sự NEGATIVE, model dự đoán đúng NEGATIVE
- **False Positive (FP):** Review NEGATIVE nhưng model dự đoán POSITIVE (Type I error)
- **False Negative (FN):** Review POSITIVE nhưng model dự đoán NEGATIVE (Type II error)

> File: `artifacts/figures/20261110_214819_confusion_matrix.png`

### Phân tích Overfitting / Underfitting

| Chi so | Epoch 1 | Epoch 2 | Nhan xet |
|---|---|---|---|
| Train Loss | 0.1547 | 0.1045 | Giam deu -> model dang hoc |
| Val Loss | 0.2629 | 0.3223 | Tang -> dau hieu overfitting |
| Train Acc | 94.42% | 97.33% | Tang nhanh |
| Val Acc | 90.68% | 91.75% | Tang cham |
| Gap (Train - Val Acc) | 3.74% | **5.58%** | Gap mo rong -> overfitting nhe |

**Ket luan:**
- Model bat dau overfit o Epoch 2 (Val Loss tang trong khi Train Loss giam).
- Tuy nhien, Test Acc (91.57%) rat gan Val Acc (91.75%) -> Delta = 0.18% -> khong co data leakage.
- Neu muon giam overfitting: them Early Stopping hoac tang weight_decay.

### So sánh Val vs Test

| Metric | Validation (Ep.2) | Test | Chenh lech |
|---|---:|---:|---:|
| Accuracy | 91.75% | 91.57% | 0.18% ✅ |
| F1-score | 91.80% | 91.66% | 0.14% ✅ |

Chenh lech < 0.2% -> Model generalise tot tren du lieu chua thay.

### Success Criteria

| Criterion | Target | Result | Status |
|---|---|---|---|
| Test Accuracy | >= 90% | 91.57% | ✅ PASS |
| Test F1-score | >= 90% | 91.66% | ✅ PASS |
| Val ~ Test (no leakage) | Delta < 1% | Delta = 0.18% | ✅ PASS |
| Probabilities sum to 1 | Yes | Yes | ✅ PASS |
| Freeze backbone | 592K trainable | 592,130 | ✅ PASS |

> **Quy tac quan trong:** Test set chi duoc dung 1 lan duy nhat sau khi training ket thuc. Khong dung de tune hyperparameters.
