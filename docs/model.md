# Model — Lab 3 Practice 3

## Exercise 1 — Inference Model

### Model ID

```
distilbert-base-uncased-finetuned-sst-2-english
```

Nguồn: Hugging Face Hub — được HuggingFace fine-tune sẵn trên SST-2 (Stanford Sentiment Treebank).

### Kiến trúc đầy đủ

```
DistilBertForSequenceClassification
│
├── distilbert
│   ├── embeddings
│   │   ├── word_embeddings     : Embedding(30522, 768)
│   │   ├── position_embeddings : Embedding(512, 768)
│   │   └── LayerNorm + Dropout(p=0.1)
│   │
│   └── transformer
│       └── layer x 6
│           ├── attention (MultiHeadSelfAttention, 12 heads, dim=64 each)
│           ├── sa_layer_norm  : LayerNorm(768)
│           ├── ffn            : Linear(768->3072) + GELU + Linear(3072->768)
│           ├── output_layer_norm : LayerNorm(768)
│           └── Dropout(p=0.1)
│
├── pre_classifier : Linear(768, 768) + ReLU
├── classifier     : Linear(768, 2)
└── dropout        : Dropout(p=0.2)
```

### Thông số tổng hợp

| Attribute | Value |
|---|---|
| Total parameters | ~66,955,010 |
| Input | Token IDs + Attention Mask |
| Output | Logits shape (1, 2) -> softmax -> probabilities |
| Output classes | 2: NEGATIVE (0), POSITIVE (1) |
| Inference mode | `model.eval()` + `torch.no_grad()` |
| Weight update | Không — inference only |

---

## Exercise 2 — Fine-tuning Model

### Backbone

```
distilbert-base-uncased
Pre-trained trên: BookCorpus (800M words) + English Wikipedia (2.5B words)
```

### Kiến trúc đầy đủ với Freeze Strategy

```
DistilBertForSequenceClassification
│
├── distilbert                              [FROZEN - freeze_strategy="backbone"]
│   ├── embeddings                          [FROZEN]
│   │   ├── word_embeddings     Embedding(30522, 768)
│   │   ├── position_embeddings Embedding(512, 768)
│   │   └── LayerNorm + Dropout
│   └── transformer (6 layers)             [FROZEN]
│       └── layer x 6: Attention + FFN + LayerNorm
│
├── pre_classifier : Linear(768 -> 768)    [TRAINABLE]
│   ├── weight : (768, 768) = 589,824 params
│   └── bias   : (768,)    =     768 params
│
├── classifier : Linear(768 -> 2)          [TRAINABLE]
│   ├── weight : (2, 768)  =   1,536 params
│   └── bias   : (2,)      =       2 params
│
└── dropout : Dropout(p=0.2)
```

### Thống kê tham số

| Nhóm | Trạng thái | Số tham số |
|---|---|---|
| distilbert.embeddings | FROZEN | ~23,837,184 |
| distilbert.transformer | FROZEN | ~42,525,696 |
| pre_classifier | TRAINABLE | 590,592 |
| classifier | TRAINABLE | 1,538 |
| **Tổng Trainable** | | **592,130 (0.88%)** |
| **Tổng Frozen** | | **66,362,880 (99.12%)** |
| **Tổng cộng** | | **66,955,010** |

### Lý do chọn Freeze Backbone

- DistilBERT đã được pre-train trên 16 GB văn bản — backbone đã mã hoá ngữ nghĩa tốt.
- Chỉ cần dạy head phân biệt POSITIVE/NEGATIVE từ vector `[CLS]`.
- Giảm nguy cơ catastrophic forgetting.
- Tốc độ train nhanh ~3x so với full fine-tuning.
- Phù hợp với dataset vừa (50K mẫu) và tài nguyên hạn chế.

### Loss Function

```
CrossEntropyLoss  (HF Trainer mặc định cho classification)
```

Tính tự động khi truyền `labels` vào `model.forward()`.

### Optimizer & Scheduler

| Attribute | Value |
|---|---|
| **Optimizer** | AdamW (HF Trainer default) |
| **Learning rate** | 2e-5 |
| **Weight decay** | 0.01 (L2 regularization) |
| **Beta1, Beta2** | 0.9, 0.999 |
| **Epsilon** | 1e-8 |
| **Max grad norm** | 1.0 (gradient clipping) |
| **LR Scheduler** | Linear decay (warmup_ratio=0.0) |

### Initial Model vs Trained Model

| | Initial Model | Trained Model |
|---|---|---|
| **Weights** | Pre-trained on BookCorpus + Wikipedia | Fine-tuned on IMDb 34,999 mẫu (2 epochs) |
| **Head** | Randomly initialised | Trained — classifier head |
| **Location** | Hugging Face Hub | `saved_model/` |
| **Test Accuracy** | ~50% (random) | **91.57%** |
| **Test F1** | ~50% (random) | **91.66%** |
