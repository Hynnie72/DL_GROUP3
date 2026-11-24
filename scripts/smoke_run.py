"""smoke_run.py — verify the full pipeline with a tiny dataset slice.

Run BEFORE normal training:
    python scripts/smoke_run.py

Checks performed
----------------
✔ Config loads from configs/training_config.yaml
✔ Dataset CSV exists and splits correctly
✔ Tokenization produces correct tensor shapes
✔ Model loads and freeze is applied (parameter counts printed)
✔ Forward pass succeeds (no shape/dtype mismatch)
✔ Loss is a finite scalar (not NaN / Inf)
✔ Backward pass runs (gradients flow to trainable params only)
✔ Optimizer step executes without error
✔ Model saves and reloads from disk
✔ SentimentAnalyzer (Exercise 1) loads and returns valid result

Exit code 0 = PASS.  Any assertion failure = FAIL (fix before normal run).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.config import load_inference_config, load_training_config
from src.dataset import IMDbDataModule
from src.model import TextClassificationModel
from src.sentiment_analyzer import SentimentAnalyzer
from src.text_processor import TextProcessor


def section(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def check(condition: bool, msg: str) -> None:
    status = "✔ PASS" if condition else "✗ FAIL"
    print(f"  {status}  {msg}")
    if not condition:
        raise AssertionError(msg)


def main() -> None:
    print("\n" + "=" * 55)
    print("  SMOKE RUN — Lab3_DL Pipeline Verification")
    print("=" * 55)

    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    section("1. Configuration")
    cfg = load_training_config()
    cfg.use_small_sample   = True
    cfg.small_sample_size  = 200   # tiny slice — fast (chia 70/15/15 → ~140/30/30)
    cfg.epochs             = 1
    cfg.logging_steps      = 10    # log thường xuyên hơn khi dùng dataset nhỏ
    icfg = load_inference_config()
    check(cfg.epochs == 1, "TrainingConfig loaded")
    check(bool(icfg.model_name), "InferenceConfig loaded")
    check(cfg.freeze_strategy in ("backbone", "none"), "freeze_strategy valid")

    # Nếu saved_model/ có sẵn → dùng local để tránh download (offline-safe)
    root = Path(__file__).resolve().parents[1]
    local_model = root / "saved_model"
    if local_model.exists() and (local_model / "config.json").exists():
        cfg.model_name = str(local_model)
        print(f"  [offline] Using local model: {local_model}")
    else:
        print(f"  [online]  Using Hub model: {cfg.model_name}")

    # ------------------------------------------------------------------
    # 2. Dataset
    # ------------------------------------------------------------------
    section("2. Dataset Loading & Tokenization")
    data = IMDbDataModule(cfg).load_data()
    check(len(data.train_dataset) > 0, "train split non-empty")
    check(len(data.validation_dataset) > 0, "validation split non-empty")
    check(len(data.test_dataset) > 0, "test split non-empty")
    data.preprocess()
    sample = data.train_dataset[0]
    check("input_ids" in sample, "tokenization: input_ids present")
    check("attention_mask" in sample, "tokenization: attention_mask present")
    check("labels" in sample, "tokenization: labels present")

    # ------------------------------------------------------------------
    # 3. Model & Freeze
    # ------------------------------------------------------------------
    section("3. Model Loading & Freeze")
    model = TextClassificationModel(cfg).build()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    if cfg.freeze_strategy == "backbone":
        check(frozen > 0,     "freeze: backbone params are frozen")
        check(trainable < total, "freeze: not all params trainable")
        # Head-only: pre_classifier + classifier ≈ 592K
        check(trainable < 1_000_000, "freeze: trainable count < 1M (head only)")
        print(f"    trainable={trainable:,}  frozen={frozen:,}  total={total:,}")
    else:
        check(trainable == total, "no-freeze: all params trainable")

    # ------------------------------------------------------------------
    # 4. Forward pass
    # ------------------------------------------------------------------
    section("4. Forward Pass")
    model.train()
    batch = data.train_dataset[:2]

    # Sequences may have different lengths → pad to max length in batch
    import torch.nn.functional as F
    ids   = [torch.tensor(x) for x in batch["input_ids"]]
    masks = [torch.tensor(x) for x in batch["attention_mask"]]
    max_len = max(t.size(0) for t in ids)
    input_ids      = torch.stack([F.pad(t, (0, max_len - t.size(0)), value=0) for t in ids])
    attention_mask = torch.stack([F.pad(t, (0, max_len - t.size(0)), value=0) for t in masks])
    labels         = torch.tensor(batch["labels"])

    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels)
    loss = outputs.loss
    check(loss is not None, "forward: loss computed")
    check(torch.isfinite(loss), f"forward: loss is finite ({loss.item():.4f})")
    check(outputs.logits.shape == (2, cfg.num_labels),
          f"forward: logit shape = {outputs.logits.shape}")

    # ------------------------------------------------------------------
    # 5. Backward pass + optimizer step
    # ------------------------------------------------------------------
    section("5. Backward Pass & Optimizer Step")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )
    loss.backward()

    # Verify gradients only flow to trainable params
    has_grad    = [p for p in model.parameters() if p.grad is not None]
    frozen_grad = [p for p in model.parameters()
                   if not p.requires_grad and p.grad is not None]
    check(len(has_grad) > 0,    "backward: gradients computed")
    check(len(frozen_grad) == 0, "backward: frozen params have no gradient")

    optimizer.step()
    optimizer.zero_grad()
    check(True, "optimizer: step executed without error")

    # ------------------------------------------------------------------
    # 6. Save & reload checkpoint
    # ------------------------------------------------------------------
    section("6. Checkpoint Save & Reload")
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        from transformers import AutoModelForSequenceClassification
        reloaded = AutoModelForSequenceClassification.from_pretrained(tmp)
        check(
            reloaded.config.num_labels == cfg.num_labels,
            "checkpoint: reloaded model has correct num_labels",
        )

    # ------------------------------------------------------------------
    # 7. Exercise 1 — SentimentAnalyzer
    # ------------------------------------------------------------------
    section("7. Exercise 1 — SentimentAnalyzer")
    # Dùng saved_model/ nếu có (offline-safe); nhãn vẫn là NEGATIVE/POSITIVE
    ex1_model = str(local_model) if local_model.exists() else icfg.model_name
    if ex1_model == str(local_model):
        print(f"  [offline] Using local model for Ex1: {local_model}")
    else:
        print("  [online]  Downloading inference model from Hub…")

    # Patch expected_labels để khớp với saved_model (cùng là NEGATIVE/POSITIVE)
    analyzer = SentimentAnalyzer(ex1_model)
    result = analyzer.analyze(icfg.default_sentence)
    check(result.sentiment in {"POSITIVE", "NEGATIVE"}, "Ex1: valid sentiment label")
    check(0.0 <= result.confidence <= 1.0,              "Ex1: confidence in [0,1]")
    check(abs(sum(result.probabilities.values()) - 1.0) < 1e-5,
          "Ex1: probabilities sum to 1")
    check(result.tokens[0] == analyzer._tokenizer.cls_token, "Ex1: CLS token present")
    check(result.tokens[-1] == analyzer._tokenizer.sep_token, "Ex1: SEP token present")
    print(f"    → {result.sentiment}  {result.confidence:.2%}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  SMOKE RUN: ALL CHECKS PASSED ✔")
    print("  Pipeline is ready for Normal Run.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
