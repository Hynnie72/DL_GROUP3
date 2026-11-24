"""main.py — CLI entry point for Lab3_DL.

Usage
-----
# Exercise 1 — inference
python main.py --mode inference
python main.py --mode inference --text "This film was fantastic!"
python main.py --mode inference --save           # save result to reports/
python main.py --mode inference --save-model     # save model to saved_model/

# Exercise 2 — fine-tune
python main.py --mode train
python main.py --mode train --no-freeze          # full fine-tuning
python main.py --mode train --smoke              # 200-sample quick test

# Load saved model and evaluate only (no retraining)
python main.py --mode eval --model saved_model/
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Exercise 1
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> None:
    from src.config import load_inference_config
    from src.result_saver import ResultSaver
    from src.sentiment_analyzer import SentimentAnalyzer

    cfg  = load_inference_config()
    text = args.text or cfg.default_sentence

    print(f"\n{'='*55}\n  Exercise 1 — Sentiment Analysis\n{'='*55}")
    model_src = args.model or cfg.model_name
    analyzer  = SentimentAnalyzer(model_src)

    encoded = analyzer.tokenize(text)
    result  = analyzer.analyze(text)

    print(f"\n  Loaded model : {analyzer.model_name}")
    print(f"  Sentence     : {result.sentence}")
    print(f"  Tokens       : {result.tokens}")
    print(f"  Token IDs    : {encoded['input_ids'].tolist()}")
    print(f"  Attn mask    : {encoded['attention_mask'].tolist()}")
    print(f"\n  Sentiment    : {result.sentiment}")
    print(f"  Confidence   : {result.confidence:.2%}")
    print(  "  Probabilities:")
    for label, prob in result.probabilities.items():
        print(f"    {label}: {prob:.2%}")

    if args.save:
        ResultSaver().save_inference([result])

    if getattr(args, "save_model", False):
        analyzer.save()

    print("\n  All Exercise 1 steps completed. ✔\n")


# ---------------------------------------------------------------------------
# Exercise 2 — Training
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> None:
    from src.config import load_training_config
    from src.dataset import IMDbDataModule
    from src.evaluator import ModelEvaluator
    from src.model import TextClassificationModel
    from src.result_saver import ResultSaver
    from src.trainer import FineTuningTrainer

    cfg = load_training_config()
    if args.no_freeze:
        cfg.freeze_strategy = "none"
    if args.smoke:
        cfg.use_small_sample  = True
        cfg.small_sample_size = 200

    _seed_everything(cfg.seed)
    device = "GPU" if torch.cuda.is_available() else "CPU"

    print(f"\n{'='*55}\n  Exercise 2 — Fine-tuning\n{'='*55}")
    print(f"  Device          : {device}")
    print(f"  Freeze strategy : {cfg.freeze_strategy}")
    print(f"  Small sample    : {cfg.use_small_sample}")

    print("\n[1] Loading dataset …")
    data = IMDbDataModule(cfg).load_data()

    print("\n[2] Tokenizing …")
    data.preprocess()

    print("\n[3] Loading pretrained model …")
    model = TextClassificationModel(cfg).build()

    print("\n[4-5] Setting up Trainer …")
    trainer = FineTuningTrainer(cfg, model, data.tokenizer,
                                data.train_dataset, data.validation_dataset)

    print("\n[6] Fine-tuning …")
    trainer.train()

    print("\n[7] Evaluating on test set …")
    metrics = ModelEvaluator(trainer.trainer).evaluate(data.test_dataset)
    print("\n  Test metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    saved_path = trainer.save()

    saver = ResultSaver(reports_dir=cfg.reports_dir, figures_dir=cfg.figures_dir)
    saver.save_training_metrics(metrics)
    history_csv = Path(cfg.reports_dir) / "training_history.csv"
    if history_csv.exists():
        saver.save_training_curves(history_csv)

    print(f"\n  Model saved → {saved_path}")
    print("  All Exercise 2 steps completed. ✔\n")


# ---------------------------------------------------------------------------
# Evaluate only (load saved model, no retraining)
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace) -> None:
    from src.config import load_training_config
    from src.dataset import IMDbDataModule
    from src.evaluator import ModelEvaluator
    from src.result_saver import ResultSaver
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    cfg = load_training_config()
    _seed_everything(cfg.seed)
    model_path = args.model or "saved_model"

    print(f"\n{'='*55}\n  Evaluate — Loading {model_path}\n{'='*55}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    cfg.model_name = model_path
    data = IMDbDataModule(cfg).load_data()
    data.preprocess()

    eval_args = TrainingArguments(output_dir="checkpoints", report_to="none")
    trainer   = Trainer(model=model, args=eval_args, processing_class=tokenizer)
    evaluator = ModelEvaluator(trainer)

    print("\n[7] Evaluating on test set …")
    metrics = evaluator.evaluate(data.test_dataset)
    print("\n  Test metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    print("\n  Computing Confusion Matrix …")
    evaluator.confusion_matrix(data.test_dataset, figures_dir=cfg.figures_dir)

    saver = ResultSaver(reports_dir=cfg.reports_dir, figures_dir=cfg.figures_dir)
    saver.save_training_metrics(metrics)

    history_csv = Path(cfg.reports_dir) / "training_history.csv"
    if history_csv.exists():
        saver.save_training_curves(history_csv)

    print("\n  Evaluation complete. ✔\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lab3_DL — Hugging Face Sentiment Analysis & Fine-tuning"
    )
    p.add_argument("--mode", choices=["inference", "train", "eval"],
                   default="inference")
    # Inference options
    p.add_argument("--text", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--save", action="store_true")
    p.add_argument("--save-model", dest="save_model", action="store_true")
    # Training options
    p.add_argument("--no-freeze", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dispatch = {"inference": run_inference, "train": run_training, "eval": run_eval}
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
