"""
tests/test_pipeline.py — Unit tests cho Lab 3 Practice 3

Chạy: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
import torch

# Thêm root vào sys.path để import src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────
# Test 1 — Config
# ─────────────────────────────────────────────────────────────
class TestConfig:
    def test_config_loads(self):
        from src.config import TrainingConfig
        cfg = TrainingConfig()
        assert cfg is not None

    def test_freeze_strategy_valid(self):
        from src.config import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.freeze_strategy in ("backbone", "none", "all")

    def test_epochs_positive(self):
        from src.config import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.epochs > 0

    def test_batch_size_positive(self):
        from src.config import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.batch_size > 0

    def test_learning_rate_range(self):
        from src.config import TrainingConfig
        cfg = TrainingConfig()
        assert 1e-7 < cfg.learning_rate < 1e-1


# ─────────────────────────────────────────────────────────────
# Test 2 — Text Processor
# ─────────────────────────────────────────────────────────────
class TestTextProcessor:
    def test_valid_sentence_returns_cleaned(self):
        from src.text_processor import TextProcessor
        tp = TextProcessor()
        result = tp.clean("  Hello world  ")
        assert result == "Hello world"

    def test_empty_string_raises(self):
        from src.text_processor import TextProcessor
        tp = TextProcessor()
        with pytest.raises(ValueError):
            tp.clean("")

    def test_whitespace_only_raises(self):
        from src.text_processor import TextProcessor
        tp = TextProcessor()
        with pytest.raises(ValueError):
            tp.clean("   ")

    def test_non_string_raises(self):
        from src.text_processor import TextProcessor
        tp = TextProcessor()
        with pytest.raises(TypeError):
            tp.clean(123)


# ─────────────────────────────────────────────────────────────
# Test 3 — Dataset paths
# ─────────────────────────────────────────────────────────────
class TestDataPaths:
    def test_imdb_csv_exists(self):
        csv = ROOT / "data" / "IMDB Dataset.csv"
        assert csv.exists(), f"IMDB Dataset.csv not found at {csv}"

    def test_saved_model_exists(self):
        sm = ROOT / "saved_model" / "model.safetensors"
        assert sm.exists(), f"saved_model not found at {sm}"

    def test_saved_model_config_exists(self):
        cfg = ROOT / "saved_model" / "config.json"
        assert cfg.exists()

    def test_test_metrics_json_exists(self):
        m = ROOT / "reports" / "test_metrics.json"
        assert m.exists(), "test_metrics.json not found"

    def test_training_history_csv_exists(self):
        h = ROOT / "reports" / "training_history.csv"
        assert h.exists(), "training_history.csv not found"

    def test_confusion_matrix_png_exists(self):
        figs = list((ROOT / "artifacts" / "figures").glob("*confusion_matrix*.png"))
        assert len(figs) > 0, "No confusion matrix PNG found in artifacts/figures/"


# ─────────────────────────────────────────────────────────────
# Test 4 — Test metrics values
# ─────────────────────────────────────────────────────────────
class TestMetrics:
    def test_test_accuracy_above_90(self):
        import json
        m = json.loads((ROOT / "reports" / "test_metrics.json").read_text())
        assert m["test_accuracy"] >= 0.90, f"Test accuracy {m['test_accuracy']:.4f} < 0.90"

    def test_test_f1_above_90(self):
        import json
        m = json.loads((ROOT / "reports" / "test_metrics.json").read_text())
        assert m["test_f1"] >= 0.90, f"Test F1 {m['test_f1']:.4f} < 0.90"

    def test_precision_positive(self):
        import json
        m = json.loads((ROOT / "reports" / "test_metrics.json").read_text())
        assert m["test_precision"] > 0

    def test_recall_positive(self):
        import json
        m = json.loads((ROOT / "reports" / "test_metrics.json").read_text())
        assert m["test_recall"] > 0


# ─────────────────────────────────────────────────────────────
# Test 5 — Model loading + freeze
# ─────────────────────────────────────────────────────────────
class TestModel:
    @pytest.fixture(scope="class")
    @classmethod
    def model(cls):
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained(
            str(ROOT / "saved_model"), local_files_only=True
        )

    def test_model_loads(self, model):
        assert model is not None

    def test_model_has_2_labels(self, model):
        assert model.config.num_labels == 2

    def test_output_labels_correct(self, model):
        labels = set(model.config.id2label.values())
        assert labels == {"NEGATIVE", "POSITIVE"}

    def test_model_forward_shape(self, model):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(ROOT / "saved_model"), local_files_only=True)
        inputs = tok("Test sentence.", return_tensors="pt", max_length=64, truncation=True)
        model.eval()
        with torch.no_grad():
            out = model(**inputs)
        assert out.logits.shape == (1, 2)

    def test_softmax_sums_to_one(self, model):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(ROOT / "saved_model"), local_files_only=True)
        inputs = tok("Great movie!", return_tensors="pt", max_length=64, truncation=True)
        model.eval()
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        assert abs(float(probs.sum()) - 1.0) < 1e-5
