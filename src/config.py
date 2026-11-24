"""
Unified configuration loader — reads configs/training_config.yaml.
All hyper-parameters and paths are centralised here; no value is
hard-coded anywhere else in the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    import yaml                           # pyyaml (optional fast path)
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Default config file location (relative to project root)
# ---------------------------------------------------------------------------
_DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs" / "training_config.yaml"


def _load_yaml(path: Path) -> dict:
    """Load YAML; fall back to a minimal built-in parser if PyYAML absent."""
    if _HAS_YAML:
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    raise ImportError(
        "PyYAML is required to load training_config.yaml.\n"
        "Install it with: pip install pyyaml"
    )


# ---------------------------------------------------------------------------
# Exercise 1 — Inference config
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    default_sentence: str = (
        "I really enjoyed learning how to use Hugging Face Transformers."
    )
    expected_labels: frozenset = field(
        default_factory=lambda: frozenset({"NEGATIVE", "POSITIVE"})
    )


# ---------------------------------------------------------------------------
# Exercise 2 — Fine-tuning config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # Model
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2

    # Paths
    data_path: str = "data/IMDB Dataset.csv"
    output_dir: str = "checkpoints"
    saved_model_dir: str = "saved_model"
    reports_dir: str = "reports"
    figures_dir: str = "artifacts/figures"
    logs_dir: str = "logs"

    # Dataset split
    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42

    # Tokenization
    max_length: int = 256

    # Training hyper-parameters
    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 100

    # Freeze strategy
    # "backbone" → freeze DistilBERT layers, train only classifier head
    # "none"     → train all parameters (full fine-tuning)
    freeze_strategy: Literal["backbone", "none"] = "backbone"

    # Sampling
    use_small_sample: bool = False
    small_sample_size: int = 200


# ---------------------------------------------------------------------------
# Factory — build from YAML
# ---------------------------------------------------------------------------

def load_inference_config(cfg_path: Path = _DEFAULT_CFG) -> InferenceConfig:
    raw = _load_yaml(cfg_path).get("inference", {})
    return InferenceConfig(
        model_name=raw.get("model_name", InferenceConfig.model_name),
        default_sentence=raw.get("default_sentence", InferenceConfig.default_sentence),
    )


def load_training_config(cfg_path: Path = _DEFAULT_CFG) -> TrainingConfig:
    raw = _load_yaml(cfg_path).get("training", {})
    cfg = TrainingConfig()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
