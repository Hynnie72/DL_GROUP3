"""Configuration loader and schema for Medical Image Regression.
Reads configs/regression_config.yaml with robust defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Tuple, List
import yaml

_DEFAULT_CFG = Path(__file__).resolve().parents[2] / "configs" / "regression_config.yaml"


@dataclass
class DataConfig:
    data_dir: str = "data/nih_chestxray"
    images_dir: str = "data/nih_chestxray/images"
    metadata_file: str = "data/nih_chestxray/Data_Entry_2017.csv"
    image_size: Tuple[int, int] = (224, 224)
    in_channels: int = 3
    normalize_target: bool = False
    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15
    use_synthetic: bool = True
    num_synthetic_samples: int = 300


@dataclass
class AugmentationConfig:
    type: Literal["none", "standard", "advanced"] = "standard"
    rotation_degrees: float = 15.0
    horizontal_flip_prob: float = 0.5
    brightness_jitter: float = 0.15
    contrast_jitter: float = 0.15
    random_affine_translate: Tuple[float, float] = (0.05, 0.05)


@dataclass
class ModelConfig:
    backbone: Literal["custom_cnn", "resnet18", "resnet34", "resnet50"] = "resnet18"
    pretrained: bool = True
    dropout_rate: float = 0.2
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    loss_fn: Literal["mse", "mae", "huber"] = "mse"
    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 4


@dataclass
class PathsConfig:
    checkpoints_dir: str = "checkpoints/regression"
    saved_model_dir: str = "saved_model/regression"
    reports_dir: str = "reports/regression"
    figures_dir: str = "artifacts/figures/regression"
    logs_dir: str = "logs/regression"


@dataclass
class RegressionConfig:
    name: str = "Medical Image Regression"
    task_type: str = "regression"
    target_name: str = "Patient Age"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def load_regression_config(path: str | Path | None = None) -> RegressionConfig:
    """Load regression config from YAML, falling back to defaults."""
    cfg_path = Path(path) if path else _DEFAULT_CFG
    if not cfg_path.exists():
        return RegressionConfig()

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    proj = raw.get("project", {})
    d_raw = raw.get("data", {})
    a_raw = raw.get("augmentation", {})
    m_raw = raw.get("model", {})
    t_raw = raw.get("training", {})
    p_raw = raw.get("paths", {})

    data_cfg = DataConfig(
        data_dir=d_raw.get("data_dir", "data/nih_chestxray"),
        images_dir=d_raw.get("images_dir", "data/nih_chestxray/images"),
        metadata_file=d_raw.get("metadata_file", "data/nih_chestxray/Data_Entry_2017.csv"),
        image_size=tuple(d_raw.get("image_size", (224, 224))),
        in_channels=d_raw.get("in_channels", 3),
        normalize_target=d_raw.get("normalize_target", False),
        train_size=d_raw.get("train_size", 0.70),
        validation_size=d_raw.get("validation_size", 0.15),
        test_size=d_raw.get("test_size", 0.15),
        use_synthetic=d_raw.get("use_synthetic", True),
        num_synthetic_samples=d_raw.get("num_synthetic_samples", 300),
    )

    aug_cfg = AugmentationConfig(
        type=a_raw.get("type", "standard"),
        rotation_degrees=float(a_raw.get("rotation_degrees", 15.0)),
        horizontal_flip_prob=float(a_raw.get("horizontal_flip_prob", 0.5)),
        brightness_jitter=float(a_raw.get("brightness_jitter", 0.15)),
        contrast_jitter=float(a_raw.get("contrast_jitter", 0.15)),
        random_affine_translate=tuple(a_raw.get("random_affine_translate", (0.05, 0.05))),
    )

    model_cfg = ModelConfig(
        backbone=m_raw.get("backbone", "resnet18"),
        pretrained=m_raw.get("pretrained", True),
        dropout_rate=float(m_raw.get("dropout_rate", 0.2)),
        freeze_backbone=m_raw.get("freeze_backbone", False),
    )

    training_cfg = TrainingConfig(
        loss_fn=t_raw.get("loss_fn", "mse"),
        epochs=int(t_raw.get("epochs", 8)),
        batch_size=int(t_raw.get("batch_size", 16)),
        learning_rate=float(t_raw.get("learning_rate", 1.0e-4)),
        weight_decay=float(t_raw.get("weight_decay", 1.0e-4)),
        optimizer=t_raw.get("optimizer", "adamw"),
        scheduler=t_raw.get("scheduler", "cosine"),
        gradient_clip_val=float(t_raw.get("gradient_clip_val", 1.0)),
        early_stopping_patience=int(t_raw.get("early_stopping_patience", 4)),
    )

    paths_cfg = PathsConfig(
        checkpoints_dir=p_raw.get("checkpoints_dir", "checkpoints/regression"),
        saved_model_dir=p_raw.get("saved_model_dir", "saved_model/regression"),
        reports_dir=p_raw.get("reports_dir", "reports/regression"),
        figures_dir=p_raw.get("figures_dir", "artifacts/figures/regression"),
        logs_dir=p_raw.get("logs_dir", "logs/regression"),
    )

    return RegressionConfig(
        name=proj.get("name", "Medical Image Regression"),
        task_type=proj.get("task_type", "regression"),
        target_name=proj.get("target_name", "Patient Age"),
        seed=int(proj.get("seed", 42)),
        data=data_cfg,
        augmentation=aug_cfg,
        model=model_cfg,
        training=training_cfg,
        paths=paths_cfg,
    )
