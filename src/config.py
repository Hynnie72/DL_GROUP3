"""ExperimentConfig: every hyperparameter of one experiment in a single object."""

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from .utils import SEED


@dataclass
class ExperimentConfig:
    name: str
    group: str
    hidden_sizes: Tuple[int, ...] = (256, 128)
    activation: str = "relu"          # relu | tanh | gelu | leaky_relu
    dropout: float = 0.2
    batch_norm: bool = False
    optimizer: str = "adam"           # adam | sgd
    lr: float = 1e-3
    momentum: float = 0.9             # only used by SGD
    batch_size: int = 64
    epochs: int = 30
    seed: int = SEED

    @property
    def architecture(self) -> str:
        """Layer sizes as text, e.g. '784 -> 256 -> 128 -> 10'."""
        return " -> ".join(["784", *map(str, self.hidden_sizes), "10"])

    @property
    def optimizer_text(self) -> str:
        """Optimizer and its settings as text, e.g. 'Adam (lr=0.001)'."""
        if self.optimizer == "sgd":
            return f"SGD (lr={self.lr}, momentum={self.momentum})"
        return f"Adam (lr={self.lr})"


def configs_to_dataframe(configs: List[ExperimentConfig]) -> pd.DataFrame:
    """One row per experiment, used to preview the experiment list before training."""
    return pd.DataFrame([{
        "Thí nghiệm": c.name,
        "Nhóm": c.group,
        "Kiến trúc": c.architecture,
        "Activation": c.activation,
        "Dropout": c.dropout,
        "BatchNorm": c.batch_norm,
        "Optimizer": c.optimizer_text,
        "Batch size": c.batch_size,
        "Epoch": c.epochs,
    } for c in configs])
