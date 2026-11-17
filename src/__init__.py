"""Lab 1 - Fashion-MNIST classification with an MLP (OOP, many experiments)."""

from .utils import CLASS_NAMES, CLASS_NAMES_VI, DEVICE, SEED, set_seed, slugify
from .config import ExperimentConfig, configs_to_dataframe
from .model import ACTIVATIONS, MLP
from .data import FashionMNISTData
from .explorer import DatasetExplorer
from .trainer import Trainer
from .evaluator import Evaluator
from .runner import ExperimentResult, ExperimentRunner
from .experiments import (BASELINE, CORE_GROUPS, DEPTH_SIZES, SWEEP_PARAMS, SWEEPS,
                          build_experiments, build_sweep, select_experiments)

__all__ = [
    "CLASS_NAMES", "CLASS_NAMES_VI", "DEVICE", "SEED", "set_seed", "slugify",
    "ExperimentConfig", "configs_to_dataframe",
    "ACTIVATIONS", "MLP",
    "FashionMNISTData", "DatasetExplorer",
    "Trainer", "Evaluator",
    "ExperimentResult", "ExperimentRunner",
    "BASELINE", "CORE_GROUPS", "DEPTH_SIZES", "SWEEP_PARAMS", "SWEEPS",
    "build_experiments", "build_sweep", "select_experiments",
]
