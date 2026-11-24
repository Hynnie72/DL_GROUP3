"""Medical Image Regression package for predicting continuous variables (e.g. Age, Tumor Size)
from chest X-ray images (NIH ChestX-ray14 dataset).
"""

from .config import RegressionConfig, load_regression_config
from .dataset import NIHChestXrayDataset, NIHDataModule, get_transforms, generate_synthetic_nih_data
from .model import build_regression_model, CustomMedicalCNN, PretrainedMedicalCNN
from .loss import build_loss_fn, RegressionLossType
from .trainer import RegressionTrainer
from .evaluator import RegressionEvaluator, RegressionMetrics
from .visualizer import RegressionVisualizer
from .experiment_runner import ExperimentRunner

__all__ = [
    "RegressionConfig",
    "load_regression_config",
    "NIHChestXrayDataset",
    "NIHDataModule",
    "get_transforms",
    "generate_synthetic_nih_data",
    "build_regression_model",
    "CustomMedicalCNN",
    "PretrainedMedicalCNN",
    "build_loss_fn",
    "RegressionLossType",
    "RegressionTrainer",
    "RegressionEvaluator",
    "RegressionMetrics",
    "RegressionVisualizer",
    "ExperimentRunner",
]
