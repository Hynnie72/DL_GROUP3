"""The list of 23 experiments. Each one changes exactly one factor compared with the baseline.

Groups 1-7 are run by the main notebook. Groups 8-10 complete three increasing sweeps
(hidden layers, learning rate, batch size), each run by its own notebook.
"""

from dataclasses import fields
from typing import Dict, List, Optional

from .config import ExperimentConfig

BASELINE = "Cơ sở: 2 lớp ẩn (256-128)"

# Groups run by the main notebook.
CORE_GROUPS: List[str] = [
    "1. Số lớp ẩn", "2. Độ rộng lớp ẩn", "3. Dropout", "4. Hàm kích hoạt",
    "5. Bộ tối ưu & learning rate", "6. BatchNorm", "7. Batch size",
]

# Hidden layer sizes used by the "number of hidden layers" sweep: k layers = the first k sizes.
DEPTH_SIZES = (256, 128, 64, 32, 16)

# Sweep groups: one factor increased step by step. Each sweep also uses the experiments of
# groups 1, 5 and 7 (and the baseline) that lie on the same scale.
SWEEPS: Dict[str, str] = {
    "8. Số lớp ẩn tăng dần": "num_layers",
    "9. Learning rate tăng dần": "lr",
    "10. Batch size tăng dần": "batch_size",
}

# The earlier group whose experiments are also steps of each sweep.
SWEEP_RELATED: Dict[str, str] = {
    "8. Số lớp ẩn tăng dần": "1. Số lớp ẩn",
    "9. Learning rate tăng dần": "5. Bộ tối ưu & learning rate",
    "10. Batch size tăng dần": "7. Batch size",
}

# How each sweep reads its value from a config, which config fields it is allowed to change,
# and whether its x-axis is logarithmic.
SWEEP_PARAMS = {
    "num_layers": {
        "label": "Số lớp ẩn",
        "group": "8. Số lớp ẩn tăng dần",
        "changes": {"hidden_sizes"},
        "value": lambda c: len(c.hidden_sizes),
        "on_scale": lambda c: tuple(c.hidden_sizes) == DEPTH_SIZES[:len(c.hidden_sizes)],
        "log": False,
    },
    "lr": {
        "label": "Learning rate (Adam)",
        "group": "9. Learning rate tăng dần",
        "changes": {"lr"},
        "value": lambda c: c.lr,
        "on_scale": lambda c: True,
        "log": True,
    },
    "batch_size": {
        "label": "Batch size",
        "group": "10. Batch size tăng dần",
        "changes": {"batch_size"},
        "value": lambda c: c.batch_size,
        "on_scale": lambda c: True,
        "log": True,
    },
}


def build_experiments() -> List[ExperimentConfig]:
    """All 23 experiments in 10 groups. Add a new experiment by adding one line here."""
    E = ExperimentConfig
    return [
        # Group 1: number of hidden layers (the baseline belongs to this group)
        E(BASELINE, "1. Số lớp ẩn"),
        E("1 lớp ẩn (256)", "1. Số lớp ẩn", hidden_sizes=(256,)),
        E("3 lớp ẩn (256-128-64)", "1. Số lớp ẩn", hidden_sizes=(256, 128, 64)),
        # Group 2: hidden layer width
        E("Hẹp: 2 lớp ẩn (64-32)", "2. Độ rộng lớp ẩn", hidden_sizes=(64, 32)),
        E("Rộng: 2 lớp ẩn (512-256)", "2. Độ rộng lớp ẩn", hidden_sizes=(512, 256)),
        # Group 3: Dropout
        E("Dropout 0.0", "3. Dropout", dropout=0.0),
        E("Dropout 0.5", "3. Dropout", dropout=0.5),
        # Group 4: activation function
        E("Activation Tanh", "4. Hàm kích hoạt", activation="tanh"),
        E("Activation RELU", "4. Hàm kích hoạt", activation="relu"),
        # Group 5: optimizer and learning rate
        E("Adam lr=0.0001", "5. Bộ tối ưu & learning rate", lr=1e-4),
        E("Adam lr=0.01", "5. Bộ tối ưu & learning rate", lr=1e-2),
        E("SGD lr=0.01 momentum=0.9", "5. Bộ tối ưu & learning rate", optimizer="sgd", lr=1e-2, momentum=0.9),
        # Group 6: BatchNorm
        E("Có BatchNorm", "6. BatchNorm", batch_norm=True),
        # Group 7: batch size
        E("Batch size 32", "7. Batch size", batch_size=32),
        E("Batch size 256", "7. Batch size", batch_size=256),
        # Group 8: number of hidden layers increased step by step (continues group 1: 1 -> 2 -> 3 layers)
        E("4 lớp ẩn (256-128-64-32)", "8. Số lớp ẩn tăng dần", hidden_sizes=DEPTH_SIZES[:4]),
        E("5 lớp ẩn (256-128-64-32-16)", "8. Số lớp ẩn tăng dần", hidden_sizes=DEPTH_SIZES[:5]),
        # Group 9: Adam learning rate increased step by step (fills in group 5: 0.0001, 0.001, 0.01)
        E("Adam lr=0.0005", "9. Learning rate tăng dần", lr=5e-4),
        E("Adam lr=0.005", "9. Learning rate tăng dần", lr=5e-3),
        E("Adam lr=0.05", "9. Learning rate tăng dần", lr=5e-2),
        # Group 10: batch size increased step by step (fills in group 7: 32, 64, 256)
        E("Batch size 16", "10. Batch size tăng dần", batch_size=16),
        E("Batch size 128", "10. Batch size tăng dần", batch_size=128),
        E("Batch size 512", "10. Batch size tăng dần", batch_size=512),
    ]


def select_experiments(epochs: int = 30, groups: Optional[List[str]] = None) -> List[ExperimentConfig]:
    """Build the experiment list, set the epoch count and optionally keep only some groups.

    The baseline is always kept so every group can be compared with it. Selecting a sweep group
    (8, 9 or 10) also keeps its related group (1, 5 or 7) so the sweep has every step.
    """
    experiments = build_experiments()
    for cfg in experiments:
        cfg.epochs = epochs
    if groups is not None:
        groups = list(groups) + [SWEEP_RELATED[g] for g in groups if g in SWEEP_RELATED]
        experiments = [c for c in experiments if c.group in groups or c.name == BASELINE]
    return experiments


def on_sweep(config: ExperimentConfig, baseline: ExperimentConfig, param: str) -> bool:
    """True if `config` differs from `baseline` only in the factor of sweep `param`."""
    spec = SWEEP_PARAMS[param]
    ignore = {"name", "group"} | spec["changes"]
    return spec["on_scale"](config) and all(
        getattr(config, f.name) == getattr(baseline, f.name)
        for f in fields(ExperimentConfig) if f.name not in ignore
    )


def build_sweep(param: str, epochs: int = 30) -> List[ExperimentConfig]:
    """Only the steps of one sweep, sorted by increasing value (used by the sweep notebooks).

    Example: build_sweep("lr") -> Adam lr 0.0001, 0.0005, 0.001 (baseline), 0.005, 0.01, 0.05.
    """
    experiments = select_experiments(epochs=epochs)
    baseline = next(c for c in experiments if c.name == BASELINE)
    steps = [c for c in experiments if on_sweep(c, baseline, param)]
    return sorted(steps, key=SWEEP_PARAMS[param]["value"])
