
import json
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from IPython.display import display
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import accuracy_score, classification_report

from .config import ExperimentConfig
from .data import FashionMNISTData
from .evaluator import Evaluator
from .experiments import SWEEP_PARAMS, SWEEPS, on_sweep
from .model import MLP
from .trainer import Trainer
from .utils import DEVICE, set_seed, slugify

import platform
import threading
import psutil

NOT_AVAILABLE = "Not Available"
NOT_PERFORMED = "Not Performed"

DATASET_NAME = "FashionMNIST"
JSON_SCHEMA_VERSION = 1

# Sub-directories (inside figures_dir) used for each kind of figure.
FIG_GROUP_CURVES = "group_curves"
FIG_COMPARISONS = "comparisons"
FIG_SWEEPS = "sweeps"
FIG_PER_CLASS = "per_class"
FIG_DETAILED = "detailed"
FIG_BENCHMARK = "benchmark"


# ================================================================
# GENERIC HELPERS
# ================================================================

def to_jsonable(obj: Any) -> Any:
    """Convert (nearly) any object into JSON-compatible Python types.

    Handles NumPy scalars/arrays, torch tensors, dataclasses, pandas
    objects, enums, paths, sets and tuples. NaN / Inf become ``None`` and
    unknown objects fall back to ``str(obj)``.
    """

    if obj is None or isinstance(obj, (str, bool)):
        return obj

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, (int, np.integer)):
        return int(obj)

    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None

    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())

    if torch.is_tensor(obj):
        return to_jsonable(obj.detach().cpu().tolist())

    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))

    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())

    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_jsonable(getattr(obj, f.name, None))
            for f in fields(obj)
        }

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, Enum):
        return to_jsonable(obj.value)

    if isinstance(obj, (datetime,)):
        return obj.isoformat()

    if isinstance(obj, os.PathLike):
        return os.fspath(obj)

    return str(obj)


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """Serialize an experiment config without assuming its class layout."""

    if is_dataclass(config) and not isinstance(config, type):
        base = {f.name: getattr(config, f.name, None) for f in fields(config)}
    elif hasattr(config, "__dict__"):
        base = dict(vars(config))
    else:
        base = {}

    # Attributes the runner itself relies on (may be properties).
    for attr in ("name", "group", "architecture", "optimizer_text",
                 "batch_size", "epochs", "seed"):

        if attr not in base:
            try:
                base[attr] = getattr(config, attr)
            except Exception:
                pass

    return to_jsonable(base)


def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    """Flatten nested dicts to ``a.b`` keys; lists become strings."""

    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, list):
        out[prefix] = ", ".join(str(v) for v in value)
    else:
        out[prefix] = value


def _hist_at(history: Dict[str, List[float]], key: str, index: int) -> Optional[float]:
    """Safe history lookup returning ``None`` if unavailable."""

    try:
        return float(history[key][index])
    except Exception:
        return None


def _safe_sheet_name(name: str, used: set) -> str:
    """Excel-safe, unique sheet name (<= 31 chars, no []:*?/\\)."""

    clean = re.sub(r"[\[\]:*?/\\]", "_", str(name)).strip() or "Sheet"
    clean = clean[:31]
    candidate, i = clean, 2

    while candidate.lower() in used:
        suffix = f"_{i}"
        candidate = clean[:31 - len(suffix)] + suffix
        i += 1

    used.add(candidate.lower())

    return candidate


def _lines_to_markdown(title: str, lines: Sequence[str]) -> str:
    """Convert console-style lines into simple Markdown."""

    out = [f"# {title}", ""]

    for line in lines:

        if line.startswith("  - "):
            out.append("- " + line[4:])
        elif not line.strip():
            if out[-1] != "":
                out.append("")
        else:
            if out[-1] != "":
                out.append("")
            out.append(line)

    return "\n".join(out).rstrip() + "\n"


def _fmt_delta(value: float, base: Optional[float]) -> str:
    """Format difference to the baseline in percentage points."""

    if base is None:
        return ""

    return f"{value - base:+.2f} pp vs baseline"


MIB = 1024 ** 2


class PeakRssSampler:
    """Context manager that estimates peak process RAM (RSS) usage.

    A background thread samples the resident set size while the block runs.
    ``delta_mb`` is the largest increase over the RSS at entry. The value
    is approximate: the OS allocator may reuse already-reserved memory
    (giving 0) and very short blocks can be missed between two samples.
    """

    def __init__(self, interval: float = 0.002) -> None:

        self._process = psutil.Process(os.getpid())
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.start_rss = 0
        self.peak_rss = 0

    def _run(self) -> None:

        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self._process.memory_info().rss)
            self._stop.wait(self._interval)

    def __enter__(self) -> "PeakRssSampler":

        self.start_rss = self.peak_rss = self._process.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        return self

    def __exit__(self, *exc: Any) -> None:

        self._stop.set()

        if self._thread is not None:
            self._thread.join()

        self.peak_rss = max(self.peak_rss, self._process.memory_info().rss)

    @property
    def delta_mb(self) -> float:
        """Peak RSS increase in MiB."""

        return max(0, self.peak_rss - self.start_rss) / MIB


# ================================================================
# EXPERIMENT RESULT
# ================================================================

@dataclass
class ExperimentResult:
    """Everything produced by one experiment."""

    config: ExperimentConfig
    history: Dict[str, List[float]]

    # Metrics are calculated using BEST checkpoint
    metrics: Dict[str, float]

    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray

    n_params: int
    train_seconds: float

    # Checkpoint paths
    model_path: str
    last_model_path: str

    # Best checkpoint information
    best_epoch: int
    best_loss: float


@dataclass
class ReportContext:
    """Mutable state shared by the steps of one ``create_report`` call."""

    output_dir: str
    report_path: str
    include_figures: bool
    include_sweeps: bool
    include_benchmark: bool
    include_detailed_reports: bool
    include_classification_reports: bool
    generated_files: Dict[str, List[str]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


# ================================================================
# EXPERIMENT RUNNER
# ================================================================

class ExperimentRunner:

    def __init__(
        self,
        data: FashionMNISTData,
        evaluator: Evaluator,
        baseline_name: str,
        save_dir: str = "./artifacts/saved_models",
        figures_dir: str = "./artifacts/saved_figures"
    ):

        self.data = data
        self.evaluator = evaluator
        self.baseline_name = baseline_name
        self.save_dir = save_dir

        # Directory where figures are saved
        self.figures_dir = figures_dir

        # Default root for tables / text / JSON exports
        self.output_dir = "./experiment_outputs"

        os.makedirs(save_dir, exist_ok=True)

        self.results: Dict[str, ExperimentResult] = {}

        # Figure registry: logical key -> saved PNG path
        self._figure_paths: Dict[str, str] = {}
        self._path_owner: Dict[str, str] = {}

        # Latest inference benchmark (set by benchmark_inference)
        self.benchmark_df: Optional[pd.DataFrame] = None
        self.benchmark_meta: Dict[str, Any] = {}

        # Latest checkpoint verification results, keyed (name, type)
        self._verification: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # ============================================================
    # CHECKPOINT PATH MANAGEMENT
    # ============================================================

    def _get_checkpoint_paths(
        self,
        experiment_name: str
    ) -> Dict[str, str]:
        """Return best and last checkpoint paths of one experiment."""

        experiment_dir = os.path.join(
            self.save_dir,
            slugify(experiment_name)
        )

        os.makedirs(experiment_dir, exist_ok=True)

        return {
            "directory": experiment_dir,
            "best": os.path.join(
                experiment_dir,
                "best_checkpoint.pth"
            ),
            "last": os.path.join(
                experiment_dir,
                "last_checkpoint.pth"
            )
        }

    # ============================================================
    # FIGURE SAVING
    # ============================================================

    def _save_figure(
        self,
        fig,
        filename: str,
        save: bool = True,
        dpi: int = 200,
        subdir: Optional[str] = None,
        key: Optional[str] = None,
        close: bool = False
    ) -> Optional[str]:
        """
        Save one figure as a high-resolution PNG.

        The file goes to ``figures_dir[/subdir]/<slug(filename)>.png``.
        Output directories are created automatically. If another figure
        (different ``key``) already owns the target path, a numeric suffix
        is added so unrelated figures are never overwritten; saving the same
        ``key`` again intentionally replaces its own previous file.

        Parameters
        ----------
        fig      : matplotlib figure.
        filename : base name (slugified, extension added automatically).
        save     : if False nothing is written and None is returned.
        dpi      : PNG resolution.
        subdir   : optional sub-directory inside ``figures_dir``.
        key      : logical key used in the figure registry (defaults to
                   the subdir/filename combination).
        close    : close the figure after saving.

        Returns the saved path (or None if ``save`` is False).
        """

        path = None

        if save:

            directory = (
                os.path.join(self.figures_dir, subdir)
                if subdir else self.figures_dir
            )

            os.makedirs(directory, exist_ok=True)

            key = key or f"{subdir or ''}/{filename}"
            stem = slugify(filename)

            path = os.path.join(directory, f"{stem}.png")
            counter = 2

            while self._path_owner.get(path, key) != key:
                path = os.path.join(directory, f"{stem}-{counter}.png")
                counter += 1

            fig.savefig(
                path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white"
            )

            self._path_owner[path] = key
            self._figure_paths[key] = path

            print(f"Figure saved: {path}")

        if close:
            plt.close(fig)

        return path

    def _finish_figure(
        self,
        fig,
        filename: str,
        subdir: str,
        key: str,
        save: bool = True,
        show: bool = True
    ) -> Optional[str]:
        """Save, optionally display, and always close a figure."""

        path = self._save_figure(
            fig,
            filename,
            save=save,
            subdir=subdir,
            key=key
        )

        if show:
            plt.show()

        plt.close(fig)

        return path

    def _figure_path(self, key: str) -> Optional[str]:
        """Return the saved path registered under ``key`` if it exists."""

        path = self._figure_paths.get(key)

        return path if path and os.path.isfile(path) else None

    def _run_evaluator_plot(
        self,
        plot_fn: Callable[..., Any],
        args: Tuple[Any, ...],
        filename: str,
        key: str,
        save: bool = True,
        show: bool = True
    ) -> List[str]:
        """
        Run an Evaluator plotting method and save the figures it creates.

        The Evaluator is not modified: ``plt.show`` is temporarily wrapped
        so every new figure is saved right before it would be displayed
        (figures left open without a ``show()`` call are saved afterwards).
        Display behaviour is preserved when ``show=True``.
        """

        before = set(plt.get_fignums())
        original_show = plt.show
        saved: List[str] = []

        def flush() -> None:

            new_numbers = [n for n in plt.get_fignums() if n not in before]

            for number in new_numbers:

                fig = plt.figure(number)
                idx = len(saved)

                path = self._save_figure(
                    fig,
                    filename if idx == 0 else f"{filename}_{idx + 1}",
                    save=save,
                    subdir=FIG_DETAILED,
                    key=key if idx == 0 else f"{key}_{idx + 1}"
                )

                if path:
                    saved.append(path)

            if new_numbers and show:
                original_show()

            for number in new_numbers:
                plt.close(number)

        def patched_show(*a, **k) -> None:
            flush()

        plt.show = patched_show

        try:
            plot_fn(*args)
        finally:
            plt.show = original_show

        flush()

        return saved

    # ============================================================
    # RUNNING EXPERIMENTS
    # ============================================================

    def run(
        self,
        config: ExperimentConfig,
        log_every: int = 5
    ) -> ExperimentResult:
        """
        Train one fresh model.

        - best_checkpoint: lowest evaluation/validation loss.
        - last_checkpoint: final training epoch.
        - Metrics and predictions are calculated using BEST.

        - train_seconds: total elapsed time for the entire experiment,
        including data loading, model initialization, training,
        checkpoint loading and evaluation.
        """

        # Start timing the entire experiment
        total_start = time.perf_counter()

        print(
            f"\n===== {config.name} | "
            f"{config.architecture} | "
            f"{config.optimizer_text} | "
            f"batch {config.batch_size} | "
            f"{config.epochs} epochs ====="
        )

        # 1. Reproducibility
        set_seed(config.seed)

        # 2. Data loaders
        train_loader, evaluation_loader = self.data.get_loaders(
            config.batch_size,
            config.seed
        )

        # 3. Initialize model and trainer
        model = MLP(config).to(DEVICE)

        trainer = Trainer(
            model=model,
            config=config,
            device=DEVICE
        )

        checkpoint_paths = self._get_checkpoint_paths(
            config.name
        )

        # 4. Training + checkpoint saving
        history = trainer.fit(
            train_loader=train_loader,
            evaluation_loader=evaluation_loader,
            log_every=log_every,
            checkpoint_dir=checkpoint_paths["directory"]
        )

        # 5. Checkpoint paths
        best_path = checkpoint_paths["best"]
        last_path = checkpoint_paths["last"]

        if not os.path.isfile(best_path):
            raise FileNotFoundError(
                f"BEST checkpoint not found: {best_path}"
            )

        if not os.path.isfile(last_path):
            raise FileNotFoundError(
                f"LAST checkpoint not found: {last_path}"
            )

        # 6. Load BEST checkpoint
        best_checkpoint = trainer.load_checkpoint(
            best_path,
            load_optimizer=False
        )

        best_epoch = int(best_checkpoint["epoch"])
        best_loss = float(best_checkpoint["best_loss"])

        # 7. Evaluate BEST checkpoint
        y_true, y_pred, y_prob = trainer.predict(
            evaluation_loader
        )

        metrics = self.evaluator.compute_metrics(
            y_true,
            y_pred
        )

        # Synchronize CUDA to ensure all queued GPU operations finish
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)

        # Total experiment execution time
        seconds = time.perf_counter() - total_start

        # 8. Create experiment result
        result = ExperimentResult(
            config=config,
            history=history,

            metrics=metrics,

            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,

            n_params=model.count_parameters(),
            train_seconds=seconds,

            model_path=best_path,
            last_model_path=last_path,

            best_epoch=best_epoch,
            best_loss=best_loss
        )

        self.results[config.name] = result

        # 9. Print result
        final_eval_acc = history["evaluation_accuracy"][-1]

        print("\n" + "=" * 65)
        print(f"EXPERIMENT COMPLETED: {config.name}")
        print("=" * 65)

        print(f"Best Epoch             : {best_epoch}")
        print(f"Best Evaluation Loss   : {best_loss:.6f}")

        print(
            f"Best Checkpoint Acc    : "
            f"{metrics['Accuracy (%)']:.2f}%"
        )

        print(
            f"Last Epoch Eval Acc    : "
            f"{final_eval_acc:.2f}%"
        )

        print(f"Total Execution Time   : {seconds:.2f}s")

        print(f"\nBest Checkpoint        : {best_path}")
        print(f"Last Checkpoint        : {last_path}")

        print("=" * 65)

        return result

    # ------------------------------------------------------------

    def run_all(
        self,
        configs: List[ExperimentConfig],
        log_every: int = 5
    ) -> None:
        """Run every config in order."""

        for i, config in enumerate(configs, start=1):

            print(f"\n[{i}/{len(configs)}]", end="")

            self.run(
                config,
                log_every=log_every
            )

        print("\nAll experiments completed.")

    # ============================================================
    # EXPERIMENT GROUPS
    # ============================================================

    def groups(self) -> List[str]:
        """Experiment groups in the order they were run."""

        seen = []

        for r in self.results.values():

            if r.config.group not in seen:
                seen.append(r.config.group)

        return seen

    # ============================================================
    # SUMMARY TABLE
    # ============================================================

    def summary(
        self,
        sort_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        One row per experiment.

        Main metrics represent BEST checkpoint.
        Last-epoch metrics are displayed separately.
        """

        rows = []

        for r in self.results.values():

            c = r.config
            h = r.history

            best_idx = r.best_epoch - 1
            last_idx = len(h["evaluation_loss"]) - 1

            best_train_acc = h["train_accuracy"][best_idx]
            best_eval_acc = r.metrics["Accuracy (%)"]

            last_train_acc = h["train_accuracy"][last_idx]
            last_eval_acc = h["evaluation_accuracy"][last_idx]

            rows.append({

                "Experiment": c.name,
                "Group": c.group,
                "Architecture": c.architecture,

                "Parameters": r.n_params,

                # BEST checkpoint
                "Best Epoch": r.best_epoch,
                "Best Evaluation Loss": r.best_loss,

                "Best Train Acc (%)": best_train_acc,
                "Best Evaluation Acc (%)": best_eval_acc,

                "Best Train-Evaluation Gap":
                    best_train_acc - best_eval_acc,

                "Macro Precision": r.metrics["Macro Precision"],
                "Macro Recall": r.metrics["Macro Recall"],
                "Macro F1-score": r.metrics["Macro F1-score"],

                # LAST checkpoint
                "Last Epoch": len(h["evaluation_loss"]),

                "Last Train Acc (%)": last_train_acc,
                "Last Evaluation Acc (%)": last_eval_acc,

                "Last Evaluation Loss": h["evaluation_loss"][-1],

                "Last Train-Evaluation Gap":
                    last_train_acc - last_eval_acc,

                "Training Time (s)": r.train_seconds

            })

        df = pd.DataFrame(rows)

        if sort_by:

            if sort_by not in df.columns:
                raise ValueError(
                    f"Column '{sort_by}' not found in summary."
                )

            df = df.sort_values(
                sort_by,
                ascending=False
            ).reset_index(drop=True)

        return df

    # ------------------------------------------------------------

    def best_name(
        self,
        key: str = "Accuracy (%)"
    ) -> str:
        """Name of the experiment with the highest metric."""

        return max(
            self.results,
            key=lambda n: self.results[n].metrics[key]
        )

    # ============================================================
    # COMPARISON PLOTS
    # ============================================================

    def plot_group_curves(
        self,
        group: str,
        save: bool = True,
        show: bool = True
    ) -> None:
        """
        Evaluation accuracy, evaluation loss and train accuracy.

        Baseline is displayed as a dashed line.
        """

        if group in SWEEPS:

            names = self.sweep_members(SWEEPS[group])

        else:

            names = [
                n for n, r in self.results.items()
                if r.config.group == group
                or n == self.baseline_name
            ]

        if not names:
            print(f"No experiments in group {group}.")
            return

        fig, axes = plt.subplots(
            1, 3,
            figsize=(19, 5)
        )

        specs = [
            ("evaluation_accuracy", "Evaluation Accuracy (%)"),
            ("evaluation_loss", "Evaluation Loss"),
            ("train_accuracy", "Train Accuracy (%)")
        ]

        for ax, (key, label) in zip(axes, specs):

            for n in names:

                values = self.results[n].history[key]

                is_base = n == self.baseline_name

                ax.plot(
                    range(1, len(values) + 1),
                    values,
                    linestyle="--" if is_base else "-",
                    linewidth=2.4 if is_base else 1.6,
                    color="black" if is_base else None,
                    label=n + (" (baseline)" if is_base else "")
                )

            ax.set_title(label)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.xaxis.set_major_locator(
                MaxNLocator(integer=True)
            )
            ax.grid(alpha=0.25)

        axes[0].legend(fontsize=8)

        fig.suptitle(
            f"Group: {group}",
            fontsize=14,
            fontweight="bold"
        )

        fig.tight_layout()

        self._finish_figure(
            fig,
            f"group_curves_{group}",
            FIG_GROUP_CURVES,
            f"group_curves/{group}",
            save,
            show
        )

    # ------------------------------------------------------------

    def plot_final_comparison(
        self,
        save: bool = True,
        show: bool = True
    ) -> None:
        """
        Compare BEST checkpoint accuracy and train-evaluation gap.
        """

        if not self.results:
            print("No experiments have been executed.")
            return

        df = self.summary().sort_values(
            "Best Evaluation Acc (%)"
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(18, 0.45 * len(df) + 3)
        )

        # Best checkpoint accuracy
        acc = df["Best Evaluation Acc (%)"].values

        bars = axes[0].barh(
            df["Experiment"],
            acc,
            color="tab:blue"
        )

        axes[0].set_xlim(
            max(0, acc.min() - 3),
            min(100, acc.max() + 1.5)
        )

        axes[0].set_title(
            "Best Checkpoint Evaluation Accuracy (%)"
        )
        axes[0].set_xlabel("Accuracy (%)")

        for bar, v in zip(bars, acc):

            axes[0].text(
                v + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}",
                va="center",
                fontsize=9
            )

        axes[0].grid(axis="x", alpha=0.25)

        # Best checkpoint generalization gap
        gap = df["Best Train-Evaluation Gap"].values

        bars = axes[1].barh(
            df["Experiment"],
            gap,
            color=[
                "tab:red" if g > 0 else "tab:green"
                for g in gap
            ]
        )

        axes[1].axvline(
            0,
            color="black",
            linewidth=0.8
        )

        axes[1].set_title(
            "Best Checkpoint Train - Evaluation Gap (points)"
        )
        axes[1].set_xlabel("Gap (percentage points)")

        for bar, v in zip(bars, gap):

            axes[1].text(
                v,
                bar.get_y() + bar.get_height() / 2,
                f" {v:.2f}",
                va="center",
                fontsize=9
            )

        axes[1].set_yticklabels([])
        axes[1].grid(axis="x", alpha=0.25)

        fig.tight_layout()

        self._finish_figure(
            fig,
            "final_comparison",
            FIG_COMPARISONS,
            "final_comparison",
            save,
            show
        )

    # ------------------------------------------------------------

    def _per_class_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Per-class accuracy (BEST predictions): experiments x classes."""

        names = list(self.results)

        matrix = np.array([

            self.evaluator.per_class_accuracy(
                self.results[n].y_true,
                self.results[n].y_pred
            )

            for n in names

        ])

        return names, matrix

    def plot_per_class_accuracy(
        self,
        save: bool = True,
        show: bool = True
    ) -> None:
        """
        Heatmap: rows are experiments, columns are classes.
        Uses predictions from BEST checkpoint.
        """

        if not self.results:
            print("No experiments have been executed.")
            return

        names, matrix = self._per_class_matrix()

        fig, ax = plt.subplots(
            figsize=(14, 0.55 * len(names) + 2.5)
        )

        im = ax.imshow(
            matrix,
            cmap="YlGnBu",
            vmin=matrix.min() - 2,
            vmax=100,
            aspect="auto"
        )

        ax.set_xticks(range(matrix.shape[1]))

        ax.set_xticklabels(
            self.evaluator.class_names,
            rotation=45,
            ha="right"
        )

        ax.set_xlabel("Class")
        ax.set_ylabel("Experiment")

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)

        for r in range(matrix.shape[0]):

            for c in range(matrix.shape[1]):

                ax.text(
                    c,
                    r,
                    f"{matrix[r, c]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=(
                        "white"
                        if matrix[r, c] >
                        (matrix.min() + 100) / 2 + 8
                        else "black"
                    )
                )

        plt.colorbar(
            im,
            ax=ax,
            fraction=0.03,
            pad=0.02,
            label="Per-class accuracy (%)"
        )

        ax.set_title(
            "Best Checkpoint - Per-class Accuracy (%)",
            fontsize=13,
            fontweight="bold"
        )

        fig.tight_layout()

        self._finish_figure(
            fig,
            "per_class_accuracy",
            FIG_PER_CLASS,
            "per_class_accuracy",
            save,
            show
        )

    # ============================================================
    # INCREASING PARAMETER SWEEPS
    # ============================================================

    def sweep_members(
        self,
        param: str
    ) -> List[str]:
        """
        Experiments that differ from baseline only in param.
        Sorted by increasing parameter value.
        """

        spec = SWEEP_PARAMS[param]

        base = self.results[self.baseline_name].config

        names = [
            n for n, r in self.results.items()
            if on_sweep(r.config, base, param)
        ]

        return sorted(
            names,
            key=lambda n: spec["value"](
                self.results[n].config
            )
        )

    def available_sweeps(self) -> List[str]:
        """Sweep parameters with at least two executed experiments."""

        if self.baseline_name not in self.results:
            return []

        available = []

        for param in SWEEP_PARAMS:

            try:
                if len(self.sweep_members(param)) >= 2:
                    available.append(param)
            except Exception:
                continue

        return available

    # ------------------------------------------------------------

    def sweep_table(
        self,
        param: str
    ) -> pd.DataFrame:
        """
        One row per sweep value.

        Main evaluation metrics use BEST checkpoint.
        """

        spec = SWEEP_PARAMS[param]

        df = self.summary().set_index("Experiment")

        rows = []

        for n in self.sweep_members(param):

            r = df.loc[n]

            rows.append({

                spec["label"]:
                    spec["value"](self.results[n].config),

                "Experiment": n,
                "Group": r["Group"],

                "Best Epoch": r["Best Epoch"],
                "Best Evaluation Loss": r["Best Evaluation Loss"],

                "Best Train Acc (%)": r["Best Train Acc (%)"],
                "Evaluation Acc (%)": r["Best Evaluation Acc (%)"],

                "Train-Evaluation Gap":
                    r["Best Train-Evaluation Gap"],

                "Macro F1-score": r["Macro F1-score"],

                "Last Evaluation Acc (%)":
                    r["Last Evaluation Acc (%)"],

                "Last Evaluation Loss":
                    r["Last Evaluation Loss"],

                "Parameters": r["Parameters"],
                "Training Time (s)": r["Training Time (s)"]

            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------

    def plot_sweep(
        self,
        param: str,
        save: bool = True,
        show: bool = True
    ) -> None:
        """
        Best accuracy, train-evaluation gap and evaluation curves.
        """

        spec = SWEEP_PARAMS[param]

        names = self.sweep_members(param)

        if len(names) < 2:

            print(
                f"Not enough experiments for sweep "
                f"'{spec['label']}' (at least 2 required)."
            )

            return

        x = [
            spec["value"](self.results[n].config)
            for n in names
        ]

        evaluation_acc = [
            self.results[n].metrics["Accuracy (%)"]
            for n in names
        ]

        train_acc = [
            self.results[n].history["train_accuracy"][
                self.results[n].best_epoch - 1
            ]
            for n in names
        ]

        gap = [
            tr - te
            for tr, te in zip(train_acc, evaluation_acc)
        ]

        base_x = spec["value"](
            self.results[self.baseline_name].config
        )

        tick_labels = [f"{v:g}" for v in x]

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(20, 5)
        )

        # Best accuracy
        axes[0].plot(
            x,
            evaluation_acc,
            marker="o",
            linewidth=2,
            label="Best Evaluation Accuracy"
        )

        axes[0].plot(
            x,
            train_acc,
            marker="s",
            linewidth=1.5,
            linestyle="--",
            label="Train Accuracy at Best Epoch"
        )

        for xi, v in zip(x, evaluation_acc):

            axes[0].annotate(
                f"{v:.2f}",
                (xi, v),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=9
            )

        axes[0].set_title(
            "Best Checkpoint Accuracy (%)"
        )

        axes[0].set_ylabel("Accuracy (%)")
        axes[0].margins(y=0.12)

        # Generalization gap
        axes[1].bar(
            range(len(x)),
            gap,
            color=[
                "tab:red" if g > 0 else "tab:green"
                for g in gap
            ]
        )

        axes[1].axhline(
            0,
            color="black",
            linewidth=0.8
        )

        axes[1].set_xticks(range(len(x)))
        axes[1].set_xticklabels(tick_labels)

        axes[1].set_title(
            "Best Epoch Train - Evaluation Gap"
        )
        axes[1].set_ylabel("Gap (percentage points)")

        for i, v in enumerate(gap):

            axes[1].text(
                i,
                v,
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=9
            )

        # Accuracy curves
        colors = plt.cm.viridis(
            np.linspace(0, 0.9, len(names))
        )

        for n, xi, color in zip(names, x, colors):

            values = self.results[n].history[
                "evaluation_accuracy"
            ]

            is_base = n == self.baseline_name

            axes[2].plot(
                range(1, len(values) + 1),
                values,
                color=color,
                linewidth=2.4 if is_base else 1.6,
                linestyle="--" if is_base else "-",
                label=f"{xi:g}" + (
                    " (baseline)" if is_base else ""
                )
            )

        axes[2].set_title(
            "Evaluation Accuracy per Epoch"
        )

        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Evaluation Accuracy (%)")

        axes[2].xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )

        axes[2].legend(
            title=spec["label"],
            fontsize=8
        )

        # X axis
        if spec["log"]:
            axes[0].set_xscale("log")

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(tick_labels)
        axes[0].minorticks_off()

        axes[0].axvline(
            base_x,
            color="gray",
            linestyle=":",
            alpha=0.8,
            label="Baseline"
        )

        axes[0].legend()

        for ax in axes[:2]:
            ax.set_xlabel(spec["label"])

        for ax in axes:
            ax.grid(alpha=0.25)

        fig.suptitle(
            f"Effect of increasing: {spec['label']}",
            fontsize=14,
            fontweight="bold"
        )

        fig.tight_layout()

        self._finish_figure(
            fig,
            f"sweep_{param}",
            FIG_SWEEPS,
            f"sweep/{param}",
            save,
            show
        )

    # ============================================================
    # DETAILED ANALYSIS
    # ============================================================

    def show_detail(
        self,
        name: str,
        save: bool = True,
        show: bool = True,
        save_report: bool = False,
        report_dir: Optional[str] = None,
        report_format: str = "md"
    ) -> None:
        """
        Metrics table, learning curves, classification report,
        confusion matrix and sample predictions.

        All prediction-based analyses use BEST checkpoint.

        Parameters
        ----------
        save          : save the figures as PNG (default True).
        show          : display the figures (default True).
        save_report   : also write the detailed text report to disk.
        report_dir    : directory for the report (default
                        ``<output_dir>/reports/detailed_reports``).
        report_format : 'md' or 'txt'.
        """

        r = self.results[name]

        print("=" * 90)

        print(
            f"DETAILED ANALYSIS: {name} "
            f"({r.config.architecture}, "
            f"{r.config.optimizer_text})"
        )

        print("=" * 90)

        print(f"Best Epoch: {r.best_epoch}")
        print(f"Best Evaluation Loss: {r.best_loss:.6f}")

        print(f"Best Checkpoint: {r.model_path}")
        print(f"Last Checkpoint: {r.last_model_path}")

        display(
            pd.DataFrame([
                {
                    "Experiment": name,
                    **r.metrics
                }
            ]).round(4)
        )

        self._run_evaluator_plot(
            self.evaluator.plot_learning_curves,
            (r.history, name),
            f"{name}__learning_curves",
            f"detail/{name}/learning_curves",
            save, show
        )

        print("Per-class classification report (BEST checkpoint):")

        print(self._classification_report_text(r))

        self._run_evaluator_plot(
            self.evaluator.plot_confusion_matrix,
            (r.y_true, r.y_pred, name),
            f"{name}__confusion_matrix",
            f"detail/{name}/confusion_matrix",
            save, show
        )

        self._run_evaluator_plot(
            self.evaluator.plot_predictions,
            (r.y_true, r.y_pred, r.y_prob, name),
            f"{name}__sample_predictions",
            f"detail/{name}/sample_predictions",
            save, show
        )

        if save_report:
            self.save_detail_report(
                name,
                output_dir=report_dir,
                fmt=report_format
            )

    def _render_detail_figures(
        self,
        name: str,
        save: bool = True,
        show: bool = False
    ) -> None:
        """Generate (and save) the three detail figures without printing."""

        r = self.results[name]

        self._run_evaluator_plot(
            self.evaluator.plot_learning_curves,
            (r.history, name),
            f"{name}__learning_curves",
            f"detail/{name}/learning_curves",
            save, show
        )

        self._run_evaluator_plot(
            self.evaluator.plot_confusion_matrix,
            (r.y_true, r.y_pred, name),
            f"{name}__confusion_matrix",
            f"detail/{name}/confusion_matrix",
            save, show
        )

        self._run_evaluator_plot(
            self.evaluator.plot_predictions,
            (r.y_true, r.y_pred, r.y_prob, name),
            f"{name}__sample_predictions",
            f"detail/{name}/sample_predictions",
            save, show
        )

    def _classification_report_text(self, r: ExperimentResult) -> str:
        """Classification report computed from BEST predictions."""

        return classification_report(
            r.y_true,
            r.y_pred,
            labels=list(range(10)),
            target_names=self.evaluator.class_names,
            digits=4,
            zero_division=0
        )

    # ============================================================
    # SAVE / LOAD CHECKPOINTS
    # ============================================================

    def load_model(
        self,
        name: str,
        checkpoint_type: str = "best"
    ) -> MLP:
        """
        Load BEST or LAST checkpoint.

        checkpoint_type:
            'best' -> lowest evaluation loss
            'last' -> final training epoch
        """

        if name not in self.results:

            raise KeyError(
                f"Experiment '{name}' not found."
            )

        r = self.results[name]

        if checkpoint_type == "best":

            path = r.model_path

        elif checkpoint_type == "last":

            path = r.last_model_path

        else:

            raise ValueError(
                "checkpoint_type must be 'best' or 'last'."
            )

        if not os.path.isfile(path):

            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        # Rebuild model
        model = MLP(r.config).to(DEVICE)

        # Load checkpoint
        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        print("\nCheckpoint loaded successfully.")

        print(f"Experiment : {name}")
        print(f"Type       : {checkpoint_type.upper()}")
        print(f"Path       : {path}")
        print(f"Epoch      : {checkpoint['epoch']}")
        print(f"Best Loss  : {checkpoint['best_loss']:.6f}")

        return model

    # ------------------------------------------------------------

    def _verify_checkpoint(
        self,
        name: str,
        checkpoint_type: str = "best"
    ) -> Dict[str, Any]:
        """
        Reload a checkpoint, re-evaluate it and compare with expectations.

        BEST is compared with BEST metrics.
        LAST is compared with final-epoch history.
        The result is stored for later export and returned.
        """

        r = self.results[name]

        model = self._load_model_quiet(name, checkpoint_type)

        _, evaluation_loader = self.data.get_loaders(
            r.config.batch_size,
            r.config.seed
        )

        trainer = Trainer(
            model,
            r.config,
            device=DEVICE
        )

        y_true, y_pred, _ = trainer.predict(
            evaluation_loader
        )

        acc = accuracy_score(y_true, y_pred) * 100

        if checkpoint_type == "best":
            expected_acc = r.metrics["Accuracy (%)"]
        else:
            expected_acc = r.history["evaluation_accuracy"][-1]

        difference = abs(acc - expected_acc)

        verified = bool(np.isclose(acc, expected_acc, atol=1e-4))

        result = {
            "experiment": name,
            "checkpoint_type": checkpoint_type.upper(),
            "path": (
                r.model_path if checkpoint_type == "best"
                else r.last_model_path
            ),
            "reloaded_accuracy": float(acc),
            "expected_accuracy": float(expected_acc),
            "difference": float(difference),
            "status": (
                "VERIFIED SUCCESSFULLY" if verified
                else "WARNING - ACCURACY MISMATCH"
            ),
        }

        self._verification[(name, checkpoint_type.lower())] = result

        return result

    def verify_saved_model(
        self,
        name: str,
        checkpoint_type: str = "best",
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reload checkpoint and verify its accuracy.

        BEST is compared with BEST metrics.
        LAST is compared with final-epoch history.

        The result is printed, remembered for the PDF report, optionally
        saved to ``save_path`` and returned as a dictionary.
        """

        # Keeps the verbose loading output of the original behaviour.
        self.load_model(name, checkpoint_type=checkpoint_type)

        res = self._verify_checkpoint(name, checkpoint_type)

        print("\n" + "=" * 60)
        print(f"VERIFY {res['checkpoint_type']} CHECKPOINT")
        print("=" * 60)

        print(f"Experiment         : {name}")
        print(f"Reloaded Accuracy  : {res['reloaded_accuracy']:.4f}%")
        print(f"Expected Accuracy  : {res['expected_accuracy']:.4f}%")

        print(
            f"Difference         : "
            f"{res['difference']:.8f} percentage points"
        )

        print(f"STATUS: {res['status']}")

        print("=" * 60)

        if save_path:
            self._write_text(
                save_path,
                self._format_verification([res], self._ext(save_path))
            )

        return res

    # ============================================================
    # AUTOMATIC CONCLUSIONS
    # ============================================================

    def build_conclusions(self) -> List[str]:
        """Return the automatic conclusions as console-style lines."""

        if not self.results:
            return ["No experiments have been executed."]

        lines: List[str] = []

        base = self.results.get(self.baseline_name)
        base_acc = base.metrics["Accuracy (%)"] if base else None

        if base:
            lines.append(
                f"Baseline model '{self.baseline_name}': "
                f"Best Accuracy {base_acc:.2f}%, "
                f"Best Evaluation Loss {base.best_loss:.6f}, "
                f"Macro F1 {base.metrics['Macro F1-score']:.4f}."
            )
        else:
            lines.append(
                f"Baseline '{self.baseline_name}' was not found; "
                f"comparisons with the baseline are not available."
            )

        lines.append("")

        # Group conclusions
        for group in self.groups():

            members = [
                n for n, r in self.results.items()
                if r.config.group == group
                and n != self.baseline_name
            ]

            if not members:
                continue

            best = max(
                members,
                key=lambda n: self.results[n].metrics["Accuracy (%)"]
            )

            worst = min(
                members,
                key=lambda n: self.results[n].metrics["Accuracy (%)"]
            )

            b_acc = self.results[best].metrics["Accuracy (%)"]
            w_acc = self.results[worst].metrics["Accuracy (%)"]

            lines.append(f"Group {group}:")

            if len(members) == 1:

                lines.append(
                    f"  - {best}: {b_acc:.2f}% "
                    f"({_fmt_delta(b_acc, base_acc)})".replace(" ()", "")
                )

            else:

                lines.append(
                    f"  - Highest: {best} ({b_acc:.2f}%, "
                    f"{_fmt_delta(b_acc, base_acc)})".replace(", )", ")")
                )

                lines.append(
                    f"  - Lowest: {worst} ({w_acc:.2f}%, "
                    f"{_fmt_delta(w_acc, base_acc)})".replace(", )", ")")
                )

        # Best experiment
        best_all = self.best_name()

        r = self.results[best_all]

        lines.append("")
        lines.append(
            f"Experiment with the highest Best Accuracy: "
            f"{best_all} "
            f"({r.metrics['Accuracy (%)']:.2f}%, "
            f"Macro F1 {r.metrics['Macro F1-score']:.4f})."
        )

        # Lowest validation loss
        best_loss_name = min(
            self.results,
            key=lambda n: self.results[n].best_loss
        )

        best_loss_result = self.results[best_loss_name]

        lines.append(
            f"Experiment with the lowest Best Evaluation Loss: "
            f"{best_loss_name} "
            f"({best_loss_result.best_loss:.6f}, "
            f"epoch {best_loss_result.best_epoch})."
        )

        # Overfitting analysis at best epoch
        df = self.summary()

        most_overfit = df.loc[
            df["Best Train-Evaluation Gap"].idxmax()
        ]

        lines.append(
            f"Experiment with the largest Train - Evaluation gap "
            f"at its best epoch: "
            f"{most_overfit['Experiment']} "
            f"({most_overfit['Best Train-Evaluation Gap']:.2f} points)."
        )

        # Per-class analysis
        _, pca = self._per_class_matrix()

        mean_per_class = pca.mean(axis=0)

        hardest = int(np.argmin(mean_per_class))
        easiest = int(np.argmax(mean_per_class))

        lines.append(
            f"Hardest class (mean over experiments): "
            f"{self.evaluator.class_names[hardest]} "
            f"({mean_per_class[hardest]:.1f}%); "
            f"easiest: "
            f"{self.evaluator.class_names[easiest]} "
            f"({mean_per_class[easiest]:.1f}%)."
        )

        # Sweep conclusions
        for param in self.available_sweeps():

            lines.append("")
            lines.extend(self.build_sweep_conclusion(param))

        return lines

    def print_conclusions(
        self,
        save_path: Optional[str] = None
    ) -> None:
        """
        Print key numbers for writing the report.

        If ``save_path`` is given (``.md`` or ``.txt``) the same
        conclusions are also written to that file.
        """

        lines = self.build_conclusions()

        for line in lines:
            print(line)

        if save_path:
            self._write_lines(save_path, "Experiment Conclusions", lines)

    # ------------------------------------------------------------

    def build_sweep_conclusion(self, param: str) -> List[str]:
        """Return the key numbers of one increasing sweep as lines."""

        spec = SWEEP_PARAMS[param]

        if self.baseline_name not in self.results:
            return [
                f"Sweep '{spec['label']}': baseline "
                f"'{self.baseline_name}' not found."
            ]

        names = self.sweep_members(param)

        if len(names) < 2:

            return [
                f"Not enough experiments for sweep "
                f"'{spec['label']}' (at least 2 required)."
            ]

        label = spec["label"]

        rows = [
            (
                spec["value"](self.results[n].config),
                self.results[n]
            )
            for n in names
        ]

        accs = [
            r.metrics["Accuracy (%)"]
            for _, r in rows
        ]

        gaps = [

            r.history["train_accuracy"][r.best_epoch - 1]
            - r.metrics["Accuracy (%)"]

            for _, r in rows

        ]

        base_acc = self.results[
            self.baseline_name
        ].metrics["Accuracy (%)"]

        lines = [
            f"Increasing {label}: "
            + " -> ".join(
                f"{v:g}: {a:.2f}%"
                for (v, _), a in zip(rows, accs)
            )
        ]

        i_best = int(np.argmax(accs))
        i_worst = int(np.argmin(accs))

        lines.append(
            f"  - Best: {label} = "
            f"{rows[i_best][0]:g} "
            f"({accs[i_best]:.2f}%, "
            f"{accs[i_best] - base_acc:+.2f} pp vs baseline)"
        )

        lines.append(
            f"  - Worst: {label} = "
            f"{rows[i_worst][0]:g} "
            f"({accs[i_worst]:.2f}%, "
            f"{accs[i_worst] - base_acc:+.2f} pp vs baseline)"
        )

        lines.append(
            f"  - Train-Evaluation gap at best epoch: "
            f"from {min(gaps):.2f} to {max(gaps):.2f} points"
        )

        lines.append(
            f"  - Training time: "
            f"from {min(r.train_seconds for _, r in rows):.0f}s "
            f"to {max(r.train_seconds for _, r in rows):.0f}s"
        )

        lines.append(
            f"  - Number of parameters: "
            f"from {min(r.n_params for _, r in rows):,} "
            f"to {max(r.n_params for _, r in rows):,}"
        )

        return lines

    def print_sweep_conclusion(
        self,
        param: str,
        save_path: Optional[str] = None
    ) -> None:
        """
        Key numbers of one increasing sweep.

        If ``save_path`` is given the lines are also written to a file.
        """

        lines = self.build_sweep_conclusion(param)

        for line in lines:
            print(line)

        if save_path:
            self._write_lines(
                save_path,
                f"Sweep Conclusion: {SWEEP_PARAMS[param]['label']}",
                lines
            )

    # ============================================================
    # INFERENCE BENCHMARK
    # ============================================================

    @torch.no_grad()
    def benchmark_inference(
        self,
        names: Optional[List[str]] = None,
        checkpoint_type: str = "best",
        batch_size: int = 256,
        n_warmup: int = 10,
        n_runs: int = 50,
        include_single_sample: bool = True,
        plot: bool = True,
        save: bool = True,
        show: bool = True,
        measure_memory: bool = True
    ) -> pd.DataFrame:
        """
        Measure inference speed and memory use of the experiments.

        - Loads the BEST (or LAST) checkpoint of every experiment.
        - Warms up before timing to remove initialisation cost.
        - Uses torch.cuda.synchronize() on GPU for accurate timing.
        - Two modes:
            * batch  : throughput (images/s) for a fixed batch size.
            * single : latency for one image at a time (batch = 1).

        The result is stored in ``self.benchmark_df`` so that
        ``create_report()`` can include it without re-running the timing.

        Parameters
        ----------
        names : experiments to measure; None = all.
        checkpoint_type : 'best' or 'last'.
        batch_size : batch size of the batch mode.
        n_warmup : warm-up runs (not counted).
        n_runs : timed repetitions.
        include_single_sample : also measure batch = 1 latency.
        plot : draw the comparison figure.
        save : save the figure under figures_dir/benchmark.
        show : display the figure.
        measure_memory : also record memory use (see ``_measure_memory``):
            model size, peak CPU RAM and, on CUDA, GPU model / peak VRAM.

        Returns
        -------
        DataFrame, one row per experiment.
        """

        if not self.results:
            print("No experiments have been executed.")
            return pd.DataFrame()

        names = names or list(self.results)

        for n in names:
            if n not in self.results:
                raise KeyError(
                    f"Experiment '{n}' not found."
                )

        use_cuda = str(DEVICE).startswith("cuda")

        def sync() -> None:
            if use_cuda:
                torch.cuda.synchronize()

        # One sample batch shared by all experiments (fair comparison)
        _, evaluation_loader = self.data.get_loaders(
            batch_size,
            self.results[names[0]].config.seed
        )

        sample_x, _ = next(iter(evaluation_loader))
        sample_x = sample_x[:batch_size].to(DEVICE)

        actual_bs = sample_x.shape[0]
        single_x = sample_x[:1]

        def time_forward(
            model: torch.nn.Module,
            x: torch.Tensor
        ) -> np.ndarray:
            """Return an array with the seconds of n_runs forward passes."""

            for _ in range(n_warmup):
                model(x)

            sync()

            times = np.empty(n_runs, dtype=np.float64)

            for i in range(n_runs):

                sync()
                t0 = time.perf_counter()

                model(x)

                sync()
                times[i] = time.perf_counter() - t0

            return times

        rows = []

        print(
            f"\nInference benchmark | device: {DEVICE} | "
            f"checkpoint: {checkpoint_type.upper()} | "
            f"batch {actual_bs} | warmup {n_warmup} | "
            f"runs {n_runs}"
        )

        for n in names:

            r = self.results[n]

            # Memory is measured in its own pass so it never disturbs timing.
            memory = (
                self._measure_memory(n, checkpoint_type, sample_x)
                if measure_memory else {}
            )

            model = self._load_model_quiet(
                n,
                checkpoint_type
            )

            # ---- Batch mode ----
            batch_times = time_forward(model, sample_x)

            batch_ms = batch_times.mean() * 1000
            batch_std_ms = batch_times.std() * 1000

            throughput = actual_bs / batch_times.mean()

            row = {
                "Experiment": n,
                "Group": r.config.group,
                "Architecture": r.config.architecture,
                "Parameters": r.n_params,

                f"Batch {actual_bs} (ms)": batch_ms,
                f"Batch {actual_bs} Std (ms)": batch_std_ms,
                "Throughput (images/s)": throughput,
                "ms/image (batch)": batch_ms / actual_bs,
            }

            # ---- Single-image mode ----
            if include_single_sample:

                single_times = time_forward(model, single_x)

                row["Single-Image Latency Mean (ms)"] = (
                    single_times.mean() * 1000
                )
                row["Single-Image Latency P95 (ms)"] = (
                    np.percentile(single_times, 95) * 1000
                )

            row.update(memory)

            # Accuracy of the BEST checkpoint (speed / quality trade-off)
            row["Evaluation Acc (%)"] = (
                r.metrics["Accuracy (%)"]
            )

            rows.append(row)

            print(
                f"  {n:<30} "
                f"{batch_ms:8.3f} ms/batch | "
                f"{throughput:12,.0f} images/s"
            )

            del model

            if use_cuda:
                torch.cuda.empty_cache()

        df = pd.DataFrame(rows).sort_values(
            "Throughput (images/s)",
            ascending=False
        ).reset_index(drop=True)

        self.benchmark_df = df
        self.benchmark_meta = {
            "device": str(DEVICE),
            "checkpoint_type": checkpoint_type.upper(),
            "batch_size": int(actual_bs),
            "n_warmup": n_warmup,
            "n_runs": n_runs,
            "include_single_sample": include_single_sample,
            "memory_measured": measure_memory,
            "system_ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
        }

        if use_cuda:
            try:
                props = torch.cuda.get_device_properties(0)
                self.benchmark_meta["gpu_name"] = props.name
                self.benchmark_meta["gpu_total_vram_gb"] = round(
                    props.total_memory / 1024 ** 3, 2
                )
            except Exception:
                pass

        if plot:
            self._plot_inference_benchmark(
                df,
                actual_bs,
                save=save,
                show=show
            )

        return df

    # ------------------------------------------------------------

    def _load_model_quiet(
        self,
        name: str,
        checkpoint_type: str = "best"
    ) -> MLP:
        """Load a checkpoint in eval mode without printing anything."""

        r = self.results[name]

        if checkpoint_type == "best":
            path = r.model_path
        elif checkpoint_type == "last":
            path = r.last_model_path
        else:
            raise ValueError(
                "checkpoint_type must be 'best' or 'last'."
            )

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        model = MLP(r.config).to(DEVICE)

        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=True
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model

    # ------------------------------------------------------------

    @staticmethod
    def _model_size_mb(model: Any, fallback_params: int) -> float:
        """Size of parameters and buffers in MiB (4 bytes/param fallback)."""

        try:
            total = sum(p.numel() * p.element_size() for p in model.parameters())
            total += sum(b.numel() * b.element_size() for b in model.buffers())
            return total / MIB
        except Exception:
            return fallback_params * 4 / MIB

    @torch.no_grad()
    def _measure_memory(
        self,
        name: str,
        checkpoint_type: str,
        sample_x: torch.Tensor
    ) -> Dict[str, float]:
        """
        Measure the memory needed to run one experiment's model.

        Columns (all in MiB):

        - ``Model Size (MB)``: parameters + buffers (exact).
        - ``CPU RAM Peak (MB)``: peak process RSS increase while loading
          the checkpoint and running one forward pass of ``sample_x``
          (approximate, see ``PeakRssSampler``).
        - ``GPU Model (MB)`` / ``GPU Peak (MB)``: CUDA memory allocated by
          the loaded model, and its peak including activations of the
          batch. Only present when running on CUDA; nothing is estimated
          on CPU.
        """

        result: Dict[str, float] = {}
        use_cuda = str(DEVICE).startswith("cuda")

        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

        with PeakRssSampler() as sampler:

            model = self._load_model_quiet(name, checkpoint_type)

            if use_cuda:
                torch.cuda.synchronize()
                after_load = torch.cuda.memory_allocated()

            model(sample_x)

            if use_cuda:
                torch.cuda.synchronize()

        result["Model Size (MB)"] = self._model_size_mb(
            model, self.results[name].n_params
        )
        result["CPU RAM Peak (MB)"] = sampler.delta_mb

        if use_cuda:
            result["GPU Model (MB)"] = (after_load - baseline) / MIB
            result["GPU Peak (MB)"] = (
                torch.cuda.max_memory_allocated() - baseline
            ) / MIB

        del model

        if use_cuda:
            torch.cuda.empty_cache()

        return result

    # ------------------------------------------------------------

    def _plot_inference_benchmark(
        self,
        df: pd.DataFrame,
        batch_size: int,
        save: bool = True,
        show: bool = True
    ) -> None:
        """Throughput, latency and speed-accuracy trade-off."""

        df_plot = df.sort_values("Throughput (images/s)")

        has_single = "Single-Image Latency Mean (ms)" in df.columns

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(21, 0.45 * len(df) + 4)
        )

        # 1. Throughput
        bars = axes[0].barh(
            df_plot["Experiment"],
            df_plot["Throughput (images/s)"],
            color="tab:blue"
        )

        axes[0].set_title(
            f"Throughput (images/s) - batch {batch_size}"
        )
        axes[0].set_xlabel("Images per second")

        for bar, v in zip(
            bars,
            df_plot["Throughput (images/s)"]
        ):
            axes[0].text(
                v,
                bar.get_y() + bar.get_height() / 2,
                f" {v:,.0f}",
                va="center",
                fontsize=9
            )

        axes[0].grid(axis="x", alpha=0.25)

        # 2. Single-image latency (or batch time if not measured)
        if has_single:
            lat_col = "Single-Image Latency Mean (ms)"
            lat_title = "Single-image inference latency (ms)"
        else:
            lat_col = f"Batch {batch_size} (ms)"
            lat_title = f"Time per batch of {batch_size} (ms)"

        bars = axes[1].barh(
            df_plot["Experiment"],
            df_plot[lat_col],
            color="tab:orange"
        )

        axes[1].set_title(lat_title)
        axes[1].set_xlabel("Milliseconds")

        for bar, v in zip(bars, df_plot[lat_col]):
            axes[1].text(
                v,
                bar.get_y() + bar.get_height() / 2,
                f" {v:.3f}",
                va="center",
                fontsize=9
            )

        axes[1].set_yticklabels([])
        axes[1].grid(axis="x", alpha=0.25)

        # 3. Trade-off: throughput vs accuracy
        axes[2].scatter(
            df["Throughput (images/s)"],
            df["Evaluation Acc (%)"],
            s=60,
            color="tab:green"
        )

        for _, row in df.iterrows():
            axes[2].annotate(
                row["Experiment"],
                (
                    row["Throughput (images/s)"],
                    row["Evaluation Acc (%)"]
                ),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8
            )

        axes[2].set_xlabel("Throughput (images/s)")
        axes[2].set_ylabel("Evaluation Accuracy (%)")
        axes[2].set_title("Speed - accuracy trade-off")
        axes[2].grid(alpha=0.25)

        fig.suptitle(
            f"Inference benchmark ({DEVICE})",
            fontsize=14,
            fontweight="bold"
        )

        fig.tight_layout()

        self._finish_figure(
            fig,
            "inference_benchmark",
            FIG_BENCHMARK,
            "inference_benchmark",
            save,
            show
        )

    # ============================================================
    # LOW-LEVEL FILE HELPERS
    # ============================================================

    @staticmethod
    def _ext(path: str) -> str:
        """Lower-case file extension without the dot."""

        return os.path.splitext(path)[1].lstrip(".").lower()

    @staticmethod
    def _write_text(path: str, text: str) -> str:
        """Write UTF-8 text, creating parent directories."""

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

        return path

    def _write_lines(
        self,
        path: str,
        title: str,
        lines: Sequence[str]
    ) -> str:
        """Write console-style lines as Markdown (.md) or plain text."""

        if self._ext(path) == "md":
            text = _lines_to_markdown(title, lines)
        else:
            text = title + "\n" + "=" * len(title) + "\n\n" + "\n".join(lines) + "\n"

        self._write_text(path, text)

        print(f"Report saved: {path}")

        return path

    def _resolve_dir(
        self,
        output_dir: Optional[str],
        *parts: str
    ) -> str:
        """Join ``output_dir`` (or the default) with sub-parts."""

        return os.path.join(output_dir or self.output_dir, *parts)

    # ============================================================
    # TABLE EXPORTING
    # ============================================================

    @staticmethod
    def _write_excel(
        sheets: Dict[str, pd.DataFrame],
        path: str
    ) -> str:
        """Write DataFrames to an .xlsx file (requires openpyxl).

        Header row is bold and frozen, column widths are fitted and float
        cells use a 4-decimal display format (stored values keep full
        precision).
        """

        from openpyxl.styles import Font
        from openpyxl.utils import get_column_letter

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        used: set = set()

        with pd.ExcelWriter(path, engine="openpyxl") as writer:

            for raw_name, df in sheets.items():

                name = _safe_sheet_name(raw_name, used)

                df.to_excel(writer, sheet_name=name, index=False)

                ws = writer.sheets[name]
                ws.freeze_panes = "A2"

                for idx, col in enumerate(df.columns, start=1):

                    values = [str(v) for v in df[col].head(200)]
                    width = max([len(str(col))] + [len(v) for v in values]) + 2

                    ws.column_dimensions[get_column_letter(idx)].width = min(width, 60)
                    ws.cell(row=1, column=idx).font = Font(bold=True)

                    if pd.api.types.is_float_dtype(df[col]):
                        for cells in ws.iter_rows(
                            min_row=2, min_col=idx, max_col=idx
                        ):
                            cells[0].number_format = "0.0000"

        return path

    def _export_dataframe(
        self,
        df: pd.DataFrame,
        path_stem: str,
        formats: Iterable[str] = ("csv", "xlsx"),
        sheet_name: str = "Sheet1"
    ) -> List[str]:
        """
        Export one DataFrame as CSV and/or Excel.

        ``path_stem`` is the file path without extension. Parent
        directories are created. Column names and full numeric precision
        are preserved. If Excel export is impossible (e.g. openpyxl is not
        installed) a warning is printed and the other formats still work.
        """

        saved: List[str] = []

        os.makedirs(os.path.dirname(os.path.abspath(path_stem)), exist_ok=True)

        for fmt in formats:

            fmt = fmt.lower().lstrip(".")

            if fmt == "csv":
                path = f"{path_stem}.csv"
                df.to_csv(path, index=False, encoding="utf-8-sig")

            elif fmt in ("xlsx", "excel"):
                path = f"{path_stem}.xlsx"

                try:
                    self._write_excel({sheet_name: df}, path)
                except ImportError:
                    print(
                        "Excel export skipped: install openpyxl "
                        "(pip install openpyxl)."
                    )
                    continue

            else:
                raise ValueError(f"Unsupported table format: '{fmt}'")

            saved.append(path)
            print(f"Table saved: {path}")

        return saved

    def save_summary(
        self,
        output_dir: Optional[str] = None,
        formats: Iterable[str] = ("csv", "xlsx"),
        filename: str = "summary"
    ) -> List[str]:
        """Export the full table returned by ``summary()``.

        Contains BEST-checkpoint columns ("Best ...") and LAST-epoch
        columns ("Last ...") side by side.
        """

        df = self.summary()

        if df.empty:
            print("No experiments to export.")
            return []

        return self._export_dataframe(
            df,
            os.path.join(self._resolve_dir(output_dir, "tables"), filename),
            formats,
            sheet_name="Summary"
        )

    def save_sweep_tables(
        self,
        output_dir: Optional[str] = None,
        formats: Iterable[str] = ("csv", "xlsx"),
        filename: str = "sweep_tables"
    ) -> List[str]:
        """Export every available sweep table.

        Excel: one workbook (``sweep_tables.xlsx``) with one sheet per
        sweep parameter. CSV: one file per sweep (``sweep_<param>.csv``).
        """

        params = self.available_sweeps()

        if not params:
            print("No parameter sweep available to export.")
            return []

        directory = self._resolve_dir(output_dir, "tables")
        tables = {p: self.sweep_table(p) for p in params}

        saved: List[str] = []

        for fmt in formats:

            fmt = fmt.lower().lstrip(".")

            if fmt == "csv":
                for param, df in tables.items():
                    saved += self._export_dataframe(
                        df,
                        os.path.join(directory, f"sweep_{slugify(param)}"),
                        ("csv",)
                    )

            elif fmt in ("xlsx", "excel"):
                path = os.path.join(directory, f"{filename}.xlsx")

                try:
                    self._write_excel(tables, path)
                    saved.append(path)
                    print(f"Table saved: {path}")
                except ImportError:
                    print("Excel export skipped: pip install openpyxl")

            else:
                raise ValueError(f"Unsupported table format: '{fmt}'")

        return saved

    def save_benchmark_results(
        self,
        df: Optional[pd.DataFrame] = None,
        output_dir: Optional[str] = None,
        formats: Iterable[str] = ("csv", "xlsx"),
        filename: str = "inference_benchmark"
    ) -> List[str]:
        """Export inference benchmark results.

        Uses ``df`` if given, otherwise the last ``benchmark_inference()``
        result. Returns an empty list when no benchmark was performed.
        """

        df = df if df is not None else self.benchmark_df

        if df is None or df.empty:
            print(f"Inference benchmark: {NOT_PERFORMED}.")
            return []

        return self._export_dataframe(
            df,
            os.path.join(self._resolve_dir(output_dir, "tables"), filename),
            formats,
            sheet_name="Benchmark"
        )

    def detailed_metrics_table(self) -> pd.DataFrame:
        """Structured one-row-per-experiment metrics table.

        BEST columns come from the BEST checkpoint (re-evaluated
        predictions). LAST columns come from the final-epoch training
        history only; LAST precision / recall / F1 / per-class accuracy are
        not computed because no LAST-checkpoint predictions are stored.
        """

        rows = []
        class_names = list(self.evaluator.class_names)

        for name, r in self.results.items():

            h = r.history
            last_idx = len(h.get("evaluation_loss", [])) - 1

            row: Dict[str, Any] = {
                "Experiment": name,
                "Group": r.config.group,
                "Architecture": r.config.architecture,
                "Parameters": r.n_params,
                "Training Time (s)": r.train_seconds,
                "BEST Epoch": r.best_epoch,
                "BEST Evaluation Loss": r.best_loss,
                "BEST Train Acc (%)": _hist_at(
                    h, "train_accuracy", r.best_epoch - 1
                ),
            }

            for key, value in r.metrics.items():
                row[f"BEST {key}"] = value

            row.update({
                "LAST Epoch": last_idx + 1,
                "LAST Evaluation Loss": _hist_at(h, "evaluation_loss", last_idx),
                "LAST Train Acc (%)": _hist_at(h, "train_accuracy", last_idx),
                "LAST Evaluation Acc (%)": _hist_at(h, "evaluation_accuracy", last_idx),
            })

            try:
                per_class = self.evaluator.per_class_accuracy(r.y_true, r.y_pred)
                for cname, value in zip(class_names, per_class):
                    row[f"BEST Class Acc: {cname} (%)"] = float(value)
            except Exception:
                pass

            row["BEST Checkpoint"] = r.model_path
            row["LAST Checkpoint"] = r.last_model_path

            rows.append(row)

        return pd.DataFrame(rows)

    def save_detailed_metrics(
        self,
        output_dir: Optional[str] = None,
        formats: Iterable[str] = ("csv", "xlsx"),
        filename: str = "detailed_metrics"
    ) -> List[str]:
        """Export per-experiment metrics (BEST and LAST information)."""

        df = self.detailed_metrics_table()

        if df.empty:
            print("No experiments to export.")
            return []

        return self._export_dataframe(
            df,
            os.path.join(self._resolve_dir(output_dir, "tables"), filename),
            formats,
            sheet_name="Detailed Metrics"
        )

    # ============================================================
    # TEXT / MARKDOWN EXPORTING
    # ============================================================

    def save_conclusions(
        self,
        output_dir: Optional[str] = None,
        fmt: str = "md",
        filename: str = "conclusions"
    ) -> str:
        """Save the automatic experiment conclusions (``md`` or ``txt``)."""

        path = os.path.join(
            self._resolve_dir(output_dir, "reports"), f"{filename}.{fmt}"
        )

        return self._write_lines(
            path, "Experiment Conclusions", self.build_conclusions()
        )

    def save_sweep_conclusions(
        self,
        output_dir: Optional[str] = None,
        fmt: str = "md",
        filename: str = "sweep_conclusions"
    ) -> Optional[str]:
        """Save the conclusions of every available sweep in one file."""

        params = self.available_sweeps()

        if not params:
            print("No parameter sweep available.")
            return None

        lines: List[str] = []

        for param in params:
            lines.extend(self.build_sweep_conclusion(param))
            lines.append("")

        path = os.path.join(
            self._resolve_dir(output_dir, "reports"), f"{filename}.{fmt}"
        )

        return self._write_lines(path, "Parameter Sweep Conclusions", lines)

    def _format_verification(
        self,
        results: Sequence[Dict[str, Any]],
        fmt: str = "txt"
    ) -> str:
        """Render verification results as text or Markdown."""

        if not results:
            return f"Checkpoint verification: {NOT_PERFORMED}.\n"

        if fmt == "md":

            lines = [
                "# Checkpoint Verification", "",
                "| Experiment | Checkpoint | Reloaded Acc (%) | "
                "Expected Acc (%) | Difference (pp) | Status |",
                "|---|---|---|---|---|---|",
            ]

            for res in results:
                lines.append(
                    f"| {res['experiment']} | {res['checkpoint_type']} | "
                    f"{res['reloaded_accuracy']:.4f} | "
                    f"{res['expected_accuracy']:.4f} | "
                    f"{res['difference']:.8f} | {res['status']} |"
                )

            return "\n".join(lines) + "\n"

        lines = ["CHECKPOINT VERIFICATION", "=" * 60]

        for res in results:
            lines += [
                f"Experiment         : {res['experiment']}",
                f"Checkpoint type    : {res['checkpoint_type']}",
                f"Path               : {res['path']}",
                f"Reloaded Accuracy  : {res['reloaded_accuracy']:.4f}%",
                f"Expected Accuracy  : {res['expected_accuracy']:.4f}%",
                f"Difference         : {res['difference']:.8f} percentage points",
                f"Status             : {res['status']}",
                "-" * 60,
            ]

        return "\n".join(lines) + "\n"

    def save_verification_report(
        self,
        output_dir: Optional[str] = None,
        fmt: str = "txt",
        filename: str = "verification"
    ) -> str:
        """Save all checkpoint-verification results collected so far.

        Run ``verify_saved_model()`` (or ``create_report(
        run_verification=True)``) first; otherwise the file states that
        verification was not performed.
        """

        path = os.path.join(
            self._resolve_dir(output_dir, "reports"), f"{filename}.{fmt}"
        )

        self._write_text(
            path,
            self._format_verification(list(self._verification.values()), fmt)
        )

        print(f"Report saved: {path}")

        return path

    def build_detail_report(
        self,
        name: str,
        fmt: str = "md",
        relative_to: Optional[str] = None
    ) -> str:
        """Build the detailed analysis of one experiment as text.

        Includes configuration, BEST and LAST checkpoint information,
        BEST evaluation metrics, the classification report (BEST
        predictions) and paths to related figures (if generated).
        """

        r = self.results[name]
        h = r.history
        md = fmt == "md"
        last_idx = len(h.get("evaluation_loss", [])) - 1

        def heading(text: str) -> List[str]:
            return [f"## {text}", ""] if md else ["", text, "-" * len(text)]

        def item(key: str, value: Any) -> str:
            return f"- **{key}**: {value}" if md else f"{key:<28}: {value}"

        def fmt_val(v: Optional[float], spec: str) -> str:
            return NOT_AVAILABLE if v is None else format(v, spec)

        lines: List[str] = [
            f"# Detailed Analysis: {name}" if md
            else f"DETAILED ANALYSIS: {name}"
        ]

        lines += heading("Configuration")

        flat: Dict[str, Any] = {}
        _flatten("", _config_to_dict(r.config), flat)
        lines += [item(k, v) for k, v in flat.items()]

        lines += heading("BEST checkpoint (lowest evaluation loss)")
        lines += [
            item("Path", r.model_path),
            item("File exists", os.path.isfile(r.model_path)),
            item("Best epoch", r.best_epoch),
            item("Best evaluation loss", f"{r.best_loss:.6f}"),
            item("Train accuracy at best epoch (%)",
                 fmt_val(_hist_at(h, "train_accuracy", r.best_epoch - 1), ".2f")),
            item("Evaluation accuracy (%)", f"{r.metrics['Accuracy (%)']:.2f}"),
        ]

        lines += heading("LAST checkpoint (final training epoch)")
        lines += [
            item("Path", r.last_model_path),
            item("File exists", os.path.isfile(r.last_model_path)),
            item("Last epoch", last_idx + 1),
            item("Last evaluation loss",
                 fmt_val(_hist_at(h, "evaluation_loss", last_idx), ".6f")),
            item("Train accuracy (%)",
                 fmt_val(_hist_at(h, "train_accuracy", last_idx), ".2f")),
            item("Evaluation accuracy (%)",
                 fmt_val(_hist_at(h, "evaluation_accuracy", last_idx), ".2f")),
        ]

        lines += heading("Evaluation metrics (BEST checkpoint)")
        lines += [item(k, f"{v:.4f}") for k, v in r.metrics.items()]
        lines += [
            item("Number of parameters", f"{r.n_params:,}"),
            item("Training time (s)", f"{r.train_seconds:.2f}"),
        ]

        lines += heading("Classification report (BEST checkpoint)")

        report = self._classification_report_text(r)

        if md:
            lines += ["```text", report.rstrip(), "```"]
        else:
            lines += [report.rstrip()]

        lines += heading("Related figures")

        for label, suffix in (
            ("Learning curves", "learning_curves"),
            ("Confusion matrix", "confusion_matrix"),
            ("Sample predictions", "sample_predictions"),
        ):
            path = self._figure_path(f"detail/{name}/{suffix}")

            if path and relative_to:
                path = os.path.relpath(path, relative_to)

            lines.append(item(label, path or NOT_AVAILABLE))

        return "\n".join(lines).rstrip() + "\n"

    def save_detail_report(
        self,
        name: str,
        output_dir: Optional[str] = None,
        fmt: str = "md"
    ) -> str:
        """Save the detailed analysis of one experiment to a file.

        Default location: ``<output_dir>/reports/detailed_reports/``.
        """

        directory = (
            output_dir
            or self._resolve_dir(None, "reports", "detailed_reports")
        )

        path = os.path.join(directory, f"{slugify(name)}.{fmt}")

        self._write_text(
            path,
            self.build_detail_report(name, fmt, relative_to=directory)
        )

        print(f"Report saved: {path}")

        return path

    # ============================================================
    # STRUCTURED DATA EXPORT (JSON)
    # ============================================================

    def _result_to_dict(
        self,
        name: str,
        include_predictions: bool = False
    ) -> Dict[str, Any]:
        """Serialize one ExperimentResult (without large arrays)."""

        r = self.results[name]
        h = r.history
        last_idx = len(h.get("evaluation_loss", [])) - 1

        data: Dict[str, Any] = {
            "name": name,
            "config": _config_to_dict(r.config),
            "n_params": int(r.n_params),
            "train_seconds": float(r.train_seconds),
            "history": to_jsonable(h),
            "metrics_best_checkpoint": to_jsonable(r.metrics),
            "checkpoints": {
                "best": {
                    "path": r.model_path,
                    "exists": os.path.isfile(r.model_path),
                    "epoch": int(r.best_epoch),
                    "evaluation_loss": float(r.best_loss),
                    "evaluation_accuracy_pct": float(r.metrics["Accuracy (%)"]),
                    "train_accuracy_pct": _hist_at(
                        h, "train_accuracy", r.best_epoch - 1
                    ),
                },
                "last": {
                    "path": r.last_model_path,
                    "exists": os.path.isfile(r.last_model_path),
                    "epoch": last_idx + 1,
                    "evaluation_loss": _hist_at(h, "evaluation_loss", last_idx),
                    "evaluation_accuracy_pct": _hist_at(
                        h, "evaluation_accuracy", last_idx
                    ),
                    "train_accuracy_pct": _hist_at(h, "train_accuracy", last_idx),
                },
            },
        }

        # Per-class accuracy (BEST predictions)
        try:
            per_class = self.evaluator.per_class_accuracy(r.y_true, r.y_pred)
            data["per_class_accuracy_best"] = {
                cname: float(v)
                for cname, v in zip(self.evaluator.class_names, per_class)
            }
        except Exception:
            data["per_class_accuracy_best"] = None

        # Prediction statistics (BEST predictions) instead of full arrays
        try:
            y_true = np.asarray(r.y_true)
            y_pred = np.asarray(r.y_pred)
            correct = y_true == y_pred

            stats: Dict[str, Any] = {
                "n_samples": int(y_true.size),
                "n_correct": int(correct.sum()),
                "accuracy_pct": float(correct.mean() * 100),
                "true_class_counts": np.bincount(y_true, minlength=10).tolist(),
                "predicted_class_counts": np.bincount(y_pred, minlength=10).tolist(),
            }

            y_prob = np.asarray(r.y_prob)

            if y_prob.ndim == 2 and y_prob.shape[0] == y_true.size:
                confidence = y_prob.max(axis=1)
                stats["mean_confidence"] = float(confidence.mean())
                stats["mean_confidence_correct"] = (
                    float(confidence[correct].mean()) if correct.any() else None
                )
                stats["mean_confidence_incorrect"] = (
                    float(confidence[~correct].mean()) if (~correct).any() else None
                )

            data["prediction_stats_best"] = stats

            if include_predictions:
                data["y_true"] = y_true.tolist()
                data["y_pred"] = y_pred.tolist()

        except Exception:
            data["prediction_stats_best"] = None

        return to_jsonable(data)

    def export_results_json(
        self,
        path: Optional[str] = None,
        include_predictions: bool = False
    ) -> str:
        """
        Export all experiment data to a reloadable JSON file.

        Contains configuration, histories, BEST / LAST checkpoint
        metadata, metrics, parameter counts, training times, per-class
        accuracy and prediction statistics. Full ``y_true`` / ``y_pred``
        arrays are only included when ``include_predictions=True``;
        probability matrices are never exported.

        Reload with :meth:`load_results_json`.
        """

        path = path or os.path.join(
            self._resolve_dir(None, "data"), "experiment_results.json"
        )

        payload = {
            "schema_version": JSON_SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": DATASET_NAME,
            "device": str(DEVICE),
            "baseline": self.baseline_name,
            "class_names": to_jsonable(list(self.evaluator.class_names)),
            "experiments": {
                name: self._result_to_dict(name, include_predictions)
                for name in self.results
            },
            "inference_benchmark": (
                {
                    "meta": to_jsonable(self.benchmark_meta),
                    "rows": to_jsonable(self.benchmark_df),
                }
                if self.benchmark_df is not None else None
            ),
            "verification": to_jsonable(list(self._verification.values())),
        }

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)

        print(f"Data saved: {path}")

        return path

    @staticmethod
    def load_results_json(path: str) -> Dict[str, Any]:
        """Load a file written by :meth:`export_results_json`."""

        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    # ============================================================
    # REPORT HELPERS (analysis tables / facts)
    # ============================================================

    def config_table(self) -> pd.DataFrame:
        """One row per experiment with all scalar config values."""

        rows = []

        for name, r in self.results.items():
            flat: Dict[str, Any] = {}
            _flatten("", _config_to_dict(r.config), flat)
            flat["name"] = name
            rows.append(flat)

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        preferred = [
            "name", "group", "architecture", "optimizer_text", "optimizer",
            "learning_rate", "lr", "batch_size", "epochs", "weight_decay",
            "seed",
        ]

        ordered = (
            [c for c in preferred if c in df.columns]
            + [c for c in df.columns if c not in preferred]
        )

        df = df[ordered]

        return df.rename(columns={
            c: ("Experiment" if c == "name" else c.replace("_", " ").title())
            for c in df.columns
        })

    def per_class_table(self) -> pd.DataFrame:
        """Per-class accuracy statistics across experiments (BEST)."""

        if not self.results:
            return pd.DataFrame()

        names, matrix = self._per_class_matrix()

        rows = []

        for i, cname in enumerate(self.evaluator.class_names):
            col = matrix[:, i]
            rows.append({
                "Class": cname,
                "Mean Accuracy (%)": float(col.mean()),
                "Std (pp)": float(col.std()),
                "Min (%)": float(col.min()),
                "Max (%)": float(col.max()),
                "Best Experiment": names[int(col.argmax())],
                "Worst Experiment": names[int(col.argmin())],
            })

        return pd.DataFrame(rows).sort_values(
            "Mean Accuracy (%)"
        ).reset_index(drop=True)

    def _executive_facts(self) -> Dict[str, Any]:
        """Key facts for the executive summary (computed from results)."""

        df = self.summary()

        if df.empty:
            return {}

        def pick(col: str, largest: bool = True) -> pd.Series:
            idx = df[col].idxmax() if largest else df[col].idxmin()
            return df.loc[idx]

        best_acc = pick("Best Evaluation Acc (%)")
        low_loss = pick("Best Evaluation Loss", largest=False)
        best_f1 = pick("Macro F1-score")
        slowest = pick("Training Time (s)")
        gap = pick("Best Train-Evaluation Gap")

        facts: Dict[str, Any] = {
            "best_accuracy": best_acc,
            "lowest_loss": low_loss,
            "best_f1": best_f1,
            "slowest": slowest,
            "highest_gap": gap,
            "mean_time": float(df["Training Time (s)"].mean()),
            "max_time": float(df["Training Time (s)"].max()),
        }

        pc = self.per_class_table()

        if not pc.empty:
            facts["hardest_class"] = pc.iloc[0]
            facts["easiest_class"] = pc.iloc[-1]

        return facts

    # ============================================================
    # create_report()
    # ============================================================

    def _safe(
        self,
        ctx: ReportContext,
        label: str,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Run a step; on failure record a note and keep going."""

        before = set(plt.get_fignums())

        try:
            return fn(*args, **kwargs)

        except Exception as exc:  # noqa: BLE001 - report must keep going
            ctx.notes.append(f"{label}: skipped ({type(exc).__name__}: {exc})")
            print(f"[report] {label}: skipped ({type(exc).__name__}: {exc})")

            for number in plt.get_fignums():
                if number not in before:
                    plt.close(number)

            return None

    def _generate_report_figures(self, ctx: ReportContext) -> None:
        """Generate every figure used by the PDF (never displayed)."""

        for group in self.groups():
            self._safe(ctx, f"Group curves '{group}'",
                       self.plot_group_curves, group, save=True, show=False)

        self._safe(ctx, "Final comparison figure",
                   self.plot_final_comparison, save=True, show=False)

        self._safe(ctx, "Per-class heatmap",
                   self.plot_per_class_accuracy, save=True, show=False)

        if ctx.include_sweeps:
            for param in self.available_sweeps():
                self._safe(ctx, f"Sweep figure '{param}'",
                           self.plot_sweep, param, save=True, show=False)

        if ctx.include_detailed_reports:
            for name in self.results:
                self._safe(ctx, f"Detail figures '{name}'",
                           self._render_detail_figures,
                           name, save=True, show=False)

        if ctx.include_benchmark and self.benchmark_df is not None:
            bs = int(self.benchmark_meta.get("batch_size", 256))
            self._safe(ctx, "Benchmark figure",
                       self._plot_inference_benchmark,
                       self.benchmark_df, bs, save=True, show=False)

    def _export_report_files(self, ctx: ReportContext) -> None:
        """Write tables, text reports and JSON next to the PDF."""

        out = ctx.output_dir
        gen = ctx.generated_files

        def record(kind: str, result: Any) -> None:
            if not result:
                return
            gen.setdefault(kind, []).extend(
                result if isinstance(result, list) else [result]
            )

        record("Tables", self._safe(ctx, "Summary table", self.save_summary, out))
        record("Tables", self._safe(ctx, "Detailed metrics table",
                                    self.save_detailed_metrics, out))

        if ctx.include_sweeps:
            record("Tables", self._safe(ctx, "Sweep tables",
                                        self.save_sweep_tables, out))

        if ctx.include_benchmark:
            record("Tables", self._safe(ctx, "Benchmark table",
                                        self.save_benchmark_results,
                                        None, out))

        record("Reports", self._safe(ctx, "Conclusions", self.save_conclusions, out))

        if ctx.include_sweeps:
            record("Reports", self._safe(ctx, "Sweep conclusions",
                                         self.save_sweep_conclusions, out))

        record("Reports", self._safe(ctx, "Verification report",
                                     self.save_verification_report, out))

        if ctx.include_detailed_reports:
            detail_dir = os.path.join(out, "reports", "detailed_reports")
            for name in self.results:
                record("Reports", self._safe(
                    ctx, f"Detail report '{name}'",
                    self.save_detail_report, name, detail_dir))

        record("Data", self._safe(
            ctx, "JSON export", self.export_results_json,
            os.path.join(out, "data", "experiment_results.json")))

    # ------------------------------------------------------------
    # PDF sections
    # ------------------------------------------------------------

    @staticmethod
    def _device_description() -> str:
        """Human-readable description of the compute device."""

        text = str(DEVICE)

        if text.startswith("cuda"):
            try:
                text += f" ({torch.cuda.get_device_name(0)})"
            except Exception:
                pass
        else:
            text += " (CPU)"

        return text

    def _pdf_cover(self, b: Any, ctx: ReportContext) -> None:

        b.add_cover(
            subtitle="Training, evaluation and comparison of MLP models",
            info_rows=[
                ("Dataset", DATASET_NAME),
                ("Number of experiments", str(len(self.results))),
                ("Baseline experiment",
                 self.baseline_name + (
                     "" if self.baseline_name in self.results
                     else " (not found among executed experiments)"
                 )),
                ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                ("Device", self._device_description()),
                ("Platform", platform.platform()),
                ("PyTorch", torch.__version__),
            ],
        )

    def _pdf_executive_summary(self, b: Any, ctx: ReportContext) -> None:

        b.begin_section("1. Executive Summary")

        facts = self._executive_facts()

        if not facts:
            b.add_paragraph(f"No experiments have been executed. {NOT_AVAILABLE}.")
            return

        a, l, f1 = facts["best_accuracy"], facts["lowest_loss"], facts["best_f1"]
        s, g = facts["slowest"], facts["highest_gap"]

        rows = [
            ("Best accuracy (BEST checkpoint)",
             f"{a['Experiment']} - {a['Best Evaluation Acc (%)']:.2f}%"),
            ("Lowest evaluation loss (BEST)",
             f"{l['Experiment']} - {l['Best Evaluation Loss']:.6f} "
             f"(epoch {int(l['Best Epoch'])})"),
            ("Best Macro F1-score (BEST)",
             f"{f1['Experiment']} - {f1['Macro F1-score']:.4f}"),
            ("Training time (mean / max)",
             f"{facts['mean_time']:.1f}s / {facts['max_time']:.1f}s "
             f"(slowest: {s['Experiment']})"),
            ("Highest generalization gap (BEST epoch)",
             f"{g['Experiment']} - {g['Best Train-Evaluation Gap']:.2f} points"),
        ]

        if "hardest_class" in facts:
            hc, ec = facts["hardest_class"], facts["easiest_class"]
            rows += [
                ("Most difficult class",
                 f"{hc['Class']} - mean accuracy {hc['Mean Accuracy (%)']:.1f}%"),
                ("Easiest class",
                 f"{ec['Class']} - mean accuracy {ec['Mean Accuracy (%)']:.1f}%"),
            ]

        b.add_key_value_table(rows)

        b.add_note(
            "Unless stated otherwise, all accuracy, loss, precision, recall, "
            "F1 and per-class values refer to the BEST checkpoint (lowest "
            "evaluation loss). LAST values come from the final training "
            "epoch history."
        )

    def _pdf_configuration(self, b: Any, ctx: ReportContext) -> None:

        b.begin_section("2. Experimental Configuration", landscape_page=True)

        df = self.config_table()

        if df.empty:
            b.add_paragraph(f"No experiments have been executed. {NOT_AVAILABLE}.")
            return

        varying = [
            c for c in df.columns
            if c == "Experiment" or df[c].astype(str).nunique() > 1
        ]

        shared = [
            (c, str(df[c].iloc[0]))
            for c in df.columns
            if c not in varying
        ]

        if shared:
            b.add_subsection("Settings shared by all experiments")
            b.add_key_value_table(shared)

        b.add_subsection("Settings that differ between experiments")

        if len(varying) > 1:
            b.add_table(df[varying], caption="Experiment configurations (varying settings).")
        else:
            b.add_paragraph("All experiments share identical settings.")

    def _pdf_comparison(self, b: Any, ctx: ReportContext) -> None:

        b.begin_section("3. Overall Experiment Comparison", landscape_page=True)

        df = self.summary()

        if df.empty:
            b.add_paragraph(f"No experiments have been executed. {NOT_AVAILABLE}.")
            return

        b.add_paragraph(
            "BEST columns describe the checkpoint with the lowest evaluation "
            "loss; LAST columns describe the final training epoch. Together "
            "the three tables below contain the complete summary."
        )

        b.add_table(
            df[["Experiment", "Group", "Architecture", "Parameters",
                "Best Epoch", "Best Evaluation Loss", "Best Train Acc (%)",
                "Best Evaluation Acc (%)", "Best Train-Evaluation Gap"]],
            caption="BEST checkpoint: loss, accuracy and generalization gap.",
        )

        b.add_table(
            df[["Experiment", "Macro Precision", "Macro Recall",
                "Macro F1-score", "Parameters", "Training Time (s)"]],
            caption="BEST checkpoint: macro metrics, parameter counts and training time.",
        )

        b.add_table(
            df[["Experiment", "Last Epoch", "Last Evaluation Loss",
                "Last Train Acc (%)", "Last Evaluation Acc (%)",
                "Last Train-Evaluation Gap"]],
            caption="LAST checkpoint (final epoch) metrics from the training history.",
        )

        if ctx.include_figures:
            b.add_figure(
                self._figure_path("final_comparison"),
                "BEST checkpoint evaluation accuracy (left) and "
                "train-evaluation generalization gap (right).",
            )

    def _pdf_curves(self, b: Any, ctx: ReportContext) -> None:

        b.begin_section("4. Training History and Learning Curves", landscape_page=True)

        if not self.results:
            b.add_paragraph(f"No experiments have been executed. {NOT_AVAILABLE}.")
            return

        if not ctx.include_figures:
            b.add_paragraph("Figures were not included (include_figures=False).")
            return

        b.add_paragraph(
            "Each figure compares experiments of one group: evaluation "
            "accuracy, evaluation loss and training accuracy per epoch. The "
            "baseline is drawn as a dashed black line."
        )

        for group in self.groups():
            b.add_subsection(f"Group: {group}")
            b.add_figure(
                self._figure_path(f"group_curves/{group}"),
                f"Learning curves for group '{group}'.",
            )

    def _pdf_sweeps(self, b: Any, ctx: ReportContext) -> None:

        b.begin_section("5. Parameter Sweep Analysis", landscape_page=True)

        if not ctx.include_sweeps:
            b.add_paragraph("Sweeps were not included (include_sweeps=False).")
            return

        params = self.available_sweeps()

        if not params:
            b.add_paragraph(
                "No parameter sweep is available (a sweep needs the baseline "
                "and at least two experiments that differ only in one "
                f"parameter). {NOT_AVAILABLE}."
            )
            return

        for param in params:

            spec = SWEEP_PARAMS[param]
            label = spec["label"]

            b.add_subsection(f"Sweep: {label}", toc=True)

            values = [
                f"{spec['value'](self.results[n].config):g}"
                for n in self.sweep_members(param)
            ]
            base_value = spec["value"](self.results[self.baseline_name].config)

            b.add_key_value_table([
                ("Swept parameter", f"{label} ({param})"),
                ("Values", ", ".join(values)),
                ("Baseline value", f"{base_value:g}"),
                ("Scale", "logarithmic" if spec.get("log") else "linear"),
            ])

            b.add_table(
                self.sweep_table(param),
                caption=f"Sweep results for {label} (BEST checkpoint; "
                        f"'Last' columns from the final epoch).",
            )

            if ctx.include_figures:
                b.add_figure(
                    self._figure_path(f"sweep/{param}"),
                    f"Sweep '{label}': best accuracy, generalization gap "
                    f"and evaluation accuracy curves.",
                )

            b.add_subsubsection("Best and worst configurations vs baseline")
            b.add_bullets([
                line.strip()[2:] if line.startswith("  - ") else line
                for line in self.build_sweep_conclusion(param)
            ])

    def _pdf_per_class(self, b: Any, ctx: ReportContext) -> None:

        b.begin_section("6. Per-Class Performance Analysis", landscape_page=True)

        pc = self.per_class_table()

        if pc.empty:
            b.add_paragraph(f"No experiments have been executed. {NOT_AVAILABLE}.")
            return

        if ctx.include_figures:
            b.add_figure(
                self._figure_path("per_class_accuracy"),
                "Per-class accuracy (%) of every experiment (BEST checkpoint).",
            )

        b.add_table(
            pc,
            caption="Per-class accuracy statistics across experiments "
                    "(BEST checkpoint), sorted from hardest to easiest class.",
            id_cols=1,
        )

        hardest = pc.head(3)
        easiest = pc.tail(3).iloc[::-1]

        b.add_subsubsection("Observations")
        b.add_bullets([
            "Hardest classes: " + "; ".join(
                f"{r['Class']} ({r['Mean Accuracy (%)']:.1f}%)"
                for _, r in hardest.iterrows()
            ),
            "Easiest classes: " + "; ".join(
                f"{r['Class']} ({r['Mean Accuracy (%)']:.1f}%)"
                for _, r in easiest.iterrows()
            ),
            f"The class with the largest disagreement between experiments is "
            f"{pc.loc[pc['Std (pp)'].idxmax(), 'Class']} "
            f"(std {pc['Std (pp)'].max():.2f} pp).",
        ])

    def _pdf_detailed(self, b: Any, ctx: ReportContext) -> None:

        b.begin_section("7. Detailed Classification Analysis")

        if not self.results:
            b.add_paragraph(f"No experiments have been executed. {NOT_AVAILABLE}.")
            return

        if not ctx.include_detailed_reports:
            b.add_paragraph(
                "Detailed per-experiment analysis was not included "
                "(include_detailed_reports=False)."
            )
            return

        for name, r in self.results.items():

            b.add_page_break()
            b.add_subsection(name, toc=True)

            metrics_df = pd.DataFrame([{
                "Metric": k, "Value (BEST checkpoint)": v
            } for k, v in r.metrics.items()])

            b.add_table(metrics_df, caption=f"Evaluation metrics of '{name}' (BEST checkpoint).",
                        decimals=4)

            if ctx.include_classification_reports:
                b.add_subsubsection("Classification report (BEST checkpoint)")
                b.add_preformatted(self._classification_report_text(r))

            if ctx.include_figures:
                for suffix, text in (
                    ("learning_curves", "Learning curves"),
                    ("confusion_matrix", "Confusion matrix (BEST checkpoint)"),
                    ("sample_predictions", "Sample predictions (BEST checkpoint)"),
                ):
                    b.add_figure(
                        self._figure_path(f"detail/{name}/{suffix}"),
                        f"{text} - {name}.",
                        max_height_fraction=0.4,
                    )

    def _pdf_checkpoints(self, b: Any, ctx: ReportContext) -> None:

        b.begin_section("8. Checkpoint Information and Verification", landscape_page=True)

        if not self.results:
            b.add_paragraph(f"No experiments have been executed. {NOT_AVAILABLE}.")
            return

        rows = []

        for name, r in self.results.items():
            h = r.history
            last_idx = len(h.get("evaluation_loss", [])) - 1

            rows.append({
                "Experiment": name,
                "Type": "BEST",
                "Epoch": r.best_epoch,
                "Evaluation Loss": r.best_loss,
                "Path": r.model_path,
                "File Exists": os.path.isfile(r.model_path),
            })
            rows.append({
                "Experiment": name,
                "Type": "LAST",
                "Epoch": last_idx + 1,
                "Evaluation Loss": _hist_at(h, "evaluation_loss", last_idx),
                "Path": r.last_model_path,
                "File Exists": os.path.isfile(r.last_model_path),
            })

        b.add_table(
            pd.DataFrame(rows),
            caption="BEST and LAST checkpoints (BEST = lowest evaluation "
                    "loss, LAST = final epoch).",
            decimals=6,
        )

        b.add_subsection("Checkpoint verification")

        if not self._verification:
            b.add_paragraph(f"Checkpoint verification: {NOT_PERFORMED}.")
            return

        ver = pd.DataFrame([{
            "Experiment": v["experiment"],
            "Checkpoint": v["checkpoint_type"],
            "Reloaded Acc (%)": v["reloaded_accuracy"],
            "Expected Acc (%)": v["expected_accuracy"],
            "Difference (pp)": v["difference"],
            "Status": v["status"],
        } for v in self._verification.values()])

        b.add_table(
            ver,
            caption="Reloaded checkpoint accuracy versus expected accuracy "
                    "(BEST vs BEST metrics, LAST vs final-epoch history).",
            decimals=4,
        )

    def _pdf_benchmark(self, b: Any, ctx: ReportContext) -> None:

        b.begin_section("9. Inference Benchmark", landscape_page=True)

        if not ctx.include_benchmark:
            b.add_paragraph("Benchmark was not included (include_benchmark=False).")
            return

        if self.benchmark_df is None or self.benchmark_df.empty:
            b.add_paragraph(
                f"Inference benchmark: {NOT_PERFORMED}. Call "
                f"runner.benchmark_inference() before create_report()."
            )
            return

        meta = self.benchmark_meta

        rows = [
            ("Device", str(meta.get("device", NOT_AVAILABLE))),
            ("Checkpoint", str(meta.get("checkpoint_type", NOT_AVAILABLE))),
            ("Batch size", str(meta.get("batch_size", NOT_AVAILABLE))),
            ("Warm-up / timed runs",
             f"{meta.get('n_warmup', NOT_AVAILABLE)} / {meta.get('n_runs', NOT_AVAILABLE)}"),
            ("System RAM", f"{meta['system_ram_gb']} GB"
             if "system_ram_gb" in meta else NOT_AVAILABLE),
        ]

        if "gpu_name" in meta:
            rows.append((
                "GPU",
                f"{meta['gpu_name']} ({meta.get('gpu_total_vram_gb', '?')} GB VRAM)"
            ))

        if not meta.get("memory_measured", False):
            rows.append(("Memory measurement", NOT_PERFORMED))

        b.add_key_value_table(rows)

        b.add_table(
            self.benchmark_df,
            caption="Inference benchmark (throughput, batch time, "
                    "single-image latency and P95 latency).",
        )

        if ctx.include_figures:
            b.add_figure(
                self._figure_path("inference_benchmark"),
                "Throughput, latency and speed-accuracy trade-off.",
            )

    def _pdf_conclusions(self, b: Any, ctx: ReportContext) -> None:

        b.begin_section("10. Final Conclusions")

        if not self.results:
            b.add_paragraph(f"No experiments have been executed. {NOT_AVAILABLE}.")
            return

        df = self.summary()
        base_row = df[df["Experiment"] == self.baseline_name]

        b.add_subsection("Overall best experiment and baseline comparison")

        best = df.loc[df["Best Evaluation Acc (%)"].idxmax()]
        items = [
            f"Best experiment: {best['Experiment']} with "
            f"{best['Best Evaluation Acc (%)']:.2f}% accuracy and Macro F1 "
            f"{best['Macro F1-score']:.4f} (BEST checkpoint)."
        ]

        if not base_row.empty:
            base = base_row.iloc[0]
            items.append(
                f"Baseline '{self.baseline_name}': "
                f"{base['Best Evaluation Acc (%)']:.2f}% accuracy; the best "
                f"experiment differs by "
                f"{best['Best Evaluation Acc (%)'] - base['Best Evaluation Acc (%)']:+.2f} pp."
            )
            better = int((df["Best Evaluation Acc (%)"] > base["Best Evaluation Acc (%)"]).sum())
            items.append(
                f"{better} of {len(df) - 1} other experiments outperform the baseline."
            )
        else:
            items.append(f"Baseline '{self.baseline_name}' was not found: {NOT_AVAILABLE}.")

        b.add_bullets(items)

        b.add_subsection("Parameter sweep findings")

        if ctx.include_sweeps and self.available_sweeps():
            for param in self.available_sweeps():
                lines = self.build_sweep_conclusion(param)
                b.add_paragraph(lines[0])
                b.add_bullets([
                    ln.strip()[2:] for ln in lines[1:] if ln.startswith("  - ")
                ])
        else:
            b.add_paragraph(f"No parameter sweep analysis: {NOT_AVAILABLE}.")

        b.add_subsection("Generalization and overfitting")

        gaps = df.sort_values("Best Train-Evaluation Gap", ascending=False)

        b.add_bullets([
            f"Largest gap at the best epoch: {gaps.iloc[0]['Experiment']} "
            f"({gaps.iloc[0]['Best Train-Evaluation Gap']:.2f} points).",
            f"Smallest gap at the best epoch: {gaps.iloc[-1]['Experiment']} "
            f"({gaps.iloc[-1]['Best Train-Evaluation Gap']:.2f} points).",
            f"Mean gap across experiments: "
            f"{df['Best Train-Evaluation Gap'].mean():.2f} points "
            f"(BEST epoch); {df['Last Train-Evaluation Gap'].mean():.2f} points "
            f"(LAST epoch).",
        ])

        b.add_subsection("Performance trade-offs")

        fastest = df.loc[df["Training Time (s)"].idxmin()]
        smallest = df.loc[df["Parameters"].idxmin()]

        trade = [
            f"Fastest to train: {fastest['Experiment']} "
            f"({fastest['Training Time (s)']:.1f}s, "
            f"{fastest['Best Evaluation Acc (%)']:.2f}% accuracy).",
            f"Smallest model: {smallest['Experiment']} "
            f"({int(smallest['Parameters']):,} parameters, "
            f"{smallest['Best Evaluation Acc (%)']:.2f}% accuracy).",
        ]

        if self.benchmark_df is not None and not self.benchmark_df.empty and ctx.include_benchmark:
            top = self.benchmark_df.iloc[0]
            trade.append(
                f"Highest inference throughput: {top['Experiment']} "
                f"({top['Throughput (images/s)']:,.0f} images/s, "
                f"{top['Evaluation Acc (%)']:.2f}% accuracy)."
            )
        else:
            trade.append(f"Inference speed trade-off: {NOT_PERFORMED}.")

        b.add_bullets(trade)

        b.add_subsection("Automatically generated findings")

        for line in self.build_conclusions():
            if not line.strip():
                continue
            if line.startswith("  - "):
                b.add_bullets([line[4:]])
            else:
                b.add_paragraph(line)

    def _pdf_appendix(self, b: Any, ctx: ReportContext) -> None:

        b.begin_section("Appendix: Generated Files and Notes")

        b.add_subsection("Exported files")

        if ctx.generated_files:
            for kind, paths in ctx.generated_files.items():
                b.add_subsubsection(kind)
                b.add_bullets([
                    os.path.relpath(p, ctx.output_dir) for p in paths
                ])
        else:
            b.add_paragraph("No intermediate files were saved.")

        b.add_subsection("Notes")

        if ctx.notes:
            b.add_bullets(ctx.notes)
        else:
            b.add_paragraph("All requested report steps completed without errors.")

    def _build_pdf_report(self, ctx: ReportContext) -> str:
        """Assemble the PDF from the already generated figures / tables."""

        from .report_pdf import PdfReportBuilder

        builder = PdfReportBuilder(
            title="Experiment Report",
            author="ExperimentRunner",
        )

        steps: List[Tuple[str, Callable[[Any, ReportContext], None]]] = [
            ("Cover page", self._pdf_cover),
            ("Executive summary", self._pdf_executive_summary),
            ("Configuration section", self._pdf_configuration),
            ("Comparison section", self._pdf_comparison),
            ("Learning curves section", self._pdf_curves),
            ("Sweep section", self._pdf_sweeps),
            ("Per-class section", self._pdf_per_class),
            ("Detailed analysis section", self._pdf_detailed),
            ("Checkpoint section", self._pdf_checkpoints),
            ("Benchmark section", self._pdf_benchmark),
            ("Conclusions section", self._pdf_conclusions),
        ]

        for label, step in steps:
            self._safe(ctx, label, step, builder, ctx)

        # The appendix is last so it can list every note collected above.
        self._safe(ctx, "Appendix", self._pdf_appendix, builder, ctx)

        return builder.build(
            ctx.report_path,
            header_text=f"Experiment Report - {DATASET_NAME}",
        )

    def create_report(
        self,
        output_dir: str = "./experiment_outputs",
        report_name: str = "experiment_report",
        include_figures: bool = True,
        include_sweeps: bool = True,
        include_benchmark: bool = True,
        include_detailed_reports: bool = True,
        include_classification_reports: bool = True,
        save_intermediate_files: bool = True,
        run_verification: bool = True,
        run_benchmark_if_missing: bool = False
    ) -> str:
        """
        Collect all results and generate one complete PDF report.

        Output layout (created automatically)::

            <output_dir>/report/<report_name>.pdf
            <output_dir>/figures/{group_curves,comparisons,sweeps,
                                  per_class,detailed,benchmark}/
            <output_dir>/tables/        (CSV + Excel)
            <output_dir>/data/experiment_results.json
            <output_dir>/reports/       (conclusions, sweeps, verification,
                                         detailed_reports/)

        Safe to call repeatedly: files are overwritten in place. Every
        step is isolated; failures are listed in the report appendix and
        the PDF is still produced. Unavailable information is shown as
        "Not Available" / "Not Performed" and is never fabricated.

        Parameters
        ----------
        output_dir, report_name : destination and PDF base name.
        include_figures : generate and embed figures.
        include_sweeps : include parameter sweep analysis.
        include_benchmark : include the inference benchmark section.
        include_detailed_reports : per-experiment detail pages / files.
        include_classification_reports : classification reports in PDF.
        save_intermediate_files : also write tables, text reports, JSON.
        run_verification : reload BEST and LAST checkpoints of every
            experiment and verify their accuracy (adds an evaluation pass
            per checkpoint).
        run_benchmark_if_missing : run ``benchmark_inference()`` with
            default settings if no benchmark result exists yet.

        Returns
        -------
        Absolute path of the generated PDF.
        """

        output_dir = os.path.abspath(output_dir)

        report_path = os.path.join(
            output_dir, "report", f"{slugify(report_name)}.pdf"
        )

        ctx = ReportContext(
            output_dir=output_dir,
            report_path=report_path,
            include_figures=include_figures,
            include_sweeps=include_sweeps,
            include_benchmark=include_benchmark,
            include_detailed_reports=include_detailed_reports,
            include_classification_reports=include_classification_reports,
        )

        for sub in ("report", "figures", "tables", "data", "reports"):
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

        if not self.results:
            ctx.notes.append("No experiments have been executed.")

        original_figures_dir = self.figures_dir
        original_output_dir = self.output_dir
        original_registry = (self._figure_paths, self._path_owner)

        # Figures of this report live inside the report's output folder.
        self.figures_dir = os.path.join(output_dir, "figures")
        self.output_dir = output_dir
        self._figure_paths, self._path_owner = {}, {}

        try:

            if (
                run_benchmark_if_missing and include_benchmark
                and self.benchmark_df is None and self.results
            ):
                self._safe(ctx, "Inference benchmark", self.benchmark_inference,
                           plot=False)

            if run_verification:
                for name in self.results:
                    for kind in ("best", "last"):
                        self._safe(
                            ctx, f"Verification {kind.upper()} '{name}'",
                            self._verify_checkpoint, name, kind
                        )

            if include_figures and self.results:
                self._generate_report_figures(ctx)

            if save_intermediate_files and self.results:
                self._export_report_files(ctx)

            pdf_path = self._build_pdf_report(ctx)

        finally:
            self.figures_dir = original_figures_dir
            self.output_dir = original_output_dir
            self._figure_paths, self._path_owner = original_registry

        print(f"\nExperiment report generated: {pdf_path}")

        return pdf_path
