"""
Smoke test for runner.py
------------------------
Run from the project root:
    python smoke_test_runner.py

If your package is not named "src", set:
    Windows PowerShell:
      $env:RUNNER_MODULE="your_package.runner"
      python smoke_test_runner.py

This test uses synthetic Fashion-MNIST-shaped tensors and tiny fake models.
It does NOT download data or perform real training.
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import importlib
import json
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Import runner as a package module (runner.py uses relative imports).
# ---------------------------------------------------------------------
MODULE_NAME = os.getenv("RUNNER_MODULE", "src.runner")
R = importlib.import_module(MODULE_NAME)

ExperimentRunner = R.ExperimentRunner
ExperimentResult = R.ExperimentResult
ReportContext = R.ReportContext


# ---------------------------------------------------------------------
# Tiny substitutes: no real dataset, MLP, or training required.
# ---------------------------------------------------------------------
class FakeData:
    def get_loaders(self, batch_size, seed):
        g = torch.Generator().manual_seed(int(seed))
        x = torch.rand(40, 1, 28, 28, generator=g)
        y = torch.arange(40) % 10
        ds = TensorDataset(x, y)
        return (
            DataLoader(ds, batch_size=batch_size, shuffle=False),
            DataLoader(ds, batch_size=batch_size, shuffle=False),
        )


class FakeEvaluator:
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    def compute_metrics(self, y_true, y_pred):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(10)),
            average="macro", zero_division=0
        )
        return {
            "Accuracy (%)": float(accuracy_score(y_true, y_pred) * 100),
            "Macro Precision": float(p),
            "Macro Recall": float(r),
            "Macro F1-score": float(f),
        }

    def per_class_accuracy(self, y_true, y_pred):
        return np.array([
            100.0 * np.mean(np.asarray(y_pred)[np.asarray(y_true) == c] == c)
            if np.any(np.asarray(y_true) == c) else 0.0
            for c in range(10)
        ])

    def plot_learning_curves(self, history, name):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(history["train_accuracy"], label="train")
        ax.plot(history["evaluation_accuracy"], label="evaluation")
        ax.set_title(name)
        ax.legend()

    def plot_confusion_matrix(self, y_true, y_pred, name):
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        fig, ax = plt.subplots()
        ax.imshow(confusion_matrix(y_true, y_pred, labels=range(10)))
        ax.set_title(name)

    def plot_predictions(self, y_true, y_pred, y_prob, name):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(np.asarray(y_prob)[:10].max(axis=1), marker="o")
        ax.set_title(name)


class TinyMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class FakeTrainer:
    def __init__(self, model, config, device):
        self.model, self.config, self.device = model, config, device

    def fit(self, train_loader, evaluation_loader, log_every, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        history = {
            "train_loss": [1.8, 1.2, 0.9],
            "evaluation_loss": [1.9, 1.3, 1.0],
            "train_accuracy": [55.0, 72.0, 82.0],
            "evaluation_accuracy": [52.0, 70.0, 80.0],
        }
        for filename, epoch, loss in (
            ("best_checkpoint.pth", 2, 1.0),
            ("last_checkpoint.pth", 3, 1.0),
        ):
            torch.save({
                "epoch": epoch,
                "best_loss": loss,
                "model_state_dict": self.model.state_dict(),
            }, os.path.join(checkpoint_dir, filename))
        return history

    def load_checkpoint(self, path, load_optimizer=False):
        return torch.load(path, map_location=self.device, weights_only=True)

    def predict(self, loader):
        self.model.eval()
        ys, preds, probs = [], [], []
        with torch.no_grad():
            for x, y in loader:
                logits = self.model(x.to(self.device))
                p = torch.softmax(logits, dim=1).cpu().numpy()
                ys.extend(y.numpy().tolist())
                probs.extend(p.tolist())
                preds.extend(p.argmax(axis=1).tolist())
        return np.asarray(ys), np.asarray(preds), np.asarray(probs)


# Patch only the training/model dependencies used by runner.py.
R.MLP = TinyMLP
R.Trainer = FakeTrainer
R.DEVICE = torch.device("cpu")


def make_config(name, group, hidden_dim, seed):
    # runner methods access these attributes; MLP is replaced by TinyMLP.
    return SimpleNamespace(
        name=name,
        group=group,
        architecture="TinyMLP",
        optimizer_text="Adam (smoke test)",
        optimizer="Adam",
        learning_rate=0.001,
        lr=0.001,
        batch_size=8,
        epochs=3,
        weight_decay=0.0,
        seed=seed,
        hidden_dims=[hidden_dim],
        hidden_dim=hidden_dim,
    )


def make_result(config, checkpoint_dir, accuracy, seed):
    rng = np.random.default_rng(seed)
    y_true = np.arange(100) % 10
    y_pred = y_true.copy()
    # Introduce a controlled number of mistakes.
    n_wrong = int(round(100 * (1 - accuracy / 100)))
    if n_wrong:
        y_pred[:n_wrong] = (y_pred[:n_wrong] + 1) % 10

    logits = rng.normal(size=(100, 10))
    logits[np.arange(100), y_pred] += 3.0
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    y_prob = exp / exp.sum(axis=1, keepdims=True)

    paths = {
        "best": os.path.join(checkpoint_dir, "best_checkpoint.pth"),
        "last": os.path.join(checkpoint_dir, "last_checkpoint.pth"),
    }
    model = TinyMLP(config)
    for key, epoch in (("best", 2), ("last", 3)):
        torch.save({
            "epoch": epoch,
            "best_loss": 0.5,
            "model_state_dict": model.state_dict(),
        }, paths[key])

    history = {
        "train_loss": [1.8, 1.2, 0.9],
        "evaluation_loss": [1.9, 1.1, 1.0],
        "train_accuracy": [60.0, 78.0, 85.0],
        "evaluation_accuracy": [58.0, accuracy - 1.0, accuracy],
    }
    metrics = FakeEvaluator().compute_metrics(y_true, y_pred)

    return ExperimentResult(
        config=config,
        history=history,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        n_params=model.count_parameters(),
        train_seconds=1.25 + seed / 100,
        model_path=paths["best"],
        last_model_path=paths["last"],
        best_epoch=3,
        best_loss=1.0,
    )


# ---------------------------------------------------------------------
# Test runner: each method is isolated so one failure does not stop all.
# ---------------------------------------------------------------------
def main():
    passed, failed, skipped = [], [], []

    def check(label, fn):
        try:
            value = fn()
            passed.append(label)
            print(f"[PASS] {label}")
            return value
        except Exception as exc:
            failed.append((label, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
            return None

    with tempfile.TemporaryDirectory(prefix="runner_smoke_") as tmp:
        root = Path(tmp)
        runner = ExperimentRunner(
            data=FakeData(),
            evaluator=FakeEvaluator(),
            baseline_name="baseline",
            save_dir=str(root / "checkpoints"),
            figures_dir=str(root / "figures"),
        )
        runner.output_dir = str(root / "outputs")

        # Dataclasses and generic helper objects.
        cfg0 = make_config("baseline", "baseline", 32, 42)
        cfg1 = make_config("hidden_64", "hidden_width", 64, 43)
        cfg2 = make_config("hidden_128", "hidden_width", 128, 44)

        check("ExperimentResult dataclass", lambda: make_result(
            cfg0, str(root / "seed_ckpt"), 80.0, 42
        ))
        check("ReportContext dataclass", lambda: ReportContext(
            output_dir=str(root), report_path=str(root / "x.pdf"),
            include_figures=False, include_sweeps=False,
            include_benchmark=False, include_detailed_reports=False,
            include_classification_reports=False,
        ))

        # Exercise the actual run() path with fake Trainer and fake MLP.
        check("run() / checkpoint creation / result registration",
              lambda: runner.run(cfg0, log_every=1))

        # Add two synthetic results for comparisons and sweep-related paths.
        for cfg, acc, seed in ((cfg1, 82.0, 43), (cfg2, 84.0, 44)):
            def add(cfg=cfg, acc=acc, seed=seed):
                paths = runner._get_checkpoint_paths(cfg.name)
                result = make_result(cfg, paths["directory"], acc, seed)
                runner.results[cfg.name] = result
                return result
            check(f"register synthetic result: {cfg.name}", add)

        # Generic helpers.
        check("to_jsonable", lambda: R.to_jsonable({
            "array": np.array([1, 2]), "tensor": torch.tensor([3, 4]),
            "nan": float("nan"), "config": cfg0,
        }))
        check("_config_to_dict", lambda: R._config_to_dict(cfg0))
        check("_flatten", lambda: R._flatten("root", {"a": 1, "b": [2, 3]}, {}))
        check("_hist_at", lambda: R._hist_at(
            runner.results["baseline"].history, "evaluation_loss", 1
        ))
        check("_safe_sheet_name", lambda: R._safe_sheet_name("a/b:*?c", set()))
        check("_lines_to_markdown", lambda: R._lines_to_markdown(
            "Smoke", ["Header", "  - item", "", "Footer"]
        ))
        check("_fmt_delta", lambda: R._fmt_delta(82.0, 80.0))

        # Checkpoint and figure utilities.
        check("_get_checkpoint_paths", lambda: runner._get_checkpoint_paths("a/b"))
        check("_save_figure / _finish_figure / _figure_path", lambda: (
            runner._finish_figure(
                __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots()[0],
                "smoke_figure", "smoke", "smoke/figure",
                save=True, show=False
            ),
            runner._figure_path("smoke/figure")
        ))
        check("_run_evaluator_plot", lambda: runner._run_evaluator_plot(
            runner.evaluator.plot_learning_curves,
            (runner.results["baseline"].history, "baseline"),
            "smoke_eval_plot", "smoke/eval",
            save=True, show=False
        ))

        # Training orchestration.
        check("run_all()", lambda: runner.run_all(
            [make_config("run_all_case", "misc", 16, 50)], log_every=1
        ))
        check("groups()", runner.groups)
        check("summary()", runner.summary)
        check("summary(sort_by=...)", lambda: runner.summary(
            sort_by="Best Evaluation Acc (%)"
        ))
        check("best_name()", runner.best_name)

        # Plotting methods.
        for label, fn in [
            ("plot_group_curves()", lambda: runner.plot_group_curves(
                "baseline", save=True, show=False)),
            ("plot_final_comparison()", lambda: runner.plot_final_comparison(
                save=True, show=False)),
            ("_per_class_matrix()", runner._per_class_matrix),
            ("plot_per_class_accuracy()", lambda: runner.plot_per_class_accuracy(
                save=True, show=False)),
        ]:
            check(label, fn)

        # Sweep methods: available_sweeps is safe even if no sweep is configured.
        check("sweep_members()", lambda: runner.sweep_members(
            next(iter(R.SWEEP_PARAMS))
        ))
        check("available_sweeps()", runner.available_sweeps)
        for param in runner.available_sweeps():
            check(f"sweep_table({param})", lambda p=param: runner.sweep_table(p))
            check(f"plot_sweep({param})", lambda p=param: runner.plot_sweep(
                p, save=True, show=False))
            check(f"build_sweep_conclusion({param})",
                  lambda p=param: runner.build_sweep_conclusion(p))
            check(f"print_sweep_conclusion({param})",
                  lambda p=param: runner.print_sweep_conclusion(p))

        # Detailed analysis and checkpoint loading / verification.
        check("_classification_report_text()",
              lambda: runner._classification_report_text(
                  runner.results["baseline"]))
        check("_render_detail_figures()", lambda: runner._render_detail_figures(
            "baseline", save=True, show=False))
        check("show_detail()", lambda: runner.show_detail(
            "baseline", save=True, show=False,
            save_report=True, report_dir=str(root / "detail"),
        ))
        check("load_model(best)", lambda: runner.load_model("baseline", "best"))
        check("load_model(last)", lambda: runner.load_model("baseline", "last"))
        check("_load_model_quiet()", lambda: runner._load_model_quiet("baseline"))
        check("_verify_checkpoint(best)", lambda: runner._verify_checkpoint(
            "baseline", "best"))
        check("verify_saved_model(last)", lambda: runner.verify_saved_model(
            "baseline", "last"))

        # Conclusions, reports, and export helpers.
        check("build_conclusions()", runner.build_conclusions)
        check("print_conclusions()", lambda: runner.print_conclusions(
            str(root / "conclusions.md")))
        check("detailed_metrics_table()", runner.detailed_metrics_table)
        check("config_table()", runner.config_table)
        check("per_class_table()", runner.per_class_table)
        check("_executive_facts()", runner._executive_facts)
        check("_result_to_dict()", lambda: runner._result_to_dict(
            "baseline", include_predictions=True))
        json_path = check("export_results_json()", lambda:
            runner.export_results_json(str(root / "results.json"),
                                       include_predictions=True))
        if json_path:
            check("load_results_json()", lambda:
                  ExperimentRunner.load_results_json(json_path))

        check("save_summary()", lambda: runner.save_summary(
            str(root / "export"), formats=("csv",)))
        check("save_detailed_metrics()", lambda: runner.save_detailed_metrics(
            str(root / "export"), formats=("csv",)))
        check("save_conclusions()", lambda: runner.save_conclusions(
            str(root / "export")))
        check("save_detail_report()", lambda: runner.save_detail_report(
            "baseline", str(root / "export/details"), fmt="md"))
        check("build_detail_report()", lambda: runner.build_detail_report(
            "baseline", "txt"))
        check("_format_verification()", lambda: runner._format_verification(
            list(runner._verification.values()), "md"))
        check("save_verification_report()", lambda:
              runner.save_verification_report(str(root / "export")))
        check("save_sweep_tables()", lambda: runner.save_sweep_tables(
            str(root / "export"), formats=("csv",)))
        check("save_sweep_conclusions()", lambda:
              runner.save_sweep_conclusions(str(root / "export")))

        # Low-level text/table/path helpers.
        check("_ext()", lambda: runner._ext("report.md"))
        check("_resolve_dir()", lambda: runner._resolve_dir(
            str(root / "resolve"), "tables"))
        check("_write_text()", lambda: runner._write_text(
            str(root / "write" / "sample.txt"), "hello"))
        check("_write_lines()", lambda: runner._write_lines(
            str(root / "write" / "lines.md"), "Title", ["one", "two"]))
        check("_export_dataframe(csv)", lambda: runner._export_dataframe(
            pd.DataFrame({"a": [1, 2]}),
            str(root / "table_export" / "sample"), ("csv",),
            sheet_name="Smoke"))
        check("_write_excel()", lambda: runner._write_excel(
            {"Smoke": pd.DataFrame({"a": [1, 2]})},
            str(root / "table_export" / "sample.xlsx")))

        # Benchmark: small CPU-only test; a few runs to keep it quick.
        benchmark = check("benchmark_inference()", lambda:
            runner.benchmark_inference(
                names=["baseline"], batch_size=8, n_warmup=1, n_runs=2,
                include_single_sample=True, plot=True, save=True, show=False
            ))
        if isinstance(benchmark, pd.DataFrame):
            check("save_benchmark_results()", lambda:
                  runner.save_benchmark_results(
                      benchmark, str(root / "export"), formats=("csv",)
                  ))
            check("_plot_inference_benchmark()", lambda:
                  runner._plot_inference_benchmark(
                      benchmark, 8, save=True, show=False
                  ))

        # Private report sections are exercised with a minimal recording builder.
        class DummyPDFBuilder:
            def __getattr__(self, name):
                return lambda *a, **k: None

        ctx = ReportContext(
            output_dir=str(root / "report_out"),
            report_path=str(root / "report_out" / "dummy.pdf"),
            include_figures=False, include_sweeps=True,
            include_benchmark=True, include_detailed_reports=True,
            include_classification_reports=True,
        )
        dummy = DummyPDFBuilder()
        check("_generate_report_figures()", lambda:
              runner._generate_report_figures(ctx))
        check("_export_report_files()", lambda:
              runner._export_report_files(ctx))
        for method_name in (
            "_pdf_cover", "_pdf_executive_summary", "_pdf_configuration",
            "_pdf_comparison", "_pdf_curves", "_pdf_sweeps",
            "_pdf_per_class", "_pdf_detailed", "_pdf_checkpoints",
            "_pdf_benchmark", "_pdf_conclusions", "_pdf_appendix",
        ):
            check(method_name, lambda m=method_name:
                  getattr(runner, m)(dummy, ctx))

        check("_safe() catches exceptions", lambda: runner._safe(
            ctx, "intentional failure",
            lambda: (_ for _ in ()).throw(RuntimeError("expected"))
        ))
        check("_device_description()", R._device_description)

        # Full report: avoid real verification (the checkpoint path was
        # already tested), and avoid PDF dependency if report_pdf is absent.
        # This calls the actual orchestration and records any missing optional
        # dependency as a normal smoke-test failure.
        check("create_report() end-to-end", lambda: runner.create_report(
            output_dir=str(root / "full_report"),
            report_name="smoke_test_report",
            include_figures=False,
            include_sweeps=True,
            include_benchmark=True,
            include_detailed_reports=False,
            include_classification_reports=True,
            save_intermediate_files=True,
            run_verification=False,
            run_benchmark_if_missing=False,
        ))

        # Negative-path checks.
        def expect_exception(label, fn, exception_type):
            try:
                fn()
            except exception_type:
                passed.append(label)
                print(f"[PASS] {label} (expected {exception_type.__name__})")
            else:
                failed.append((label, f"Expected {exception_type.__name__}"))
                print(f"[FAIL] {label}: expected {exception_type.__name__}")

        expect_exception(
            "summary() rejects unknown sort column",
            lambda: runner.summary(sort_by="not_a_column"), ValueError
        )
        expect_exception(
            "load_model() rejects unknown experiment",
            lambda: runner.load_model("missing"), KeyError
        )
        expect_exception(
            "load_model() rejects invalid checkpoint type",
            lambda: runner.load_model("baseline", "middle"), ValueError
        )
        expect_exception(
            "benchmark_inference() rejects unknown experiment",
            lambda: runner.benchmark_inference(names=["missing"]),
            KeyError
        )

        print("\n" + "=" * 72)
        print(f"SMOKE TEST SUMMARY: {len(passed)} PASS | {len(failed)} FAIL")
        print("=" * 72)
        if failed:
            for label, error in failed:
                print(f" - {label}: {error}")
            raise SystemExit(1)

        print("All executed smoke tests passed.")
        print("Temporary test artifacts were removed automatically.")


if __name__ == "__main__":
    main()
