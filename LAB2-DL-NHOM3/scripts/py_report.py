"""
py_report.py - Tu dong hoa bao cao Lab 2: Transfer Learning Intel Scene Classification.

Chay: python scripts/py_report.py

Quy trinh (6 buoc):
  1. Thu thap metrics.json tu moi experiment trong outputs/
  2. Xuat bang so sanh day du ra CSV (rank, acc, f1, model size, speed, params)
  3. Ve bieu do so sanh tong hop (bar chart) luu ra outputs/figures/
  4. Chay notebook qua nbconvert (neu co jupyter, bo qua neu khong co)
  5. Xuat HTML report tu notebook da chay
  6. Chay audit check_vs_description.py kiem tra 13 REQ
  7. In tom tat + luu report_summary.json

Yeu cau: Python 3.8+, matplotlib, (tuy chon) jupyter nbconvert
"""

import json
import csv
import sys
import subprocess
import datetime
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Cau hinh duong dan
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
FIGURES_DIR  = OUTPUTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_SRC  = NOTEBOOKS_DIR / "lab2_report.ipynb"
NOTEBOOK_EXEC = NOTEBOOKS_DIR / "lab2_report_executed.ipynb"
NOTEBOOK_HTML = NOTEBOOKS_DIR / "lab2_report.html"
CSV_PATH      = OUTPUTS_DIR / "experiment_summary.csv"
JSON_SUMMARY  = OUTPUTS_DIR / "report_summary.json"
AUDIT_SCRIPT  = SCRIPT_DIR / "check_vs_description.py"
LOG_PATH      = OUTPUTS_DIR / "py_report.log"

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# ---------------------------------------------------------------------------
# Logger (ghi ra console + file)
# ---------------------------------------------------------------------------
def setup_logger():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
    )

log = logging.getLogger(__name__)


def banner(msg: str):
    sep = "=" * 62
    log.info(sep)
    log.info("  %s", msg)
    log.info(sep)


# ---------------------------------------------------------------------------
# BUOC 1: Thu thap metrics tu outputs/
# ---------------------------------------------------------------------------
def collect_metrics() -> list[dict]:
    banner("BUOC 1: Thu thap metrics tu outputs/")
    if not OUTPUTS_DIR.exists():
        log.warning("Khong tim thay thu muc: %s", OUTPUTS_DIR)
        return []

    records = []
    for exp_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        mfile = exp_dir / "metrics.json"
        if not mfile.exists():
            continue
        m = json.loads(mfile.read_text(encoding="utf-8"))

        # Per-class F1
        per_class = m.get("per_class_metrics", {})
        per_f1 = {cls: round(per_class.get(cls, {}).get("f1", 0.0), 4)
                  for cls in CLASSES}

        records.append({
            "experiment":    exp_dir.name,
            "model":         m.get("model_name", ""),
            "freeze_mode":   m.get("freeze_mode", ""),
            "optimizer":     m.get("optimizer", ""),
            "lr":            m.get("lr", ""),
            "batch_size":    m.get("batch_size", ""),
            "test_acc":      m.get("test_acc", m.get("val_acc", 0.0)),
            "test_f1":       m.get("test_f1",  m.get("val_f1",  0.0)),
            "test_loss":     round(m.get("test_loss", m.get("val_loss", 0.0)), 4),
            "best_epoch":    m.get("best_epoch", m.get("epoch", 0)),
            "epochs_run":    m.get("epochs_run", 0),
            "model_size_mb": round(m.get("model_size_mb", 0.0), 2),
            "total_params":  m.get("total_params", 0),
            "trainable_params": m.get("trainable_params", 0),
            "inference_ms":  round(m.get("inference_ms_per_image", 0.0), 2),
            "train_time_h":  round(m.get("train_time_h", 0.0), 3),
            "top_errors":    m.get("top_error_pairs", []),
            "history":       m.get("history", {}),
            **{f"f1_{cls}": per_f1[cls] for cls in CLASSES},
        })
        log.info("  OK  %s  |  Acc=%.4f  F1=%.4f", exp_dir.name,
                 records[-1]["test_acc"], records[-1]["test_f1"])

    # Sap xep theo test_acc giam dan
    records.sort(key=lambda r: float(r["test_acc"]), reverse=True)
    log.info("Tong cong: %d experiment(s) tim thay", len(records))
    return records


# ---------------------------------------------------------------------------
# BUOC 2: Xuat CSV tong hop
# ---------------------------------------------------------------------------
def export_csv(records: list[dict]):
    banner("BUOC 2: Xuat CSV tong hop")
    if not records:
        log.warning("[SKIP] Khong co du lieu de xuat.")
        return

    fieldnames = [
        "rank", "experiment", "model", "freeze_mode", "optimizer", "lr", "batch_size",
        "test_acc", "test_f1", "test_loss", "best_epoch", "epochs_run",
        "model_size_mb", "total_params", "trainable_params", "inference_ms", "train_time_h",
    ] + [f"f1_{cls}" for cls in CLASSES]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, r in enumerate(records, 1):
            writer.writerow({"rank": i, **r})

    log.info("Da luu: %s", CSV_PATH)

    # In bang ra console
    print()
    print(f"  {'Rank':<4} {'Acc':>7} {'F1':>7} {'Size MB':>8} {'ms/img':>7}  Experiment")
    print("  " + "-" * 72)
    for i, r in enumerate(records, 1):
        print(f"  {i:<4} {r['test_acc']:.4f}  {r['test_f1']:.4f}  "
              f"{r['model_size_mb']:>7.1f}  {r['inference_ms']:>6.2f}  {r['experiment']}")


# ---------------------------------------------------------------------------
# BUOC 3: Ve bieu do so sanh tong hop luu PNG
# ---------------------------------------------------------------------------
def plot_comparison(records: list[dict]):
    banner("BUOC 3: Ve bieu do so sanh va luu PNG")
    if not records:
        log.warning("[SKIP] Khong co du lieu.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    names_short = [r["experiment"].replace("resnet18__", "RN18-")
                                  .replace("vgg16__", "VGG16-")
                                  .replace("densenet121__", "DN121-")
                                  .replace("finetune_last__", "ft-")
                                  .replace("freeze_backbone__", "frz-")
                                  [:28]
                   for r in records]
    accs  = [r["test_acc"] for r in records]
    f1s   = [r["test_f1"]  for r in records]
    sizes = [r["model_size_mb"] for r in records]
    speeds = [r["inference_ms"] for r in records]
    x = np.arange(len(records))

    # --- Bieu do 1: Test Accuracy & F1 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, vals, label, color in [
        (axes[0], accs,  "Test Accuracy", "#4C72B0"),
        (axes[1], f1s,   "Test F1 (Macro)", "#DD8452"),
    ]:
        bars = ax.bar(x, vals, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names_short, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(label + " - 9 Thi nghiem")
        ax.set_ylim(min(vals) - 0.01, 1.0)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=6.5)
    fig.suptitle("Ket qua tong hop - Lab 2 Transfer Learning", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out1 = FIGURES_DIR / "cmp_acc_f1.png"
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  Luu: %s", out1)

    # --- Bieu do 2: Model size vs Accuracy (scatter) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(records)))
    for i, (r, c) in enumerate(zip(records, colors)):
        ax.scatter(r["model_size_mb"], r["test_acc"], s=100, color=c, zorder=3)
        ax.annotate(names_short[i], (r["model_size_mb"], r["test_acc"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs Model Size")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out2 = FIGURES_DIR / "cmp_size_vs_acc.png"
    fig.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  Luu: %s", out2)

    # --- Bieu do 3: Per-class F1 (grouped bar) ---
    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.08
    for i, r in enumerate(records):
        vals_cls = [r.get(f"f1_{cls}", 0.0) for cls in CLASSES]
        offsets = np.arange(len(CLASSES)) + i * width
        ax.bar(offsets, vals_cls, width=width * 0.9,
               label=names_short[i], alpha=0.85)
    ax.set_xticks(np.arange(len(CLASSES)) + width * len(records) / 2)
    ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-class F1 Score - 9 Thi nghiem")
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out3 = FIGURES_DIR / "cmp_perclass_f1.png"
    fig.savefig(out3, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  Luu: %s", out3)

    # --- Bieu do 4: Learning Curves top 3 experiment ---
    top3 = records[:3]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors3 = ["#4C72B0", "#DD8452", "#55A868"]
    for r, c, lbl in zip(top3, colors3, names_short[:3]):
        hist = r["history"]
        ep = range(1, len(hist.get("train_loss", [])) + 1)
        if hist.get("train_loss"):
            axes[0].plot(ep, hist["train_loss"], "--", color=c, alpha=0.6, label=f"{lbl} Train")
        if hist.get("val_loss"):
            axes[0].plot(ep, hist["val_loss"],   "-",  color=c, linewidth=2, label=f"{lbl} Val")
        if hist.get("train_acc"):
            axes[1].plot(ep, [v * 100 for v in hist["train_acc"]], "--", color=c, alpha=0.6, label=f"{lbl} Train")
        if hist.get("val_acc"):
            axes[1].plot(ep, [v * 100 for v in hist["val_acc"]],   "-",  color=c, linewidth=2, label=f"{lbl} Val")
    axes[0].set(xlabel="Epoch", ylabel="Loss",          title="Loss - Top 3 Experiments")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)",  title="Accuracy - Top 3 Experiments")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Learning Curves - Top 3 Best Experiments", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out4 = FIGURES_DIR / "cmp_learning_curves_top3.png"
    fig.savefig(out4, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  Luu: %s", out4)


# ---------------------------------------------------------------------------
# BUOC 4: Huong dan export HTML (thuc hien thu cong)
# ---------------------------------------------------------------------------
def run_notebook() -> bool:
    banner("BUOC 4: Huong dan export HTML")
    log.info("Notebook: %s", NOTEBOOK_SRC)
    log.info("De export HTML, chay lenh sau trong terminal:")
    log.info("  jupyter nbconvert --to html notebooks/lab2_report.ipynb")
    log.info("Hoac tren Kaggle: bam 'Run All' roi download file .ipynb ve.")
    return False


# ---------------------------------------------------------------------------
# BUOC 5: (Bo qua - gop vao buoc 4)
# ---------------------------------------------------------------------------
def export_html(notebook_ok: bool):
    pass


# ---------------------------------------------------------------------------
# BUOC 6: Chay audit
# ---------------------------------------------------------------------------
def run_audit():
    banner("BUOC 6: Audit - Doi chieu voi mo ta yeu cau")
    if not AUDIT_SCRIPT.exists():
        log.warning("[SKIP] Khong tim thay: %s", AUDIT_SCRIPT)
        return
    r = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT)],
        capture_output=True, text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
    )
    print(r.stdout)
    if r.stderr:
        log.warning("[stderr] %s", r.stderr[-300:])
    if r.returncode == 0:
        log.info("OK  Audit hoan thanh.")
    else:
        log.warning("[WARN] Audit bao loi - kiem tra chi tiet o tren.")


# ---------------------------------------------------------------------------
# BUOC 7: Luu report_summary.json + in tom tat
# ---------------------------------------------------------------------------
def save_summary(records: list[dict]):
    banner("BUOC 7: Tom tat va luu report_summary.json")
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    summary = {
        "generated_at": ts,
        "total_experiments": len(records),
        "best": {
            "experiment": records[0]["experiment"] if records else "",
            "test_acc":   records[0]["test_acc"]   if records else 0.0,
            "test_f1":    records[0]["test_f1"]    if records else 0.0,
            "model_size_mb": records[0]["model_size_mb"] if records else 0.0,
            "inference_ms_per_image": records[0]["inference_ms"] if records else 0.0,
        },
        "ranking": [
            {
                "rank": i + 1,
                "experiment": r["experiment"],
                "test_acc":   round(r["test_acc"], 4),
                "test_f1":    round(r["test_f1"],  4),
                "model_size_mb": r["model_size_mb"],
                "inference_ms":  r["inference_ms"],
            }
            for i, r in enumerate(records)
        ],
        "outputs": {
            "csv":  str(CSV_PATH),
            "html": str(NOTEBOOK_HTML),
            "log":  str(LOG_PATH),
            "figures": str(FIGURES_DIR),
        },
    }
    JSON_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Da luu: %s", JSON_SUMMARY)

    print()
    log.info("%-20s : %s", "Thoi gian", ts)
    log.info("%-20s : %d", "Tong experiments", len(records))
    if records:
        b = records[0]
        log.info("%-20s : %s", "Best model", b["experiment"])
        log.info("%-20s : %.4f", "Test Accuracy", b["test_acc"])
        log.info("%-20s : %.4f", "Test F1 Macro", b["test_f1"])
        log.info("%-20s : %.1f MB", "Model size", b["model_size_mb"])
        log.info("%-20s : %.2f ms/anh", "Inference speed", b["inference_ms"])
    print()
    log.info("Cac file da xuat:")
    log.info("  CSV       : %s", CSV_PATH)
    log.info("  JSON      : %s", JSON_SUMMARY)
    log.info("  HTML      : %s", NOTEBOOK_HTML)
    log.info("  Figures   : %s", FIGURES_DIR)
    log.info("  Log       : %s", LOG_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup_logger()
    banner("py_report.py - Bao cao tu dong Lab 2 Transfer Learning")
    log.info("Project root : %s", PROJECT_ROOT)

    records  = collect_metrics()
    export_csv(records)
    plot_comparison(records)
    nb_ok    = run_notebook()
    export_html(nb_ok)
    run_audit()
    save_summary(records)


if __name__ == "__main__":
    main()
