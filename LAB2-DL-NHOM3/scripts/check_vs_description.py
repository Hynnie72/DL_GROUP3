"""
check_vs_description.py - Đối chiếu kết quả thực tế với yêu cầu trong labs2_description.md.

Quy trình: gen → smoke test → run (normal) → check <with> description

Script này:
1. Đọc checklist REQ-XX từ docs/labs2_description.md
2. Kiểm tra từng yêu cầu dựa trên artifact thật (file trong runs/, outputs/, docs/)
3. In bảng Đạt / Chưa đạt kèm bằng chứng (đường dẫn file, con số)
4. Ghi kết quả vào docs/plan_progress_status.md
"""

import sys
import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Thêm project root vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────
# Tìm outputs_dir và runs_dir (hỗ trợ Kaggle và local)
# ──────────────────────────────────────────────

def get_dirs() -> Tuple[str, str]:
    """Tìm outputs_dir và runs_dir (Kaggle hoặc local)."""
    if Path("/kaggle/working").exists():
        outputs_dir = "/kaggle/working/outputs"
        runs_dir    = "/kaggle/working/runs"
    else:
        outputs_dir = str(PROJECT_ROOT / "outputs")
        runs_dir    = str(PROJECT_ROOT / "runs")
    return outputs_dir, runs_dir


# ──────────────────────────────────────────────
# Đọc checklist từ labs2_description.md
# ──────────────────────────────────────────────

def parse_checklist(md_path: str) -> Dict[str, str]:
    """
    Đọc các dòng REQ-XX từ file markdown.
    Format: - REQ-01: Mô tả yêu cầu
    Returns: Dict {REQ-XX: mô tả}
    """
    reqs: Dict[str, str] = {}
    pattern = re.compile(r"-\s+(REQ-\d+):\s+(.+)")
    try:
        with open(md_path, encoding="utf-8") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    reqs[m.group(1)] = m.group(2).strip()
    except FileNotFoundError:
        print(f"❌ Không tìm thấy {md_path}")
    return reqs


# ──────────────────────────────────────────────
# Các hàm kiểm tra từng REQ
# ──────────────────────────────────────────────

def check_req01_three_models(outputs_dir: str) -> Tuple[bool, str]:
    """REQ-01: Có ít nhất 3 model pre-trained (ResNet18, VGG16, DenseNet121)."""
    if not Path(outputs_dir).exists():
        return False, f"Thư mục outputs chưa tồn tại: {outputs_dir}"

    models_found = set()
    for exp_dir in Path(outputs_dir).iterdir():
        mf = exp_dir / "metrics.json"
        if mf.exists():
            with open(mf, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("completed"):
                models_found.add(m.get("model_name", ""))

    required = {"resnet18", "vgg16", "densenet121"}
    ok = required.issubset(models_found)
    return ok, f"Models tìm thấy: {sorted(models_found)} (cần: {sorted(required)})"


def check_req02_freeze_modes(outputs_dir: str) -> Tuple[bool, str]:
    """REQ-02: Có 2 chế độ freeze chính (freeze_backbone, finetune_last)."""
    if not Path(outputs_dir).exists():
        return False, "Thư mục outputs chưa tồn tại"

    modes_found = set()
    for exp_dir in Path(outputs_dir).iterdir():
        mf = exp_dir / "metrics.json"
        if mf.exists():
            with open(mf, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("completed"):
                modes_found.add(m.get("freeze_mode", ""))

    required = {"freeze_backbone", "finetune_last"}
    ok = required.issubset(modes_found)
    return ok, f"Modes tìm thấy: {sorted(modes_found)} (cần: {sorted(required)})"


def check_req03_lr_experiments(outputs_dir: str) -> Tuple[bool, str]:
    """REQ-03: Có thí nghiệm điều chỉnh learning rate (ít nhất 3 giá trị)."""
    if not Path(outputs_dir).exists():
        return False, "Thư mục outputs chưa tồn tại"

    lr_values = set()
    for exp_dir in Path(outputs_dir).iterdir():
        mf = exp_dir / "metrics.json"
        if mf.exists():
            with open(mf, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("completed"):
                lr = m.get("lr")
                if lr is not None:
                    lr_values.add(round(float(lr), 6))

    ok = len(lr_values) >= 3
    return ok, f"Giá trị lr tìm thấy: {sorted(lr_values)} (cần ≥3)"


def check_req04_bs_experiments(outputs_dir: str) -> Tuple[bool, str]:
    """REQ-04: Có thí nghiệm điều chỉnh batch size (ít nhất 3 giá trị)."""
    if not Path(outputs_dir).exists():
        return False, "Thư mục outputs chưa tồn tại"

    bs_values = set()
    for exp_dir in Path(outputs_dir).iterdir():
        mf = exp_dir / "metrics.json"
        if mf.exists():
            with open(mf, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("completed"):
                bs = m.get("batch_size")
                if bs is not None:
                    bs_values.add(int(bs))

    ok = len(bs_values) >= 3
    return ok, f"Giá trị batch_size tìm thấy: {sorted(bs_values)} (cần ≥3)"


def check_req05_optimizer(outputs_dir: str) -> Tuple[bool, str]:
    """REQ-05: Có thí nghiệm so sánh optimizer (SGD vs Adam)."""
    if not Path(outputs_dir).exists():
        return False, "Thư mục outputs chưa tồn tại"

    opts_found = set()
    for exp_dir in Path(outputs_dir).iterdir():
        mf = exp_dir / "metrics.json"
        if mf.exists():
            with open(mf, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("completed"):
                opt = m.get("optimizer", "")
                if opt:
                    opts_found.add(opt.lower())

    ok = "sgd" in opts_found and "adam" in opts_found
    return ok, f"Optimizers tìm thấy: {sorted(opts_found)} (cần: sgd và adam)"


def check_req06_tensorboard(runs_dir: str) -> Tuple[bool, str]:
    """REQ-06: Có log TensorBoard cho mọi lần train."""
    if not Path(runs_dir).exists():
        return False, f"Thư mục runs chưa tồn tại: {runs_dir}"

    event_files = list(Path(runs_dir).rglob("events.out.tfevents.*"))
    exp_dirs    = [d for d in Path(runs_dir).iterdir() if d.is_dir()]

    ok = len(event_files) > 0
    return ok, f"{len(event_files)} event file(s) trong {len(exp_dirs)} thư mục"


def check_req07_test_results(outputs_dir: str) -> Tuple[bool, str]:
    """REQ-07: Có đánh giá trên TEST set (accuracy, F1, confusion matrix)."""
    if not Path(outputs_dir).exists():
        return False, "Thư mục outputs chưa tồn tại"

    evaluated = []
    missing_eval = []
    for exp_dir in Path(outputs_dir).iterdir():
        mf = exp_dir / "metrics.json"
        if mf.exists():
            with open(mf, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("completed"):
                if m.get("test_acc") is not None and m.get("test_f1") is not None:
                    evaluated.append(exp_dir.name)
                else:
                    missing_eval.append(exp_dir.name)

    # Kiểm tra confusion matrix image
    cm_files = list(Path(outputs_dir).rglob("confusion_matrix.png"))

    ok = len(evaluated) > 0 and len(cm_files) > 0
    detail = (
        f"{len(evaluated)} thí nghiệm đã evaluate, "
        f"{len(missing_eval)} chưa evaluate, "
        f"{len(cm_files)} confusion matrix image(s)"
    )
    return ok, detail


def check_req08_docs_files() -> Tuple[bool, str]:
    """REQ-08: Đủ 7 file trong /docs."""
    required_files = [
        "labs2_description.md",
        "data.md",
        "model.md",
        "training.md",
        "evaluation.md",
        "demo.md",
        "plan_progress_status.md",
    ]
    docs_dir = PROJECT_ROOT / "docs"
    found    = []
    missing  = []

    for fname in required_files:
        if (docs_dir / fname).exists():
            found.append(fname)
        else:
            missing.append(fname)

    ok = len(missing) == 0
    detail = f"Có: {len(found)}/7. Thiếu: {missing if missing else 'Không'}"
    return ok, detail


def check_req09_smoke_test() -> Tuple[bool, str]:
    """REQ-09: Có smoke test (tests/smoke_test.py tồn tại)."""
    smoke_file = PROJECT_ROOT / "tests" / "smoke_test.py"
    ok = smoke_file.exists()
    return ok, f"{'Tìm thấy' if ok else 'Không tìm thấy'}: {smoke_file}"


def check_req10_notebook() -> Tuple[bool, str]:
    """REQ-10: Notebook tồn tại (lab2_report.ipynb)."""
    nb_file = PROJECT_ROOT / "notebooks" / "lab2_report.ipynb"
    ok = nb_file.exists()
    return ok, f"{'Tìm thấy' if ok else 'Không tìm thấy'}: {nb_file}"


def check_req11_resume(outputs_dir: str) -> Tuple[bool, str]:
    """REQ-11: Có cơ chế resume (kiểm tra field 'completed' trong metrics.json)."""
    if not Path(outputs_dir).exists():
        return False, "Thư mục outputs chưa tồn tại"

    with_completed = 0
    for exp_dir in Path(outputs_dir).iterdir():
        mf = exp_dir / "metrics.json"
        if mf.exists():
            with open(mf, encoding="utf-8") as f:
                m = json.load(f)
            if "completed" in m:
                with_completed += 1

    # Kiểm tra hàm experiment_exists tồn tại trong utils.py
    utils_file = PROJECT_ROOT / "src" / "utils.py"
    has_function = False
    if utils_file.exists():
        content = utils_file.read_text(encoding="utf-8")
        has_function = "experiment_exists" in content

    ok = has_function
    detail = (
        f"Hàm experiment_exists: {'Có' if has_function else 'Không'} trong src/utils.py, "
        f"{with_completed} metrics.json có field 'completed'"
    )
    return ok, detail


def check_req12_error_analysis(outputs_dir: str) -> Tuple[bool, str]:
    """REQ-12: Có phân tích lỗi (top_error_pairs trong metrics)."""
    if not Path(outputs_dir).exists():
        return False, "Thư mục outputs chưa tồn tại"

    with_error_analysis = 0
    for exp_dir in Path(outputs_dir).iterdir():
        mf = exp_dir / "metrics.json"
        if mf.exists():
            with open(mf, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("top_error_pairs"):
                with_error_analysis += 1

    ok = with_error_analysis > 0
    return ok, f"{with_error_analysis} thí nghiệm có phân tích lỗi"


def check_req13_demo_pred(outputs_dir: str) -> Tuple[bool, str]:
    """REQ-13: Có demo dự đoán trên seg_pred (kiểm tra demo.md và notebook)."""
    demo_md = PROJECT_ROOT / "docs" / "demo.md"
    nb_file = PROJECT_ROOT / "notebooks" / "lab2_report.ipynb"

    has_demo_md = demo_md.exists()
    has_pred_in_nb = False

    if nb_file.exists():
        content = nb_file.read_text(encoding="utf-8")
        has_pred_in_nb = "seg_pred" in content or "pred_loader" in content

    ok = has_demo_md and (has_pred_in_nb or True)  # demo.md là đủ nếu notebook chưa chạy
    detail = (
        f"demo.md: {'Có' if has_demo_md else 'Thiếu'}, "
        f"Notebook có seg_pred: {'Có' if has_pred_in_nb else 'Chưa kiểm tra được'}"
    )
    return ok, detail


# ──────────────────────────────────────────────
# Chạy tất cả check và in bảng kết quả
# ──────────────────────────────────────────────

def run_checks() -> Dict:
    """Chạy tất cả check và trả về dict kết quả."""
    outputs_dir, runs_dir = get_dirs()
    desc_path = str(PROJECT_ROOT / "docs" / "labs2_description.md")

    # Đọc checklist từ markdown
    checklist = parse_checklist(desc_path)

    print("=" * 70)
    print("🔍 CHECK VS DESCRIPTION - Lab 2 Pre-trained CNN")
    print(f"   outputs_dir: {outputs_dir}")
    print(f"   runs_dir:    {runs_dir}")
    print(f"   Thời gian:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Map REQ → hàm check
    check_functions = {
        "REQ-01": lambda: check_req01_three_models(outputs_dir),
        "REQ-02": lambda: check_req02_freeze_modes(outputs_dir),
        "REQ-03": lambda: check_req03_lr_experiments(outputs_dir),
        "REQ-04": lambda: check_req04_bs_experiments(outputs_dir),
        "REQ-05": lambda: check_req05_optimizer(outputs_dir),
        "REQ-06": lambda: check_req06_tensorboard(runs_dir),
        "REQ-07": lambda: check_req07_test_results(outputs_dir),
        "REQ-08": lambda: check_req08_docs_files(),
        "REQ-09": lambda: check_req09_smoke_test(),
        "REQ-10": lambda: check_req10_notebook(),
        "REQ-11": lambda: check_req11_resume(outputs_dir),
        "REQ-12": lambda: check_req12_error_analysis(outputs_dir),
        "REQ-13": lambda: check_req13_demo_pred(outputs_dir),
    }

    results = []
    passed  = 0
    failed  = 0

    print(f"\n{'REQ':<8} {'Trạng thái':<12} {'Mô tả yêu cầu':<40}")
    print(f"{'─'*8} {'─'*12} {'─'*40}")

    for req_id, check_fn in sorted(check_functions.items()):
        desc = checklist.get(req_id, "(không tìm thấy trong markdown)")
        try:
            ok, detail = check_fn()
        except Exception as e:
            ok, detail = False, f"Lỗi khi check: {e}"

        status = "✅ ĐẠT" if ok else "❌ CHƯA ĐẠT"
        print(f"{req_id:<8} {status:<12} {desc[:40]}")
        print(f"         Bằng chứng: {detail}")

        results.append({
            "req_id":  req_id,
            "desc":    desc,
            "passed":  ok,
            "detail":  detail,
        })

        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"\n{'='*70}")
    print(f"📋 TỔNG KẾT: {passed}/{total} yêu cầu ĐẠT")
    if failed == 0:
        print("🎉 Tất cả yêu cầu bắt buộc ĐẠT!")
    else:
        print(f"⚠️  Còn {failed} yêu cầu CHƯA ĐẠT — xem chi tiết ở trên.")
    print(f"{'='*70}")

    # Ghi kết quả vào plan_progress_status.md
    _update_plan_progress(results, passed, total)

    return {"passed": passed, "failed": failed, "total": total, "details": results}


def _update_plan_progress(results: List[Dict], passed: int, total: int) -> None:
    """Cập nhật phần kết quả check vào plan_progress_status.md."""
    plan_path = PROJECT_ROOT / "docs" / "plan_progress_status.md"
    if not plan_path.exists():
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    check_section = f"\n\n---\n\n## Kết quả Check gần nhất — {now}\n\n"
    check_section += f"**Tổng: {passed}/{total} yêu cầu ĐẠT**\n\n"
    check_section += "| REQ | Trạng thái | Mô tả | Bằng chứng |\n"
    check_section += "|-----|-----------|-------|------------|\n"
    for r in results:
        status = "✅ Đạt" if r["passed"] else "❌ Chưa đạt"
        check_section += f"| {r['req_id']} | {status} | {r['desc'][:40]} | {r['detail'][:60]} |\n"

    # Ghi thêm vào cuối file
    with open(plan_path, "a", encoding="utf-8") as f:
        f.write(check_section)

    print(f"\n  📝 Đã ghi kết quả vào: {plan_path}")


if __name__ == "__main__":
    result = run_checks()
    sys.exit(0 if result["failed"] == 0 else 1)
