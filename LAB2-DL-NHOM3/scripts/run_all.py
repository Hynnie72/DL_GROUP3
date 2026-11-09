"""
run_all.py - Pipeline 1 lenh day du cho Windows.

Chay: python scripts/run_all.py

Quy trinh:
  smoke -> (skip train neu da co outputs) -> py_report (normal run + audit)

Phu hop voi quy trinh giao vien huong dan:
  small prompt -> smoke run -> normal run -> audit
"""

import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SMOKE_TEST   = PROJECT_ROOT / "tests" / "smoke_test.py"
PY_REPORT    = SCRIPT_DIR / "py_report.py"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"


def run(cmd: list[str], cwd=None) -> int:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=cwd or str(PROJECT_ROOT))
    return r.returncode


def main():
    print("=" * 60)
    print("  run_all.py - Pipeline Lab 2 (Windows)")
    print("=" * 60)

    # ------- GIAI DOAN 3a: smoke run -------
    print("\n[GIAI DOAN 3a] SMOKE RUN")
    if not SMOKE_TEST.exists():
        print(f"  [WARN] Khong tim thay {SMOKE_TEST} - bo qua smoke.")
    else:
        code = run([sys.executable, str(SMOKE_TEST)])
        if code != 0:
            print("\n  [FAIL] Smoke test that bai. Dung pipeline.")
            print("         Sua loi o tren truoc khi chay lai.")
            sys.exit(1)
        print("\n  [OK] Smoke test PASS.")

    # ------- Kiem tra outputs/ co san khong -------
    exp_dirs = [d for d in OUTPUTS_DIR.iterdir() if (d / "metrics.json").exists()] \
               if OUTPUTS_DIR.exists() else []
    print(f"\n[INFO] Tim thay {len(exp_dirs)} experiment(s) da co metrics.json trong outputs/.")
    if len(exp_dirs) == 0:
        print("  [INFO] Chua co ket qua train. Hay chay train tren Kaggle truoc,")
        print("         sau do chay lai script nay.")

    # ------- GIAI DOAN 3b: normal run = py_report -------
    print("\n[GIAI DOAN 3b] NORMAL RUN + AUDIT (py_report.py)")
    code = run([sys.executable, str(PY_REPORT)])
    if code != 0:
        print("\n  [WARN] py_report.py ket thuc voi loi. Kiem tra log o tren.")
    else:
        print("\n  [OK] py_report.py hoan thanh.")

    print("\n" + "=" * 60)
    print("  Pipeline hoan tat!")
    print("=" * 60)


if __name__ == "__main__":
    main()
