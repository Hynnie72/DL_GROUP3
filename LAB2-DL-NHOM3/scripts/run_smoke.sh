#!/usr/bin/env bash
# run_smoke.sh - Chạy smoke test nhanh (< 2 phút)
# Dùng trên Kaggle hoặc Linux/Mac.
# Trên Windows: chạy tests/smoke_test.py trực tiếp bằng Python.

set -e  # Dừng nếu có lỗi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "🚀 Chạy Smoke Test - Lab 2 Pre-trained CNN"
echo "   Project root: $PROJECT_ROOT"
echo "=============================================="

# Chuyển về project root để đường dẫn import src/ hoạt động
cd "$PROJECT_ROOT"

python tests/smoke_test.py

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Smoke test PASS — môi trường sẵn sàng để chạy normal."
else
    echo ""
    echo "❌ Smoke test FAIL — kiểm tra lỗi ở trên trước khi chạy normal."
fi

exit $EXIT_CODE
