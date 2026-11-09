#!/usr/bin/env bash
# run_normal.sh - Chạy train + evaluate đầy đủ (normal run).
# Dùng trên Kaggle hoặc Linux/Mac.
# Trên Windows: chạy notebook lab2_report.ipynb trực tiếp.
#
# Lưu ý:
# - Script này gọi notebook qua nbconvert (nếu có) hoặc hướng dẫn chạy thủ công.
# - Trên Kaggle: nên dùng "Run All" trong giao diện notebook thay vì script này.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "🏋️  Chạy Normal Run - Lab 2 Pre-trained CNN"
echo "   Project root: $PROJECT_ROOT"
echo "=============================================="

cd "$PROJECT_ROOT"

# Kiểm tra môi trường cơ bản
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"

# Kiểm tra src có import được không
python -c "
import sys
sys.path.insert(0, '.')
from src.models import build_model, list_models
print(f'Models có sẵn: {list_models()}')
print('✅ Import src/ thành công')
"

echo ""
echo "Cách chạy full notebook (khuyến nghị):"
echo "  1. Trên Kaggle: bấm 'Run All' trong notebook lab2_report.ipynb"
echo "  2. Local (nếu đã cài jupyter): jupyter nbconvert --to notebook --execute notebooks/lab2_report.ipynb"
echo ""

# Nếu có jupyter nbconvert, chạy notebook tự động
if command -v jupyter &>/dev/null; then
    echo "Phát hiện jupyter — đang chạy notebook..."
    jupyter nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.timeout=7200 \
        --output notebooks/lab2_report_executed.ipynb \
        notebooks/lab2_report.ipynb
    echo "✅ Notebook đã chạy xong. Kết quả tại notebooks/lab2_report_executed.ipynb"
else
    echo "⚠️  Không tìm thấy jupyter. Chạy notebook thủ công theo hướng dẫn ở trên."
fi
