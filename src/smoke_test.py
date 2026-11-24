"""smoke_test.py – Kiểm tra toàn bộ pipeline trước khi training thật."""

import sys
import os
sys.path.insert(0, 'd:/coursework/src')

import torch
import numpy as np

# ============================================================
# TEST 1: Imports
# ============================================================
print("=" * 60)
print("TEST 1: Imports")
print("=" * 60)
from dataset import (ChestXrayAgeDataset, get_train_transforms, get_val_transforms,
                     METADATA_DIM, FINDING_LABELS, NUM_FINDING_LABELS,
                     encode_finding_labels)
from model import MetadataAwareEfficientNetV2
from utils import (
    TrainingHistory, check_nan_inf, compute_l1_loss,
    compute_metrics, count_parameters, format_time,
    get_device, get_logger, load_checkpoint,
    plot_training_curves, save_checkpoint, set_seed,
)
# ============================================================
# TEST 1b: encode_finding_labels unit test
# ============================================================
print()
print("=" * 60)
print("TEST 1b: encode_finding_labels")
print("=" * 60)
vec_no_finding = encode_finding_labels("No Finding")
assert vec_no_finding.sum() == 0, "No Finding → all zeros"
print(f"  'No Finding' → {vec_no_finding}  (all zeros: OK)")

vec_multi = encode_finding_labels("Cardiomegaly|Effusion")
assert vec_multi[FINDING_LABELS.index("Cardiomegaly")] == 1.0
assert vec_multi[FINDING_LABELS.index("Effusion")] == 1.0
assert vec_multi.sum() == 2.0
print(f"  'Cardiomegaly|Effusion' → sum={vec_multi.sum():.0f}  (expect 2: OK)")
print("encode_finding_labels: OK")
print(f"METADATA_DIM = {METADATA_DIM}  (sex=1, view=1, findings={NUM_FINDING_LABELS})")
print(f"FINDING_LABELS = {FINDING_LABELS}")

# ============================================================
# TEST 2: Dataset init
# ============================================================
print()
print("=" * 60)
print("TEST 2: Dataset init + __getitem__")
print("=" * 60)

val_tf = get_val_transforms(224, grayscale=True)
ds = ChestXrayAgeDataset(
    csv_path=r'd:\coursework\data\CXR8\CXR8\Data_Entry_2017_v2020.csv',
    images_root=r'd:\coursework\data\CXR8\CXR8\images',
    split_list_path=r'd:\coursework\data\CXR8\CXR8\train_val_list.txt',
    transform=val_tf,
    min_age=1, max_age=100,
)
stats = ds.get_age_stats()
print(f"Dataset size : {len(ds)}")
print(f"Age mean     : {stats['mean']:.1f}")
print(f"Age std      : {stats['std']:.1f}")
print(f"Age range    : {stats['min']:.0f} – {stats['max']:.0f}")

# Load one sample
img, meta, target = ds[0]
print(f"Image shape  : {img.shape}  (expect [1, 224, 224])")
print(f"Metadata     : shape={meta.shape}  (expect [{METADATA_DIM}] = sex+view+14 findings)")
print(f"              sex={meta[0].item():.0f}, view={meta[1].item():.0f}, findings_sum={meta[2:].sum().item():.0f}")
print(f"Target age   : {target.item():.1f}")
assert img.shape == (1, 224, 224), f"Unexpected image shape: {img.shape}"
assert meta.shape == (METADATA_DIM,), f"Unexpected metadata shape: {meta.shape}  (expected {METADATA_DIM})"
print("Dataset: OK")

# ============================================================
# TEST 3: Model init + forward pass
# ============================================================
print()
print("=" * 60)
print("TEST 3: Model init + forward pass")
print("=" * 60)

set_seed(42)
model = MetadataAwareEfficientNetV2(
    metadata_dim=METADATA_DIM,   # 16 = sex(1) + view(1) + findings(14)
    pretrained=False,   # don't download weights in smoke test
    in_channels=1,
    meta_embedding_dim=128,
    fusion_dim=256,
    head_dropout=0.3,
)
model.eval()
n_params = count_parameters(model)
print(f"Trainable parameters: {n_params:,}")

# Forward pass with batch=2
dummy_img  = torch.randn(2, 1, 224, 224)
dummy_meta = torch.randn(2, METADATA_DIM)   # 16-dim
with torch.no_grad():
    out = model(dummy_img, dummy_meta)
print(f"Output shape : {out.shape}  (expect [2, 1])")
assert out.shape == (2, 1), f"Unexpected output shape: {out.shape}"
assert not torch.isnan(out).any(), "NaN in model output!"
print("Model forward pass: OK")

# ============================================================
# TEST 4: Loss + backward
# ============================================================
print()
print("=" * 60)
print("TEST 4: Loss functions + backward")
print("=" * 60)

model.train()
dummy_img  = torch.randn(2, 1, 224, 224, requires_grad=False)
dummy_meta = torch.randn(2, METADATA_DIM)   # 16-dim
dummy_tgt  = torch.tensor([45.0, 60.0])

for loss_name, criterion in [("MAE (L1Loss)", torch.nn.L1Loss()),
                               ("MSE (MSELoss)", torch.nn.MSELoss())]:
    preds = model(dummy_img, dummy_meta).squeeze(1)
    loss  = criterion(preds, dummy_tgt)
    l1_reg = compute_l1_loss(model, lambda_l1=1e-5)
    total  = loss + l1_reg
    total.backward()
    print(f"  {loss_name}: data_loss={loss.item():.4f}, l1_reg={l1_reg.item():.6f}, total={total.item():.4f} → backward OK")
    # zero grads for next iteration
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

# ============================================================
# TEST 5: Metrics
# ============================================================
print()
print("=" * 60)
print("TEST 5: compute_metrics")
print("=" * 60)
preds_np   = np.array([40.0, 55.0, 70.0, 30.0])
targets_np = np.array([42.0, 50.0, 68.0, 35.0])
m = compute_metrics(preds_np, targets_np)
print(f"  MAE  : {m['mae']:.4f}  (expect ~3.0)")
print(f"  MSE  : {m['mse']:.4f}")
print(f"  RMSE : {m['rmse']:.4f}")
print(f"  R²   : {m['r2']:.4f}")
print("Metrics: OK")

# ============================================================
# TEST 6: Checkpoint save/load
# ============================================================
print()
print("=" * 60)
print("TEST 6: Checkpoint save/load")
print("=" * 60)
import tempfile

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
state = {
    "epoch": 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": None,
    "best_mae": 5.0,
    "best_epoch": 1,
    "early_stop_counter": 0,
    "config": {"loss": "mae"},
}
with tempfile.TemporaryDirectory() as tmpdir:
    path = save_checkpoint(state, tmpdir, "test.pth")
    # Load into fresh model
    model2 = MetadataAwareEfficientNetV2(
        metadata_dim=METADATA_DIM, pretrained=False, in_channels=1
    )
    ckpt = load_checkpoint(path, model2, device=torch.device("cpu"))
    assert ckpt["epoch"] == 1
    assert ckpt["best_mae"] == 5.0
print("Checkpoint save/load: OK")

# ============================================================
# TEST 7: TrainingHistory + plot
# ============================================================
print()
print("=" * 60)
print("TEST 7: TrainingHistory + plotting")
print("=" * 60)
import tempfile

h = TrainingHistory()
for ep in range(1, 4):
    h.append({
        "epoch": ep, "train_loss": 10.0/ep, "train_mae": 8.0/ep,
        "train_mse": 100.0/ep, "train_l1_reg": 0.001,
        "val_loss": 11.0/ep, "val_mae": 9.0/ep,
        "val_mse": 110.0/ep, "val_rmse": 10.5/ep, "val_r2": 0.3*ep,
        "lr": 1e-4, "epoch_duration_s": 60.0,
    })

with tempfile.TemporaryDirectory() as tmpdir:
    h.to_csv(os.path.join(tmpdir, "history.csv"))
    h.to_json(os.path.join(tmpdir, "history.json"))
    plot_training_curves(h, tmpdir, "smoke_test")
    files = os.listdir(tmpdir)
    print(f"  Generated files: {files}")
    assert "history.csv" in files
    assert "smoke_test_loss.png" in files
print("TrainingHistory + plotting: OK")

# ============================================================
print()
print("=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
