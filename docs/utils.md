# Module `utils` — constants and helpers

**File:** `src/utils.py`
**Depends on:** `numpy`, `torch` (no project modules)
**Used by:** every other module

## Purpose

`utils` holds the values and small helper functions that the rest of the project shares: the random seed, the training device, the class names, and two helpers for reproducibility and file naming. Keeping them in one place means a change (for example a new seed) only has to be made once.

## Contents

| Name | Type | Description |
|---|---|---|
| `SEED` | `int` | Global random seed, `42`. Also the default `seed` of every `ExperimentConfig`. |
| `DEVICE` | `torch.device` | `cuda` if a GPU is available, otherwise `cpu`. Models and batches are moved here. |
| `CLASS_NAMES` | `list[str]` | The 10 Fashion-MNIST class names in English, indexed by label (0 = T-shirt/top … 9 = Ankle boot). Used on plot axes and in reports. |
| `CLASS_NAMES_VI` | `list[str]` | The same 10 names in Vietnamese, shown in the dataset overview table. |
| `slugify(text)` | function | Converts an experiment name to a safe ASCII file name. |
| `set_seed(seed)` | function | Fixes the seed of Python `random`, NumPy and PyTorch (CPU and all GPUs). |

## Class labels

| Label | English | Vietnamese |
|---|---|---|
| 0 | T-shirt/top | Áo thun |
| 1 | Trouser | Quần dài |
| 2 | Pullover | Áo len chui đầu |
| 3 | Dress | Váy |
| 4 | Coat | Áo khoác |
| 5 | Sandal | Dép xăng-đan |
| 6 | Shirt | Áo sơ mi |
| 7 | Sneaker | Giày thể thao |
| 8 | Bag | Túi xách |
| 9 | Ankle boot | Bốt cổ thấp |

## Function details

### `slugify(text: str) -> str`

Experiment names are written in Vietnamese with spaces and punctuation, which is not safe for file names. `slugify`:

1. replaces `đ`/`Đ` with `d`/`D` (Unicode normalisation does not split this letter);
2. removes all other diacritics with NFKD normalisation;
3. replaces every run of non-alphanumeric characters with `_`;
4. trims leading/trailing `_` and lowercases the result.

```python
slugify("Cơ sở: 2 lớp ẩn (256-128)")   # 'co_so_2_lop_an_256_128'
slugify("Dropout 0.5")                 # 'dropout_0_5'
```

`ExperimentRunner` uses it to name the sub-directory inside `saved_models/` for each experiment's checkpoints (e.g. `saved_models/co_so_2_lop_an_256_128/best_checkpoint.pth`).

### `set_seed(seed: int) -> None`

Calls `random.seed`, `np.random.seed`, `torch.manual_seed` and, if CUDA is available, `torch.cuda.manual_seed_all`. `ExperimentRunner.run` calls it before building each model, so every experiment starts from the same initial state and can be reproduced.

## Notes

- On a GPU, some cuDNN operations are not fully deterministic, so results can differ very slightly between runs even with the same seed. On a CPU the results are repeatable.
