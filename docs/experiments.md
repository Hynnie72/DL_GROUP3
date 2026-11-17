# Module `experiments` — Experiment Configuration and Management

**File:** `src/experiments.py`  
**Depends on:** `dataclasses`, `typing`, `src.config` (`ExperimentConfig`)  
**Used by:** Main notebook, three sweep notebooks, `runner`, `main.py`

## Purpose

This module defines and manages all 23 experiments used to evaluate the effects of different hyperparameters and architectural configurations on an MLP model.

The experiments are organized into 10 groups:

- **Groups 1–7:** Core experiments that investigate individual factors. These are executed by the main notebook.
- **Groups 8–10:** Extended experiments that complete three increasing sweeps covering network depth, learning rate and batch size. Each sweep is executed by its own notebook.

The baseline configuration serves as the common reference point.

Each experiment is designed to change one configuration factor relative to the baseline. This makes it possible to analyze how individual design choices affect model performance.

## Contents

| Name | Description |
|---|---|
| `BASELINE` | Name of the baseline experiment. |
| `CORE_GROUPS` | List of the seven core experiment groups. |
| `DEPTH_SIZES` | Hidden-layer widths used by the increasing-depth sweep. |
| `SWEEPS` | Maps the three extended sweep groups to their corresponding parameters. |
| `SWEEP_RELATED` | Identifies the core group containing earlier steps of each sweep. |
| `SWEEP_PARAMS` | Defines sweep labels, group names, allowed configuration changes, value extraction, scale validation and axis settings. |
| `build_experiments()` | Creates all 23 experiment configurations. |
| `select_experiments()` | Selects experiments by group and assigns the epoch count. |
| `on_sweep()` | Checks whether an experiment differs from the baseline only in the selected sweep parameter. |
| `build_sweep()` | Builds and sorts the configurations belonging to a particular increasing sweep. |

## Baseline configuration

The baseline is defined by:

`BASELINE = "Cơ sở: 2 lớp ẩn (256-128)"`

It uses the default values defined in `ExperimentConfig`.

| Setting | Baseline value |
|---|---|
| Input features | 784 |
| Hidden layers | 2 |
| Hidden-layer sizes | `(256, 128)` |
| Output classes | 10 |
| Activation | ReLU (default) |
| Dropout | 0.2 (default) |
| BatchNorm | Disabled (default) |
| Optimizer | Adam (default) |
| Learning rate | 0.001 (default) |
| Momentum | Default configuration |
| Batch size | 64 (default) |
| Epochs | 30 (default) |
| Seed | 42 (default) |

**Note:** Values marked as defaults are inherited from `ExperimentConfig`, rather than explicitly assigned in `experiments.py`. If the defaults in `config.py` change, the baseline changes accordingly.

## Experiment groups

### Groups 1–7: Core experiments

These groups investigate individual factors relative to the baseline.

| # | Experiment name | Group | Changed configuration |
|---|---|---|---|
| 1 | Cơ sở: 2 lớp ẩn (256-128) | 1. Số lớp ẩn | Baseline |
| 2 | 1 lớp ẩn (256) | 1. Số lớp ẩn | `hidden_sizes=(256,)` |
| 3 | 3 lớp ẩn (256-128-64) | 1. Số lớp ẩn | `hidden_sizes=(256, 128, 64)` |
| 4 | Hẹp: 2 lớp ẩn (64-32) | 2. Độ rộng lớp ẩn | `hidden_sizes=(64, 32)` |
| 5 | Rộng: 2 lớp ẩn (512-256) | 2. Độ rộng lớp ẩn | `hidden_sizes=(512, 256)` |
| 6 | Dropout 0.0 | 3. Dropout | `dropout=0.0` |
| 7 | Dropout 0.5 | 3. Dropout | `dropout=0.5` |
| 8 | Activation Tanh | 4. Hàm kích hoạt | `activation="tanh"` |
| 9 | Activation RELU | 4. Hàm kích hoạt | `activation="relu"` |
| 10 | Adam lr=0.0001 | 5. Bộ tối ưu & learning rate | `lr=1e-4` |
| 11 | Adam lr=0.01 | 5. Bộ tối ưu & learning rate | `lr=1e-2` |
| 12 | SGD lr=0.01 momentum=0.9 | 5. Bộ tối ưu & learning rate | `optimizer="sgd"`, `lr=1e-2`, `momentum=0.9` |
| 13 | Có BatchNorm | 6. BatchNorm | `batch_norm=True` |
| 14 | Batch size 32 | 7. Batch size | `batch_size=32` |
| 15 | Batch size 256 | 7. Batch size | `batch_size=256` |

### Groups 8–10: Increasing sweeps

These groups extend the core experiments by adding intermediate or additional parameter values.

| # | Experiment name | Group | Changed configuration |
|---|---|---|---|
| 16 | 4 lớp ẩn (256-128-64-32) | 8. Số lớp ẩn tăng dần | `hidden_sizes=(256, 128, 64, 32)` |
| 17 | 5 lớp ẩn (256-128-64-32-16) | 8. Số lớp ẩn tăng dần | `hidden_sizes=(256, 128, 64, 32, 16)` |
| 18 | Adam lr=0.0005 | 9. Learning rate tăng dần | `lr=5e-4` |
| 19 | Adam lr=0.005 | 9. Learning rate tăng dần | `lr=5e-3` |
| 20 | Adam lr=0.05 | 9. Learning rate tăng dần | `lr=5e-2` |
| 21 | Batch size 16 | 10. Batch size tăng dần | `batch_size=16` |
| 22 | Batch size 128 | 10. Batch size tăng dần | `batch_size=128` |
| 23 | Batch size 512 | 10. Batch size tăng dần | `batch_size=512` |

## Increasing sweeps (Groups 8–10)

The extended sweeps combine configurations from their corresponding core groups with additional experiments.

This avoids duplicating configurations and ensures that the baseline and intermediate values are included in each sweep.

### Group 8 — Number of hidden layers

**Parameter:** `num_layers`

Hidden-layer sizes are defined by:

`DEPTH_SIZES = (256, 128, 64, 32, 16)`

Each network uses the first \(k\) elements of this tuple, where \(k\) is the number of hidden layers.

| Number of hidden layers | Hidden-layer sizes | Source |
|---|---|---|
| 1 | `(256,)` | Group 1 |
| 2 | `(256, 128)` | Baseline |
| 3 | `(256, 128, 64)` | Group 1 |
| 4 | `(256, 128, 64, 32)` | Group 8 |
| 5 | `(256, 128, 64, 32, 16)` | Group 8 |

This sweep investigates whether increasing network depth improves classification performance or eventually leads to diminishing returns.

The sweep uses a linear x-axis.

### Group 9 — Learning rate

**Parameter:** `lr`

This sweep investigates the effect of progressively increasing the Adam learning rate.

| Learning rate | Source |
|---|---|
| 0.0001 | Group 5 |
| 0.0005 | Group 9 |
| 0.001 | Baseline |
| 0.005 | Group 9 |
| 0.01 | Group 5 |
| 0.05 | Group 9 |

Only Adam configurations are included in this sweep.

The SGD experiment from Group 5 is excluded because changing the optimizer would introduce an additional experimental factor.

The learning-rate axis uses a logarithmic scale because the values span multiple orders of magnitude.

### Group 10 — Batch size

**Parameter:** `batch_size`

This sweep investigates the effect of progressively increasing the number of samples processed per optimization step.

| Batch size | Source |
|---|---|
| 16 | Group 10 |
| 32 | Group 7 |
| 64 | Baseline |
| 128 | Group 10 |
| 256 | Group 7 |
| 512 | Group 10 |

The batch-size axis uses a logarithmic scale.

With a fixed number of epochs, increasing batch size reduces the number of optimization updates per epoch, which can affect convergence, training time and final performance.

## Sweep configuration

The `SWEEP_PARAMS` dictionary defines how each sweep is constructed.

| Parameter | Label | Changed field | Value extraction | Axis scale |
|---|---|---|---|---|
| `num_layers` | Số lớp ẩn | `hidden_sizes` | `len(c.hidden_sizes)` | Linear |
| `lr` | Learning rate (Adam) | `lr` | `c.lr` | Logarithmic |
| `batch_size` | Batch size | `batch_size` | `c.batch_size` | Logarithmic |

Each parameter specification contains:

- `label`: Display name for the sweep.
- `group`: Corresponding extended experiment group.
- `changes`: Set of configuration fields allowed to differ from the baseline.
- `value`: Function that extracts the numerical value used for sorting and plotting.
- `on_scale`: Additional condition determining whether a configuration belongs to the sweep.
- `log`: Indicates whether the x-axis should use a logarithmic scale.

The `on_scale` condition for network depth additionally checks that the hidden-layer sizes follow the predefined `DEPTH_SIZES` sequence.

## Methods

### `build_experiments() -> List[ExperimentConfig]`

Creates and returns all 23 experiment configurations.

The configurations are constructed using `ExperimentConfig`, with each experiment assigned a name, group and optional parameter overrides.

The function returns the experiments in a fixed order, making the experiment definitions consistent across notebooks and execution scripts.

### `select_experiments(epochs=30, groups=None) -> List[ExperimentConfig]`

Selects experiment configurations for execution.

**Procedure:**

1. Call `build_experiments()` to generate all 23 experiments.
2. Assign the specified epoch count to every configuration.
3. If `groups` is provided, retain experiments belonging to the selected groups.
4. Always retain the baseline experiment.
5. If an extended sweep group is selected, automatically include its corresponding core group.

The relationship between sweep groups and their supporting core groups is defined by `SWEEP_RELATED`:

| Selected sweep group | Automatically included group |
|---|---|
| 8. Số lớp ẩn tăng dần | 1. Số lớp ẩn |
| 9. Learning rate tăng dần | 5. Bộ tối ưu & learning rate |
| 10. Batch size tăng dần | 7. Batch size |

**Important:** The baseline is always retained, even when it does not belong to the selected group.

The function does not remove duplicate configurations by comparing their parameter values. It filters using experiment group and baseline name.

### `on_sweep(config, baseline, param) -> bool`

Determines whether an experiment belongs to a particular increasing sweep.

The method uses the corresponding specification from `SWEEP_PARAMS`.

An experiment qualifies when:

1. It satisfies the sweep-specific `on_scale` condition.
2. All configuration fields except `name`, `group` and the selected sweep parameter are identical to the baseline.

Formally:

\[
\forall f \notin \{name, group, p\},
\quad config[f]=baseline[f]
\]

where \(p\) is the configuration field being varied.

This ensures that the sweep compares configurations that differ from the baseline only in the intended experimental factor.

### `build_sweep(param, epochs=30) -> List[ExperimentConfig]`

Builds all configurations belonging to one increasing sweep and sorts them by their corresponding numerical value.

Supported parameters:

- `"num_layers"`
- `"lr"`
- `"batch_size"`

**Procedure:**

1. Call `select_experiments()` to obtain all configurations.
2. Locate the baseline configuration.
3. Use `on_sweep()` to filter configurations that belong to the selected sweep.
4. Sort the resulting configurations using the corresponding `value` function in `SWEEP_PARAMS`.

The returned list contains the baseline and all valid sweep steps in increasing order.

## Usage

### Running the core experiments

```python
from src.experiments import CORE_GROUPS, select_experiments

main_runs = select_experiments(
    epochs=30,
    groups=CORE_GROUPS
)
```

Returns 15 experiments: the baseline and 14 alternative configurations from Groups 1–7.

### Running all experiments

```python
from src.experiments import select_experiments

all_runs = select_experiments(epochs=30)

print(len(all_runs))  # 23
```

### Running a specific increasing sweep

```python
from src.experiments import build_sweep

depth_steps = build_sweep("num_layers")
lr_steps = build_sweep("lr")
batch_steps = build_sweep("batch_size")
```

### Selecting an extended sweep through groups

```python
from src.experiments import select_experiments

lr_sweep = select_experiments(
    epochs=30,
    groups=["9. Learning rate tăng dần"]
)
```

This automatically includes:

- The baseline.
- Group 9: additional Adam learning-rate experiments.
- Group 5: supporting optimizer and learning-rate experiments.

## Adding a new experiment

To add an experiment, insert another `ExperimentConfig` entry into `build_experiments()`.

For example:

```python
E(
    "Activation Tanh",
    "4. Hàm kích hoạt",
    activation="tanh"
)
```

The new experiment will be included in the complete experiment list and can be selected using its group name.

If a new experiment is intended to become part of an increasing sweep, its configuration must also satisfy the corresponding `on_sweep()` conditions.

## Notes

- The module defines 23 experiments across 10 groups.
- The main notebook uses Groups 1–7, while the three dedicated sweep notebooks cover Groups 8–10.
- All experiments inherit unspecified settings from `ExperimentConfig`.
- The baseline is always included when using `select_experiments()` with group filtering.
- Selecting a sweep group automatically includes its related core group.
- `build_sweep()` sorts experiments by the actual numerical value of the selected parameter.
- The logarithmic scale is used for learning rate and batch size, while network depth uses a linear scale.
- The SGD experiment is intentionally excluded from the Adam learning-rate sweep.
- Experiment names and group names are maintained in Vietnamese for consistency with report tables and visualizations.