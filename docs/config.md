# Module `config` — Experiment Configuration

**File:** `src/config.py`  
**Dependencies:** `pandas`, `src.utils` (`SEED`)  
**Used by:** `model`, `trainer`, `runner`, `experiments`

## Purpose

The `config` module defines the configuration of each experiment through the `ExperimentConfig` dataclass.

Each experiment is represented by a single configuration object containing its network architecture and training hyperparameters. This makes experiments easier to create, compare, and manage without duplicating training code.

The configuration provides default values for most hyperparameters. A new experiment can therefore inherit the baseline settings and override only the parameters being investigated.

Two fields, `name` and `group`, are required and must be provided when creating a configuration.

## `ExperimentConfig` (dataclass)

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | Required | Experiment name. Used to identify the experiment in results, tables, and visualizations. |
| `group` | `str` | Required | Experiment group, such as `"3. Dropout"`. Used to organize experiments for comparison. |
| `hidden_sizes` | `Tuple[int, ...]` | `(256, 128)` | Number of neurons in each hidden layer. The tuple length determines the number of hidden layers. |
| `activation` | `str` | `"relu"` | Activation function: `relu`, `tanh`, `gelu`, or `leaky_relu`. |
| `dropout` | `float` | `0.2` | Dropout probability applied in the hidden layers. A value of `0.0` disables dropout. |
| `batch_norm` | `bool` | `False` | Whether to use Batch Normalization in the network. |
| `optimizer` | `str` | `"adam"` | Optimizer type: `adam` or `sgd`. |
| `lr` | `float` | `1e-3` | Learning rate used by the optimizer. |
| `momentum` | `float` | `0.9` | Momentum parameter, used only when the optimizer is SGD. |
| `batch_size` | `int` | `64` | Training batch size. Evaluation batch size is configured separately in the data module. |
| `epochs` | `int` | `30` | Number of training epochs. |
| `seed` | `int` | `SEED` | Random seed imported from `src.utils`. |

### Baseline configuration

When only the required fields are provided, the configuration uses these default hyperparameters:

| Parameter | Baseline value |
|---|---:|
| Hidden layers | 2 |
| Hidden sizes | `(256, 128)` |
| Activation | ReLU |
| Dropout | 0.2 |
| Batch Normalization | Disabled |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Momentum | 0.9 (SGD only) |
| Batch size | 64 |
| Epochs | 30 |
| Seed | `SEED` from `src.utils` |

The input dimension (784) and output dimension (10) are fixed in the `architecture` property rather than being configurable fields.

## Properties

### `architecture`

Returns a string representing the complete network architecture, including the input and output dimensions.

**Example:**

```python
config = ExperimentConfig(
    name="Baseline",
    group="1. Số lớp ẩn"
)

print(config.architecture)
# 784 -> 256 -> 128 -> 10
```

The architecture string is generated dynamically from `hidden_sizes`:

```python
" -> ".join(["784", *map(str, self.hidden_sizes), "10"])
```

### `optimizer_text`

Returns a human-readable description of the optimizer and its settings.

- For Adam: displays the optimizer name and learning rate.
- For SGD: displays the optimizer name, learning rate, and momentum.

**Examples:**

```python
adam_config = ExperimentConfig(
    name="Adam",
    group="5. Bộ tối ưu & learning rate"
)

sgd_config = ExperimentConfig(
    name="SGD",
    group="5. Bộ tối ưu & learning rate",
    optimizer="sgd",
    lr=0.01
)

print(adam_config.optimizer_text)
# Adam (lr=0.001)

print(sgd_config.optimizer_text)
# SGD (lr=0.01, momentum=0.9)
```

Both properties are intended to format configuration information for display in experiment tables, logs, and visualization titles. They do not construct the neural network or optimizer.

## `configs_to_dataframe(configs) -> pandas.DataFrame`

Converts a list of `ExperimentConfig` objects into a Pandas DataFrame.

Each row represents one experiment, allowing the experiment configurations to be reviewed before training begins.

### Output columns

| Column | Source | Description |
|---|---|---|
| `Thí nghiệm` | `c.name` | Experiment name. |
| `Nhóm` | `c.group` | Experiment group. |
| `Kiến trúc` | `c.architecture` | Network architecture. |
| `Activation` | `c.activation` | Activation function. |
| `Dropout` | `c.dropout` | Dropout probability. |
| `BatchNorm` | `c.batch_norm` | Batch Normalization setting. |
| `Optimizer` | `c.optimizer_text` | Optimizer and its settings. |
| `Batch size` | `c.batch_size` | Training batch size. |
| `Epoch` | `c.epochs` | Number of epochs. |

**Implementation:**

```python
def configs_to_dataframe(configs: List[ExperimentConfig]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Thí nghiệm": c.name,
        "Nhóm": c.group,
        "Kiến trúc": c.architecture,
        "Activation": c.activation,
        "Dropout": c.dropout,
        "BatchNorm": c.batch_norm,
        "Optimizer": c.optimizer_text,
        "Batch size": c.batch_size,
        "Epoch": c.epochs,
    } for c in configs])
```

Note that `seed` and `momentum` are not separate DataFrame columns. Momentum is included in the `Optimizer` column only when the optimizer is SGD.

## Example

```python
from src.config import ExperimentConfig, configs_to_dataframe

# Baseline configuration
baseline = ExperimentConfig(
    name="Baseline",
    group="1. Số lớp ẩn"
)

# Experiment with three hidden layers
deeper = ExperimentConfig(
    name="3 hidden layers",
    group="1. Số lớp ẩn",
    hidden_sizes=(256, 128, 64)
)

# Experiment using SGD
sgd = ExperimentConfig(
    name="SGD",
    group="5. Bộ tối ưu & learning rate",
    optimizer="sgd",
    lr=0.01
)

# Display configuration properties
print(deeper.architecture)
# 784 -> 256 -> 128 -> 64 -> 10

print(sgd.optimizer_text)
# SGD (lr=0.01, momentum=0.9)

# Preview experiment configurations
df = configs_to_dataframe([baseline, deeper, sgd])
print(df)
```

## Notes

- `name` and `group` are required when creating an `ExperimentConfig`.
- Experiment names should be unique because the runner may use them as keys when storing results and identifying checkpoints.
- `hidden_sizes` determines both the number of hidden layers and the number of neurons in each layer.
- `momentum` affects optimization only when `optimizer="sgd"`.
- The `seed` default is imported from `src.utils`; its actual value depends on the definition of `SEED`.
- The input dimension (784) and output dimension (10) are fixed in the architecture representation and are not configurable through this dataclass.
- `configs_to_dataframe()` is a preview utility. It does not start training or modify the configurations.