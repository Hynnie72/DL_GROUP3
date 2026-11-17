# Module `model` — the MLP network

**File:** `src/model.py`
**Depends on:** `torch`, `src.config` (`ExperimentConfig`)
**Used by:** `runner`, notebook section 3

## Purpose

`MLP` is a fully connected neural network (multi-layer perceptron) that is built automatically from an `ExperimentConfig`. The same class covers every experiment: changing the number of layers, their width, the activation function, Dropout or BatchNorm only changes the config, not the code.

## Architecture

For an input image of shape `(N, 1, 28, 28)`:

```text
Flatten                              (N, 1, 28, 28) -> (N, 784)
for each h in config.hidden_sizes:
    Linear(in_features, h)
    BatchNorm1d(h)                   only if config.batch_norm
    Activation                       ACTIVATIONS[config.activation]
    Dropout(config.dropout)          only if config.dropout > 0
Linear(last_hidden, 10)              -> (N, 10) logits
```

The output is **raw logits** (no softmax). `nn.CrossEntropyLoss` in `Trainer` applies log-softmax internally, and `Trainer.predict` applies `softmax` when probabilities are needed.

Printed structure of the baseline (784 → 256 → 128 → 10, ReLU, Dropout 0.2):

```text
MLP(
  (network): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=784, out_features=256, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.2, inplace=False)
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): ReLU()
    (6): Dropout(p=0.2, inplace=False)
    (7): Linear(in_features=128, out_features=10, bias=True)
  )
)
```

## `ACTIVATIONS`

A dictionary that maps the `activation` string of a config to a factory that creates a new layer:

| Key | Layer |
|---|---|
| `relu` | `nn.ReLU()` |
| `tanh` | `nn.Tanh()` |
| `gelu` | `nn.GELU()` |
| `leaky_relu` | `nn.LeakyReLU(0.01)` |

Factories (lambdas) are used so each hidden layer gets its own activation object. To add a new activation, add one entry here, e.g. `"elu": lambda: nn.ELU()`.

## `MLP` methods

| Method | Description |
|---|---|
| `__init__(config)` | Builds the layers above into `self.network` (`nn.Sequential`) and keeps `self.config`. |
| `forward(x)` | Returns `self.network(x)`, shape `(N, 10)`. |
| `count_parameters()` | Number of trainable parameters (weights and biases, plus BatchNorm scale/shift when enabled). |

## Parameter counts

| Architecture | Parameters |
|---|---|
| 784 → 64 → 32 → 10 | 52,650 |
| 784 → 256 → 10 | 203,530 |
| 784 → 256 → 128 → 10 (baseline) | 235,146 |
| 784 → 256 → 128 → 10 + BatchNorm | 235,914 |
| 784 → 256 → 128 → 64 → 10 | 242,762 |
| 784 → 256 → 128 → 64 → 32 → 10 | 244,522 |
| 784 → 256 → 128 → 64 → 32 → 16 → 10 | 244,890 |
| 784 → 512 → 256 → 10 | 535,818 |

Dropout, the activation function and the optimizer add no parameters. BatchNorm adds 2 parameters per hidden neuron (2 × (256 + 128) = 768).

## Example

```python
import torch
from src import ExperimentConfig, MLP

model = MLP(ExperimentConfig(name="demo", group="demo", activation="gelu", batch_norm=True))
logits = model(torch.rand(8, 1, 28, 28))
print(logits.shape)               # torch.Size([8, 10])
print(model.count_parameters())   # 235914
```

## Notes

- Because the model is rebuilt from the config, a saved `state_dict` can only be loaded into an `MLP` created from the **same** config (see `ExperimentRunner.load_model`).
- Dropout and BatchNorm behave differently in training and evaluation mode; `Trainer` switches the mode with `model.train(...)` / `model.eval()`.
