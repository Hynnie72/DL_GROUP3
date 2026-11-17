# Module `trainer` — Training, Evaluation and Checkpoint Management

**File:** `src/trainer.py`  
**Depends on:** `os`, `numpy`, `torch`, `torch.nn`, `torch.utils.data`, `src.config`, `src.model`, `src.utils` (`DEVICE`)  
**Used by:** `runner`

## Purpose

`Trainer` is responsible for training an `MLP` model according to its `ExperimentConfig`.

It manages the complete training process, including:

- Building the loss function and optimizer.
- Running training and evaluation epochs.
- Tracking loss and accuracy throughout training.
- Saving the best and latest model checkpoints.
- Loading previously saved checkpoints.
- Generating predictions and class probabilities.

The module focuses on model training and checkpoint management. It does not calculate additional evaluation metrics or generate visualizations; these responsibilities belong to `Evaluator` and `ExperimentRunner`.

## Constructor

```python
Trainer(
    model: MLP,
    config: ExperimentConfig,
    device: torch.device = DEVICE
)
```

The constructor initializes the model, experiment configuration, computation device, loss function, optimizer, training history and checkpoint tracking variables.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `model` | `MLP` | Neural network being trained. |
| `config` | `ExperimentConfig` | Configuration containing hyperparameters and experiment settings. |
| `device` | `torch.device` | Device used for computation (CPU or CUDA). |
| `criterion` | `nn.CrossEntropyLoss` | Loss function for multi-class classification. |
| `optimizer` | `torch.optim.Optimizer` | Optimizer created from the experiment configuration. |
| `history` | `Dict[str, List[float]]` | Training and evaluation metrics recorded after each epoch. |
| `best_loss` | `float` | Lowest evaluation loss achieved so far, initialized to infinity. |
| `best_epoch` | `int` | Epoch corresponding to the lowest evaluation loss. |
| `checkpoint_paths` | `dict` | Paths to the best and last checkpoint files. |

The model is expected to be moved to the selected device before training, for example:

`MLP(config).to(DEVICE)`

### Training history

The `history` dictionary contains four metrics:

| Key | Description |
|---|---|
| `train_loss` | Average training loss for each epoch. |
| `evaluation_loss` | Average evaluation loss for each epoch. |
| `train_accuracy` | Training classification accuracy in percentage. |
| `evaluation_accuracy` | Evaluation classification accuracy in percentage. |

Each list contains one value per completed epoch.

## Methods

### `_build_optimizer() -> torch.optim.Optimizer`

Creates an optimizer based on `config.optimizer`.

| Configuration | Optimizer | Parameters |
|---|---|---|
| `"adam"` | `torch.optim.Adam` | Model parameters, `lr=config.lr` |
| `"sgd"` | `torch.optim.SGD` | Model parameters, `lr=config.lr`, `momentum=config.momentum` |
| Other values | `ValueError` | Invalid optimizer configuration. |

The learning rate and momentum are obtained directly from `ExperimentConfig`.

No learning-rate scheduler or weight decay is configured in this implementation.

### `_run_epoch(loader, train) -> (mean_loss, accuracy_percent)`

Executes one complete pass through the supplied DataLoader.

The `train` argument determines whether the model performs training or evaluation.

**Training procedure:**

1. Set the model to training mode using `model.train(True)`.
2. Enable gradient computation.
3. Move input images and labels to the selected device.
4. Perform a forward pass to obtain model outputs.
5. Calculate classification loss using `CrossEntropyLoss`.
6. Clear previous gradients using `optimizer.zero_grad()`.
7. Calculate gradients using `loss.backward()`.
8. Update model parameters using `optimizer.step()`.

**Evaluation procedure:**

1. Set the model to evaluation mode using `model.train(False)`.
2. Disable gradient computation.
3. Perform forward passes and calculate the loss.
4. Do not update model parameters.

**Metric calculation:**

The average loss is calculated using sample-weighted accumulation:

\[
L_{\text{avg}}=
\frac{\sum_{i=1}^{B}L_i n_i}
{\sum_{i=1}^{B}n_i}
\]

where \(L_i\) is the mean loss of batch \(i\), and \(n_i\) is its number of samples.

Classification accuracy is calculated as:

\[
Accuracy=
\frac{\text{Number of correct predictions}}
{\text{Total number of samples}}\times100\%
\]

Predictions are obtained using `outputs.argmax(dim=1)`.

The method returns:

- `mean_loss`: Average loss across all samples.
- `accuracy_percent`: Classification accuracy in percentage.

### `_save_checkpoint(path, epoch, best_loss) -> None`

Saves the complete training state into a `.pth` checkpoint file.

The checkpoint contains:

| Key | Description |
|---|---|
| `epoch` | Epoch at which the checkpoint was saved. |
| `model_state_dict` | Model parameters and registered buffers. |
| `optimizer_state_dict` | Optimizer state, including optimizer-specific information. |
| `best_loss` | Best evaluation loss achieved so far. |
| `best_epoch` | Epoch corresponding to the best evaluation loss. |
| `history` | Training and evaluation history up to the current epoch. |

The checkpoint is saved using `torch.save()`.

Saving both model and optimizer states allows the training process to be restored more completely.

### `fit(train_loader, evaluation_loader, log_every=5, checkpoint_dir=None) -> history`

Executes the complete training process for `config.epochs` epochs.

The method accepts an optional `checkpoint_dir` to enable automatic checkpoint management.

**Training workflow:**

For each epoch, the following operations are performed:

1. **Training:** Run `_run_epoch(train_loader, train=True)`.
2. **Evaluation:** Run `_run_epoch(evaluation_loader, train=False)`.
3. **History update:** Append the four metrics to the corresponding history lists.
4. **Best checkpoint selection:** Compare the current evaluation loss against `best_loss`.
5. **Last checkpoint update:** Save the latest training state after every epoch.
6. **Logging:** Print progress at epoch 1, the final epoch and every `log_every` epochs.

**Best checkpoint selection**

The best checkpoint is determined by the lowest evaluation loss:

\[
BestLoss=\min_{1\leq e\leq E}EvaluationLoss_e
\]

An epoch is considered the new best when:

`evaluation_loss < self.best_loss`

When this condition is satisfied:

- Update `best_loss`.
- Update `best_epoch`.
- Save the model and optimizer state to `best_checkpoint.pth`.

The best checkpoint is therefore selected based on evaluation loss, not evaluation accuracy.

**Last checkpoint**

Unlike the best checkpoint, the last checkpoint is updated after every epoch, regardless of whether the current evaluation loss improves.

This preserves the state of the most recently completed epoch.

### Checkpoint file structure

When `checkpoint_dir` is provided, the method creates the directory if necessary and manages two files:

| File | Purpose | Update frequency |
|---|---|---|
| `best_checkpoint.pth` | Stores the model state with the lowest evaluation loss. | Only when a new best evaluation loss is achieved. |
| `last_checkpoint.pth` | Stores the latest completed training state. | After every epoch. |

The paths are stored in `checkpoint_paths`:

```python
{
    "best": best_path,
    "last": last_path
}
```

**Example training output:**

```text
[BEST] Epoch 01 | Evaluation Loss: 0.450000

Epoch 01/30 | Train Loss: 0.5000 | Train Acc: 82.00% |
Evaluation Loss: 0.4500 | Evaluation Acc: 84.00%

============================================================
TRAINING COMPLETED
Best Epoch       : 12
Best Eval Loss   : 0.320000
Best Checkpoint  : .../best_checkpoint.pth
Last Checkpoint  : .../last_checkpoint.pth
============================================================
```

The numerical values above are illustrative.

The method returns the complete `history` dictionary.

### `load_checkpoint(path, load_optimizer=False) -> dict`

Loads a previously saved checkpoint and restores the corresponding training state.

The checkpoint is loaded using:

`torch.load(path, map_location=self.device, weights_only=True)`

The method restores:

- Model parameters using `model.load_state_dict()`.
- Best evaluation loss.
- Best epoch.
- Training history.

If `load_optimizer=True`, the optimizer state is also restored using `optimizer.load_state_dict()`.

This allows the trainer to recover the optimizer state when continuing training from a compatible checkpoint.

The loaded checkpoint dictionary is returned to the caller.

**Note:** This implementation uses `weights_only=True`. The checkpoint contents must be compatible with PyTorch's weights-only deserialization restrictions.

### `predict(loader) -> (y_true, y_pred, y_prob)`

Generates predictions for all samples in the supplied DataLoader.

The method uses:

- `@torch.no_grad()` to disable gradient computation.
- `model.eval()` to activate evaluation behaviour.
- `torch.softmax(outputs, dim=1)` to convert logits into class probabilities.

For each batch:

1. Move images to the selected device.
2. Perform a forward pass.
3. Apply Softmax to obtain class probabilities.
4. Determine the predicted class using `argmax`.
5. Collect true labels, predicted labels and probabilities.

The method returns three NumPy arrays:

| Output | Shape | Description |
|---|---|---|
| `y_true` | `(N,)` | Ground-truth class labels. |
| `y_pred` | `(N,)` | Predicted class labels. |
| `y_prob` | `(N, 10)` | Predicted probability distribution across 10 classes. |

These outputs are used by `Evaluator` for calculating classification metrics, generating confusion matrices and analyzing model predictions.

## Important: Training accuracy vs. Evaluation accuracy

Training accuracy and evaluation accuracy are measured under different conditions.

**Training accuracy:**

- Calculated during the training pass.
- Model parameters are updated after each batch.
- Dropout is active when the model contains Dropout layers.
- Batch Normalization uses training behaviour when applicable.

**Evaluation accuracy:**

- Calculated after the training pass of each epoch.
- Model parameters remain unchanged.
- Dropout is disabled.
- Batch Normalization uses its stored running statistics when applicable.

Consequently, training accuracy can be lower than evaluation accuracy.

This is not necessarily a bug. It can occur because training uses regularization such as Dropout, while evaluation measures the model without this training-time regularization.

## Example

```python
from src import (
    ExperimentConfig,
    MLP,
    Trainer,
    FashionMNISTData,
    DEVICE
)

# 1. Configure experiment
cfg = ExperimentConfig(
    name="demo",
    group="demo",
    epochs=3
)

# 2. Load dataset
data = FashionMNISTData("./data")

train_loader, evaluation_loader = data.get_loaders(
    cfg.batch_size,
    cfg.seed
)

# 3. Initialize model and trainer
model = MLP(cfg).to(DEVICE)
trainer = Trainer(model, cfg)

# 4. Train and save checkpoints
history = trainer.fit(
    train_loader,
    evaluation_loader,
    log_every=1,
    checkpoint_dir="./outputs/demo"
)

# 5. Load the best checkpoint
trainer.load_checkpoint(
    trainer.checkpoint_paths["best"]
)

# 6. Generate predictions using the best model
y_true, y_pred, y_prob = trainer.predict(evaluation_loader)
```

## Notes

- Checkpoint saving is optional. If `checkpoint_dir=None`, training runs without saving checkpoint files.
- `best_checkpoint.pth` represents the model with the lowest evaluation loss, while `last_checkpoint.pth` represents the most recently completed epoch.
- The best checkpoint is not necessarily the checkpoint with the highest evaluation accuracy.
- The evaluation set is used after every training epoch for model monitoring and checkpoint selection.
- There is no separate validation dataset in this implementation. Therefore, evaluation-based checkpoint selection should be interpreted as model selection using the evaluation set, rather than an independent final test.
- Calling `load_checkpoint()` restores the recorded history and best-loss information as well as model weights.
- `predict()` does not automatically load the best checkpoint. The caller must explicitly load the desired checkpoint before generating predictions.