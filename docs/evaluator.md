# Module `evaluator` — metrics and per-model plots

**File:** `src/evaluator.py`
**Depends on:** `matplotlib`, `numpy`, `scikit-learn`, `src.utils` (`SEED`)
**Used by:** `runner`

## Purpose

`Evaluator` turns the predictions of one trained model into numbers and figures: overall metrics, per-class accuracy, learning curves, confusion matrices and a sample of correct and wrong predictions. It does not train anything.

## Constructor

```python
Evaluator(class_names: list[str], evaluation_images: np.ndarray)
```

`class_names` labels the axes; `evaluation_images` are the raw evaluation images (`data.evaluation_images`), used to display predictions.

## Metrics

### `compute_metrics(y_true, y_pred) -> dict`

| Key | Definition |
|---|---|
| `Accuracy (%)` | Correct predictions / all predictions × 100. |
| `Macro Precision` | Precision computed for each class, then averaged with equal weight over the 10 classes. |
| `Macro Recall` | Same for recall. |
| `Macro F1-score` | Same for F1 (harmonic mean of precision and recall per class). |

For one class *c*: precision = TP / (TP + FP), recall = TP / (TP + FN). "Macro" means every class counts the same, so a weak class (often *Shirt*) pulls the average down even though it is only 10 % of the data. `zero_division=0` avoids warnings if a class is never predicted.

### `per_class_accuracy(y_true, y_pred) -> np.ndarray (10,)`

Diagonal of the confusion matrix divided by the row sums × 100: the share of each true class that is predicted correctly. This equals the per-class recall in percent.

## Plots

| Method | Figure |
|---|---|
| `plot_learning_curves(history, title)` | Left: train and evaluation loss per epoch. Right: train and evaluation accuracy per epoch, with a dashed vertical line at the epoch with the highest evaluation accuracy. |
| `plot_confusion_matrix(y_true, y_pred, title)` | Two heatmaps side by side: raw counts, and normalised by row (each row sums to 100 %). Rows are true labels, columns are predicted labels. Then prints the **5 most frequent mistakes** (true → predicted, number of images). |
| `plot_predictions(y_true, y_pred, y_prob, title, n_correct=6, n_wrong=6, seed=SEED)` | A grid of randomly chosen evaluation images: `n_correct` correct and `n_wrong` wrong predictions, mixed together. Each title shows the true label, the predicted label and the predicted probability; green = correct, red = wrong. The choice is seeded, so the same images are shown every run. |

## How to read the results

- **Learning curves:** if the training loss keeps falling while the evaluation loss stops falling or rises, the model is starting to overfit. A flat or noisy curve points to a learning rate that is too low or too high.
- **Confusion matrix:** large off-diagonal cells show which classes are mixed up. The row-normalised version is easier to compare between classes.
- **Wrong predictions with high probability** show that the model can be confidently wrong, usually on clothes with similar shapes.

## Example

```python
from src import Evaluator, CLASS_NAMES

evaluator = Evaluator(CLASS_NAMES, data.evaluation_images)
metrics = evaluator.compute_metrics(y_true, y_pred)
evaluator.plot_confusion_matrix(y_true, y_pred, title="Baseline")
```
