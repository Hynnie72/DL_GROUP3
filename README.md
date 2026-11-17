# Lab 1 – Fashion-MNIST Image Classification Using MLP (OOP + Multiple Experiments)

## 1. Introduction

This lab implements a neural network to classify clothing images from the **Fashion-MNIST** dataset.

The codebase follows an object-oriented programming approach, with each module responsible for a specific task. Experiments are defined as configuration objects, making the project modular, extensible, and easy to maintain.

The main notebook runs **15 experiments** (Groups 1–7). Three additional notebooks independently perform progressive parameter sweeps, each focusing on one factor: the number of hidden layers, learning rate, and batch size. Every experiment is trained for exactly **30 epochs**.

- **Course:** Deep Learning
- **Dataset:** Fashion-MNIST (60,000 training images, 10,000 evaluation images, 10 classes)
- **Model:** MLP (Multi-Layer Perceptron)
- **Epochs per experiment:** 30
- **Libraries:** PyTorch, torchvision, scikit-learn, Matplotlib, pandas

## 2. Objectives

- Explore the dataset, including image counts, class distributions, sample images, mean images, and pixel values.
- Implement a configurable `MLP` class supporting different numbers of hidden layers, layer widths, Dropout rates, activation functions, and BatchNorm.
- Run multiple experiments, changing only one factor at a time relative to the baseline.
- Progressively increase the number of hidden layers, learning rate, and batch size to investigate their individual effects.
- Compare experiments using Accuracy, Macro Precision, Macro Recall, Macro F1-score, learning curves, Train–Evaluation gaps, and per-class accuracy.
- Perform detailed analyses of the baseline and best-performing models, including confusion matrices and correct/incorrect predictions.
- Save and reload model weights for each experiment.

## 3. Project Structure

```text
Lab1_FashionMNIST_MLP_OOP/
├── Lab1_FashionMNIST_MLP_OOP.ipynb   # Main notebook: data exploration + 15 experiments (Groups 1–7)
├── Lab1_Sweep_HiddenLayers.ipynb     # Sweep: 1 → 5 hidden layers
├── Lab1_Sweep_LearningRate.ipynb     # Sweep: Adam lr 0.0001 → 0.05
├── Lab1_Sweep_BatchSize.ipynb        # Sweep: batch size 16 → 512
├── main.py                           # Optional: run experiments from the terminal
├── requirements.txt
├── README.md
├── src/                              # Source code, with one responsibility per module
│   ├── __init__.py                   # Re-exports public classes and functions
│   ├── utils.py                      # SEED, DEVICE, CLASS_NAMES, set_seed, slugify
│   ├── config.py                     # ExperimentConfig, configs_to_dataframe
│   ├── model.py                      # MLP, ACTIVATIONS
│   ├── data.py                       # FashionMNISTData, AugmentedTensorDataset
│   ├── explorer.py                   # DatasetExplorer
│   ├── trainer.py                    # Trainer
│   ├── evaluator.py                  # Evaluator
│   ├── runner.py                     # ExperimentResult, ExperimentRunner
│   └── experiments.py                # BASELINE, CORE_GROUPS, SWEEP_PARAMS, build_experiments, build_sweep
└── docs/                             # Documentation for individual modules
    ├── utils.md
    ├── config.md
    ├── model.md
    ├── data.md
    ├── explorer.md
    ├── trainer.md
    ├── evaluator.md
    ├── runner.md
    └── experiments.md
```

The following directories are automatically created when the code runs:

```text
data/           # Fashion-MNIST dataset, downloaded automatically
saved_models/   # .pth checkpoints for each experiment (best and last)
results/        # summary.csv, ranking.csv, sweep_*.csv (only when using main.py)
```

## 4. Modules

| Module | Main Components | Responsibilities | Documentation |
|---|---|---|---|
| `utils` | `SEED`, `DEVICE`, `CLASS_NAMES`, `CLASS_NAMES_VI`, `set_seed`, `slugify` | Shared constants, reproducibility, and file naming utilities. | [docs/utils.md](docs/utils.md) |
| `config` | `ExperimentConfig`, `configs_to_dataframe` | Dataclass containing all hyperparameters for an experiment, including architecture, optimizer, learning rate, batch size, epochs, and seed. | [docs/config.md](docs/config.md) |
| `model` | `MLP`, `ACTIVATIONS` | Builds a neural network from `ExperimentConfig`: Flatten → [Linear → BatchNorm1d (optional) → Activation → Dropout] × N hidden layers → Linear(10). Supports `relu`, `tanh`, `gelu`, and `leaky_relu`. | [docs/model.md](docs/model.md) |
| `data` | `FashionMNISTData`, `AugmentedTensorDataset` | Downloads data, preserves raw images (0–255) for exploration, scales pixel values to [0, 1], normalizes them (mean=0.2860, std=0.3530), applies augmentation (±5° rotation, up to 5% translation) to training data, and creates seeded `DataLoader` instances. | [docs/data.md](docs/data.md) |
| `explorer` | `DatasetExplorer` | Visualizes the dataset through overviews, class distributions, sample images, mean images, and pixel-value analysis. | [docs/explorer.md](docs/explorer.md) |
| `trainer` | `Trainer` | Handles epoch-based training, records training/evaluation Loss and Accuracy, saves and loads checkpoints (best and last), and performs predictions. | [docs/trainer.md](docs/trainer.md) |
| `evaluator` | `Evaluator` | Computes macro metrics and per-class accuracy, plots learning curves and confusion matrices, and visualizes sample predictions. | [docs/evaluator.md](docs/evaluator.md) |
| `runner` | `ExperimentResult`, `ExperimentRunner` | Runs multiple experiments, generates summary tables and comparison plots, analyzes progressive sweeps, performs detailed analyses, saves and loads results, and automatically generates conclusions. | [docs/runner.md](docs/runner.md) |
| `experiments` | `BASELINE`, `CORE_GROUPS`, `SWEEP_PARAMS`, `build_experiments`, `select_experiments`, `build_sweep` | Defines the 23 experiments, sweep configurations, and helper functions for experiment selection. | [docs/experiments.md](docs/experiments.md) |

### Module Dependencies

```text
utils ← config ← model ← trainer ←┐
utils ← data ← explorer            ├─ runner ← notebooks / main.py
utils ← evaluator ←────────────────┤
config ← experiments ←─────────────┘  (also imported directly by notebooks and main.py)
```

To add a new experiment, simply add an `ExperimentConfig(...)` entry to the `build_experiments()` function in `src/experiments.py`.

## 5. Data Preprocessing

- Pixel values are divided by 255 to scale them to [0, 1], then normalized using mean=0.2860 and std=0.3530.
- **Data augmentation** is applied exclusively to the training set:
  - Random rotation within ±5°.
  - Random horizontal and vertical translation of up to 5%.
- **Evaluation set:** No augmentation is applied; only normalization is performed. The batch size is fixed at 512, and shuffling is disabled.

## 6. Experiments

**Baseline configuration:**

- Architecture: 784 → 256 → 128 → 10
- Activation function: ReLU
- Dropout: 0.2 after each hidden layer
- Optimizer: Adam (lr=0.001)
- Batch size: 64

| Group | Variable | Experiments |
|---|---|---|
| 1. Number of Hidden Layers | Number of hidden layers | 1 layer (256) • **2 layers (256-128) – baseline** • 3 layers (256-128-64) |
| 2. Hidden Layer Width | Number of neurons per layer | 64-32 • 512-256 |
| 3. Dropout | Dropout rate | 0.0 • 0.5 |
| 4. Activation Function | Activation function | Tanh • ReLU |
| 5. Optimizer & Learning Rate | Optimizer and learning rate | Adam 0.0001 • Adam 0.01 • SGD 0.01 + momentum 0.9 |
| 6. BatchNorm | Batch normalization | With BatchNorm |
| 7. Batch Size | Batch size | 32 • 256 |
| 8. Progressive Hidden Layers | Number of hidden layers | 4 layers (256-128-64-32) • 5 layers (256-128-64-32-16) |
| 9. Progressive Learning Rate | Adam learning rate | 0.0005 • 0.005 • 0.05 |
| 10. Progressive Batch Size | Batch size | 16 • 128 • 512 |

A total of **23 experiments**, including the baseline, are conducted:

- Groups 1–7: 15 experiments executed by the main notebook.
- Groups 8–10: Additional experiments performed through progressive parameter sweeps.

All experiments use random seed 42.

### Progressive Parameter Sweeps (Groups 8–10)

Each sweep notebook independently trains all configurations in its respective sequence and can be executed separately.

| Notebook | Sweep Parameter | Configurations (bold = baseline) |
|---|---|---|
| `Lab1_Sweep_HiddenLayers.ipynb` | Number of hidden layers | 1 → **2** → 3 → 4 → 5 |
| `Lab1_Sweep_LearningRate.ipynb` | Learning rate (Adam) | 0.0001 → 0.0005 → **0.001** → 0.005 → 0.01 → 0.05 |
| `Lab1_Sweep_BatchSize.ipynb` | Batch size | 16 → 32 → **64** → 128 → 256 → 512 |

## 7. Common Training Configuration

- **Epochs:** 30 for every experiment.
- **Loss function:** CrossEntropyLoss.
- **Checkpoints:** Each experiment saves two files:
  - `best_checkpoint.pth`: The checkpoint from the epoch with the lowest evaluation loss.
  - `last_checkpoint.pth`: The checkpoint from the final epoch.
- Metrics and predictions are computed using the best checkpoint.
- **Device:** GPU if available; otherwise, CPU.

After each epoch, the code records:

- Training Loss
- Training Accuracy
- Evaluation Loss
- Evaluation Accuracy

Training Accuracy is measured during the training process, while Dropout is enabled.

## 8. Notebook Structure

### Main Notebook (`Lab1_FashionMNIST_MLP_OOP.ipynb`)

1. Import libraries, configure device, and set random seed.
2. Define `ExperimentConfig`.
3. Implement the `MLP` model.
4. Load data using `FashionMNISTData`, initialize `DatasetExplorer`, and explore the dataset (overview, class distribution, sample images, mean images, and pixel values).
5. Initialize `Trainer` and `Evaluator`.
6. Define `ExperimentResult` and `ExperimentRunner`.
7. Define the experiment configurations.
8. Run all experiments.
9. Compare experiments within each group using learning curves.
10. Generate summary and ranking tables.
11. Perform overall comparisons using bar charts and per-class accuracy heatmaps.
12. Perform detailed analyses of the baseline and best-performing models.
13. Save and reload trained models.
14. Generate conclusions based on actual experimental results and provide guidance for writing the discussion.

### Sweep Notebooks (`Lab1_Sweep_*.ipynb`)

1. Import libraries, configure device, and set random seed.
2. Define experiment configurations using `build_sweep(...)`.
3. Explore the dataset overview.
4. Execute all configurations in the sweep.
5. Generate result tables ordered by parameter values.
6. Plot parameter trends, including Accuracy and Train–Evaluation gaps, as well as evaluation curves for each configuration.
7. Display learning curves for every configuration.
8. Perform overall comparisons using bar charts and heatmaps.
9. Analyze the best-performing configuration (and the baseline).
10. Save and reload trained models.
11. Generate conclusions and provide guidance for discussing the corresponding parameter.

## 9. Evaluation Metrics

All metrics are computed on the complete evaluation set of 10,000 images using the **best checkpoint**.

- **Accuracy:** Number of correctly classified images divided by the total number of images.
- **Macro Precision / Macro Recall / Macro F1-score:** Calculated independently for each class and then averaged equally across all 10 classes.
- **Train–Evaluation Gap:** Training Accuracy − Evaluation Accuracy. A larger positive gap may indicate a higher risk of overfitting.
- **Per-Class Accuracy:** The proportion of images belonging to each class that are correctly classified.
- **Confusion Matrix:** Displayed using both raw counts and row-normalized percentages.

The `summary()` table displays results from both the **best checkpoint** and the **last checkpoint** side by side for comparison.

## 10. Installation

Python 3.9 or later is required, along with Jupyter Notebook, JupyterLab, or VS Code.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

The notebooks can also run on Google Colab. Upload the entire project directory and ensure that the `src/` directory is located at the same level as the notebooks.

## 11. How to Run

### Notebook (Recommended)

1. Extract the project and open `Lab1_FashionMNIST_MLP_OOP.ipynb` using Jupyter, VS Code, or Google Colab. Keep the notebook in the same directory as `src/`.
2. Execute the cells sequentially from top to bottom. An internet connection is required on the first run to download Fashion-MNIST.
3. Section 7 contains three configurable variables:
   - `EPOCHS = 30`: Number of epochs per experiment (30 epochs are required by the assignment).
   - `SELECTED_GROUPS = CORE_GROUPS`: Runs all 15 experiments from Groups 1–7. To perform a quick test, use an example such as `["1. Number of Hidden Layers", "3. Dropout"]` (the baseline is always included).
   - `LOG_EVERY = 5`: Prints progress every 5 epochs. Set to `1` to display results after every epoch.
4. Wait for Section 8 to finish, then review the summary tables, comparison plots, and detailed analyses.
5. To inspect an individual experiment in more detail, call `runner.show_detail("experiment name")`.
6. Open the sweep notebooks and follow the same procedure. Their `EPOCHS` and `LOG_EVERY` variables are located in Section 2. Each sweep notebook runs independently of the others and of the main notebook.

### Terminal (Optional)

`main.py` executes experiments without generating plots and saves result tables to the `results/` directory.

```bash
python main.py                                      # All 23 experiments, 30 epochs each
python main.py --epochs 2                           # Quick test run
python main.py --groups "1. Number of Hidden Layers" "3. Dropout"  # Selected groups only
```

### Estimated Runtime

On a single-threaded CPU, one baseline epoch takes approximately 4–5 seconds. Therefore:

- Main notebook (15 experiments): approximately 30–40 minutes.
- Each sweep notebook (5–6 configurations): approximately 10–20 minutes.
- The batch size sweep is the slowest because **batch size 16** results in 3,750 updates per epoch.

Actual execution time depends on the hardware and may be shorter on multi-core CPUs or Google Colab.

It is recommended to start with a smaller `EPOCHS` value or a limited `SELECTED_GROUPS` selection to verify the setup before running the complete experiments.

## 12. Saved Model Checkpoints

Each experiment saves two checkpoints in a separate subdirectory under `saved_models/`. Directory names are generated using ASCII-compatible versions of the experiment names.

Example:

```text
saved_models/
├── co_so_2_lop_an_256_128/
│   ├── best_checkpoint.pth
│   └── last_checkpoint.pth
├── dropout_0_5/
│   ├── best_checkpoint.pth
│   └── last_checkpoint.pth
└── co_batchnorm/
    ├── best_checkpoint.pth
    └── last_checkpoint.pth
```

Each `.pth` file stores the following information:

- `model_state_dict`
- `optimizer_state_dict`
- `epoch`
- `best_loss`
- `best_epoch`
- `history`

## 13. Notes for Experimental Discussion

- All numerical results (Accuracy, Macro Precision, Macro Recall, Macro F1-score, and confusion matrices) must come from your actual experiment runs. Do not use illustrative or fabricated values.
- Each experiment is executed once with a single random seed. Therefore, very small differences between experiments may be caused by randomness.
- There is no separate validation set. The evaluation set is used to monitor model performance across epochs. The main conclusions are based on the model at the **best checkpoint**, corresponding to the epoch with the lowest evaluation loss.
- When interpreting learning curves, examine training and evaluation performance together. If training performance continues to improve while evaluation performance stagnates or deteriorates, the model may be starting to overfit.
- Since Training Accuracy is measured with Dropout enabled, it may be lower than Evaluation Accuracy. Therefore, a negative Train–Evaluation gap is normal and does not necessarily indicate a problem.

---

## Summary

This lab covers the complete experimental workflow for Fashion-MNIST image classification using MLP models: dataset exploration, object-oriented model implementation, systematic experimentation, performance evaluation, visualization, model comparison, and checkpoint management.

The modular design makes it straightforward to extend the project with new architectures, hyperparameters, and experimental configurations.