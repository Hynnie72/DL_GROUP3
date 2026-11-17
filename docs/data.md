# Module `data` — Loading and Preprocessing Fashion-MNIST

**File:** `src/data.py`  
**Depends on:** `numpy`, `torch`, `torchvision`, `src.utils` (class names)  
**Used by:** `explorer`, `runner`, notebook section 4

## Purpose

`FashionMNISTData` downloads Fashion-MNIST and keeps the dataset in memory in two forms:

- **Raw images and labels:** Used for data exploration, visualization, and displaying predictions.
- **Tensor representations:** Used for training and evaluation.

The module also provides efficient `DataLoader` instances with data augmentation applied to the training set and normalization applied to both training and evaluation sets.

## The dataset

| Item | Value |
|---|---|
| Training images | 60,000 |
| Evaluation images | 10,000 |
| Image size | 28 × 28 |
| Channels | 1 (grayscale) |
| Classes | 10 |
| Images per training class | 6,000 |
| Images per evaluation class | 1,000 |
| Raw pixel type and range | `uint8`, 0–255 |

The first run downloads the dataset into `root` (default `./data/FashionMNIST/raw/`), so an Internet connection is required for the initial download.

## Preprocessing pipeline

The module applies the following preprocessing steps:

1. **Convert images to tensors:** Convert raw pixel values from `uint8` to `float32`, divide by 255, and add the channel dimension.
2. **Data augmentation (training only):** Randomly rotate and translate each training image to improve model generalization.
3. **Normalization:** Normalize both training and evaluation images using the Fashion-MNIST dataset statistics.

### Normalization parameters

| Parameter | Value |
|---|---|
| Mean | `0.2860` |
| Standard deviation | `0.3530` |

Normalization is performed using:

`transforms.Normalize(mean=(0.2860,), std=(0.3530,))`

The normalized pixel values are calculated as:

\[
x_{\text{normalized}}=\frac{x-0.2860}{0.3530}
\]

Unlike the original `[0, 1]` representation, normalized values can be negative or positive.

### Data augmentation

Augmentation is applied only to the training set.

| Transformation | Configuration | Purpose |
|---|---|---|
| Random Rotation | ±5 degrees | Introduces small orientation variations. |
| Random Affine | Translation up to 5% horizontally and vertically | Simulates small positional shifts. |

The augmentation pipeline is implemented using:

`transforms.Compose([RandomRotation(5), RandomAffine(degrees=0, translate=(0.05, 0.05))])`

Each image receives independently sampled random transformations whenever it is retrieved from the training dataset.

This increases the diversity of training samples without modifying the original images stored in memory.

The evaluation set does not receive augmentation, ensuring that evaluation is performed on the original images with consistent preprocessing.

## Classes

### `AugmentedTensorDataset`

A custom dataset class extending `torch.utils.data.TensorDataset`.

It supports batch-level indexing and applies preprocessing to the retrieved images.

**Constructor:**

`AugmentedTensorDataset(*tensors, augment=None, normalize=None)`

| Component | Description |
|---|---|
| `tensors` | Image and label tensors. |
| `augment` | Optional transformation pipeline applied independently to each image. |
| `normalize` | Optional normalization transformation applied to the batch. |

**Method:**

`__getitem__(index)`

The method performs the following operations:

1. Retrieves a batch of images and labels using the provided indices.
2. Applies augmentation independently to each image if an augmentation pipeline is provided.
3. Stacks the transformed images back into a batch.
4. Applies normalization to the image batch if configured.
5. Returns the processed images and their corresponding labels.

This design allows augmentation to be performed directly on tensor batches instead of converting individual images from PIL format.

### `FashionMNISTData`

Responsible for downloading, storing, preprocessing, and providing Fashion-MNIST data.

## Attributes

| Attribute | Shape / type | Description |
|---|---|---|
| `class_names` | `list[str]` | English class names from `utils`. |
| `class_names_vi` | `list[str]` | Vietnamese class names from `utils`. |
| `train_images` | `np.ndarray (60000, 28, 28)` | Original training images. |
| `train_labels` | `np.ndarray (60000,)` | Original training labels. |
| `evaluation_images` | `np.ndarray (10000, 28, 28)` | Original evaluation images. |
| `evaluation_labels` | `np.ndarray (10000,)` | Original evaluation labels. |
| `x_train` | `torch.Tensor (60000, 1, 28, 28)` | Training images converted to float tensors in `[0, 1]`. |
| `y_train` | `torch.Tensor (60000,)` | Training labels with type `int64`. |
| `x_evaluation` | `torch.Tensor (10000, 1, 28, 28)` | Evaluation images converted to float tensors in `[0, 1]`. |
| `y_evaluation` | `torch.Tensor (10000,)` | Evaluation labels with type `int64`. |

**Note:** `x_train` and `x_evaluation` store the original tensor representations before augmentation and normalization. These transformations are applied dynamically when the DataLoaders retrieve batches.

## Methods

### `__init__(root="./data")`

Downloads the Fashion-MNIST training and evaluation datasets if they are not already available.

It stores the original images and labels as NumPy arrays and converts them into PyTorch tensors for efficient in-memory processing.

### `_to_tensor(images) -> torch.Tensor` (static)

Converts raw image arrays into PyTorch tensors.

**Conversion:**

`uint8 (N, 28, 28)` → `float32 (N, 1, 28, 28)`

Operations:
- Convert pixel values to floating-point numbers.
- Divide by 255 to obtain values in `[0, 1]`.
- Add a channel dimension for compatibility with CNN architectures.

### `get_loaders(batch_size, seed) -> (train_loader, evaluation_loader)`

Creates and returns the training and evaluation DataLoaders.

A `torch.Generator` is initialized with the supplied seed to control the training sample order.

| Configuration | Training loader | Evaluation loader |
|---|---|---|
| Dataset | `AugmentedTensorDataset` | `AugmentedTensorDataset` |
| Batch size | `batch_size` from config | 512 |
| Sampling strategy | `RandomSampler` | `SequentialSampler` |
| Shuffling | Randomized | Fixed order |
| Augmentation | Enabled | Disabled |
| Normalization | Enabled | Enabled |
| `drop_last` | `False` | `False` |

Both loaders use `BatchSampler` with `batch_size=None` in the DataLoader because the custom dataset handles batch-level indexing.

The training sampler uses the seeded generator, ensuring reproducible sample ordering when the same seed is used.

The evaluation sampler processes images sequentially, providing consistent evaluation results.

## Why batch-level indexing is used

A conventional DataLoader typically retrieves individual samples and then combines them into batches.

In this implementation, `BatchSampler` generates a list of indices for an entire batch. `AugmentedTensorDataset` receives these indices and retrieves the corresponding tensor batch directly from memory.

This approach provides several advantages:

- Avoids repeated PIL-to-tensor conversions.
- Reduces Python-level overhead during data loading.
- Allows augmentation to be applied independently to each image.
- Keeps the original dataset in memory.
- Supports efficient batch-level normalization.

## Example

```python
from src import FashionMNISTData

data = FashionMNISTData(root="./data")

train_loader, evaluation_loader = data.get_loaders(
    batch_size=64,
    seed=42
)

images, labels = next(iter(train_loader))

print(images.shape, labels.shape)
# torch.Size([64, 1, 28, 28])
# torch.Size([64])

print(len(train_loader), len(evaluation_loader))
# 938 20
```

The training images returned by `train_loader` are randomly augmented and normalized. Evaluation images are only normalized.

## Notes

- There is no separate validation set. The evaluation set is used to monitor model performance after each epoch.
- Data augmentation is applied exclusively to the training set to prevent artificially modifying evaluation samples.
- Normalization is applied consistently to both training and evaluation data.
- Original images remain unchanged in `train_images` and `evaluation_images`.
- Random augmentation is generated dynamically each time a training sample is retrieved. Therefore, the same image can appear differently across epochs.
- The random generator controls sample ordering; it does not independently guarantee identical augmentation results across separate runs.
- Both loaders retain their final incomplete batches because `drop_last=False`.
