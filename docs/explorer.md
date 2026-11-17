# Module `explorer` — data exploration

**File:** `src/explorer.py`
**Depends on:** `matplotlib`, `numpy`, `pandas`, `IPython.display`, `src.data` (`FashionMNISTData`)
**Used by:** notebook section 4

## Purpose

`DatasetExplorer` groups every table and plot used to describe the dataset before training. It only reads the raw data stored in a `FashionMNISTData` object and changes nothing.

```python
from src import FashionMNISTData, DatasetExplorer

data = FashionMNISTData("./data")
explorer = DatasetExplorer(data)
```

## Methods

| Method | Output | What it shows |
|---|---|---|
| `overview()` | 2 tables | Number of train/evaluation images, image size, number of channels and classes, raw dtype, raw pixel range (0–255), range after normalisation to `[0, 1]` (before normalizing by mean/std), tensor shape of one image `(1, 28, 28)`; then the label table with English and Vietnamese class names. |
| `plot_distribution()` | 2 bar charts | Number of images per class in the training set and in the evaluation set, with the count on each bar. Shows whether the classes are balanced. |
| `plot_samples(n_per_class=10)` | Image grid, 10 rows | The first `n_per_class` training images of each class, one row per class. |
| `plot_mean_images()` | 2 × 5 grid | Pixel-wise average image of each class over the training set. |
| `plot_pixel_values(index=0)` | 3 panels + 2 printed lines | Training image `index` with a red box on its central 12 × 12 area; that area with the raw value of each pixel written on it; a histogram of all 784 pixel values. Then prints min/max/mean pixel value and the share of background pixels (value 0). |

## How to read the plots

- **Distribution:** Fashion-MNIST is balanced (6,000 / 1,000 images per class), so plain accuracy is a fair metric and macro and weighted averages are almost the same.
- **Sample images:** show the variety inside each class and how small the images are (28 × 28).
- **Mean images:** classes whose mean images look alike, especially *T-shirt/top*, *Pullover*, *Coat* and *Shirt*, are the ones the model is expected to confuse. This is compared later with the confusion matrix and the per-class accuracy heatmap.
- **Pixel values:** show that an image is just a matrix of numbers from 0 (black) to 255 (white) and that a large part of it is background. Because the MLP flattens the image, it sees these 784 numbers without any information about which pixels are neighbours.

## Notes

- Plots use `plt.show()` and tables use `IPython.display.display`, so this class is meant for a notebook. `main.py` does not use it.
- Plot titles and labels are in Vietnamese to match the report.
