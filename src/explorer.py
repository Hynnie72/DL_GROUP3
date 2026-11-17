"""DatasetExplorer: tables and plots that describe the Fashion-MNIST dataset."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from .data import FashionMNISTData


class DatasetExplorer:
    def __init__(self, data: FashionMNISTData):
        self.data = data

    def overview(self) -> None:
        """Show dataset size, image shape, pixel range and the label table."""
        d = self.data
        sample = d.x_train[0]
        info = pd.DataFrame({
            "Thông tin": [
                "Số ảnh huấn luyện", "Số ảnh kiểm tra", "Kích thước mỗi ảnh (pixel)",
                "Số kênh màu", "Số lớp", "Kiểu dữ liệu gốc", "Khoảng giá trị pixel gốc",
                "Khoảng giá trị sau khi chuẩn hóa", "Kích thước tensor một ảnh"
            ],
            "Giá trị": [
                f"{len(d.train_images):,}", f"{len(d.evaluation_images):,}", "28 x 28",
                "1 (ảnh xám)", len(d.class_names), str(d.train_images.dtype),
                f"{d.train_images.min()} - {d.train_images.max()}",
                f"{sample.min().item():.1f} - {sample.max().item():.1f}",
                str(tuple(sample.shape))
            ]
        })
        display(info)
        display(pd.DataFrame({
            "Nhãn": range(10),
            "Tên lớp (English)": d.class_names,
            "Tên lớp (Tiếng Việt)": d.class_names_vi
        }))

    def plot_distribution(self) -> None:
        """Bar charts of the number of images per class (train and evaluation)."""
        d = self.data
        train_counts = np.bincount(d.train_labels, minlength=10)
        evaluation_counts = np.bincount(d.evaluation_labels, minlength=10)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for ax, counts, title, color in [
            (axes[0], train_counts, f"Tập huấn luyện ({len(d.train_labels):,} ảnh)".replace(",", "."), "tab:blue"),
            (axes[1], evaluation_counts, f"Tập đánh giá ({len(d.evaluation_labels):,} ảnh)".replace(",", "."), "tab:orange"),
        ]:
            bars = ax.bar(range(10), counts, color=color)
            ax.set_xticks(range(10))
            ax.set_xticklabels(d.class_names, rotation=45, ha="right")
            ax.set_ylabel("Số ảnh")
            ax.set_title(title)
            ax.set_ylim(0, counts.max() * 1.12)
            ax.grid(axis="y", alpha=0.25)
            for bar, c in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, c + counts.max() * 0.01,
                        f"{c:,}", ha="center", va="bottom", fontsize=9)
        fig.suptitle("Phân bố số lượng ảnh theo lớp", fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def plot_samples(self, n_per_class: int = 10) -> None:
        """Grid of the first n_per_class training images of each class (one row per class)."""
        d = self.data
        fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class * 1.5, 10 * 1.6))
        for cls in range(10):
            idx = np.where(d.train_labels == cls)[0][:n_per_class]
            for j, i in enumerate(idx):
                ax = axes[cls, j]
                ax.imshow(d.train_images[i], cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(f"{cls}: {d.class_names[cls]}", rotation=0,
                                  ha="right", va="center", fontsize=11, labelpad=10)
        fig.suptitle("Ảnh mẫu của 10 lớp trong Fashion-MNIST (mỗi hàng một lớp)",
                     fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

    def plot_mean_images(self) -> None:
        """Pixel-wise mean image of every class in the training set."""
        d = self.data
        fig, axes = plt.subplots(2, 5, figsize=(14, 6.5))
        for cls, ax in enumerate(axes.flat):
            ax.imshow(d.train_images[d.train_labels == cls].mean(axis=0), cmap="gray")
            ax.set_title(f"{cls}: {d.class_names[cls]}", fontsize=11)
            ax.axis("off")
        fig.suptitle("Ảnh trung bình của từng lớp (tập huấn luyện)", fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def plot_pixel_values(self, index: int = 0) -> None:
        """One training image, the raw values of its central 12x12 crop and its pixel histogram."""
        d = self.data
        img = d.train_images[index]
        crop = img[8:20, 8:20]

        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
        im0 = axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title(f"Ảnh gốc 28x28 - {d.class_names[d.train_labels[index]]}")
        axes[0].add_patch(plt.Rectangle((7.5, 7.5), 12, 12, fill=False, edgecolor="red", linewidth=2))
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Giá trị pixel")

        axes[1].imshow(crop, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Giá trị pixel vùng đỏ (12x12)")
        for r in range(crop.shape[0]):
            for c in range(crop.shape[1]):
                v = crop[r, c]
                axes[1].text(c, r, str(v), ha="center", va="center", fontsize=8,
                             color="white" if v < 128 else "black")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        axes[2].hist(img.ravel(), bins=32, color="tab:blue", edgecolor="white")
        axes[2].set_title("Phân bố giá trị pixel của ảnh")
        axes[2].set_xlabel("Giá trị pixel (0-255)")
        axes[2].set_ylabel("Số pixel")
        axes[2].grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

        print(f"Pixel nhỏ nhất: {img.min()}, lớn nhất: {img.max()}, trung bình: {img.mean():.2f}")
        print(f"Tỉ lệ pixel nền (giá trị 0): {(img == 0).mean() * 100:.1f}%")
