"""FashionMNISTData: downloads Fashion-MNIST and builds fast in-memory DataLoaders."""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torchvision import datasets, transforms          # <-- thêm transforms

from .utils import CLASS_NAMES, CLASS_NAMES_VI

MEAN, STD = 0.2860, 0.3530                            # <-- thêm


class AugmentedTensorDataset(TensorDataset):          # <-- thêm lớp này
    """TensorDataset: augment từng ảnh trong batch (ngẫu nhiên riêng cho mỗi ảnh), rồi Normalize."""

    def __init__(self, *tensors, augment=None, normalize=None):
        super().__init__(*tensors)
        self.augment = augment
        self.normalize = normalize

    def __getitem__(self, index):
        x, y = super().__getitem__(index)             # index là list các chỉ số của 1 batch
        if self.augment is not None:
            x = torch.stack([self.augment(img) for img in x])
        if self.normalize is not None:
            x = self.normalize(x)
        return x, y


class FashionMNISTData:
    def __init__(self, root: str = "./data"):
        # ... giữ nguyên toàn bộ phần __init__ như cũ ...
        train_set = datasets.FashionMNIST(root=root, train=True, download=True)
        evaluation_set = datasets.FashionMNIST(root=root, train=False, download=True)

        self.class_names = CLASS_NAMES
        self.class_names_vi = CLASS_NAMES_VI

        self.train_images = train_set.data.numpy()
        self.train_labels = train_set.targets.numpy()
        self.evaluation_images = evaluation_set.data.numpy()
        self.evaluation_labels = evaluation_set.targets.numpy()

        self.x_train = self._to_tensor(self.train_images)
        self.y_train = torch.from_numpy(self.train_labels).long()
        self.x_evaluation = self._to_tensor(self.evaluation_images)
        self.y_evaluation = torch.from_numpy(self.evaluation_labels).long()

    @staticmethod
    def _to_tensor(images: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(images).float().div_(255.0).unsqueeze(1)

    def get_loaders(self, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader]:
        generator = torch.Generator()
        generator.manual_seed(seed)

        normalize = transforms.Normalize(mean=(MEAN,), std=(STD,))
        augment = transforms.Compose([
            transforms.RandomRotation(5),                              # xoay ngẫu nhiên ±5 độ
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # dịch tối đa 5% ngang/dọc
        ])

        # Augment chỉ cho tập train; Normalize cho cả train và evaluation
        train_set = AugmentedTensorDataset(self.x_train, self.y_train,
                                           augment=augment, normalize=normalize)
        evaluation_set = AugmentedTensorDataset(self.x_evaluation, self.y_evaluation,
                                                normalize=normalize)

        # ... phần tạo train_loader / evaluation_loader giữ nguyên ...
        train_loader = DataLoader(
            train_set, batch_size=None,
            sampler=BatchSampler(RandomSampler(train_set, generator=generator),
                                 batch_size=batch_size, drop_last=False)
        )
        evaluation_loader = DataLoader(
            evaluation_set, batch_size=None,
            sampler=BatchSampler(SequentialSampler(evaluation_set), batch_size=512, drop_last=False)
        )
        return train_loader, evaluation_loader