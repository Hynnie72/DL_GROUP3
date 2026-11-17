"""MLP: a fully connected network built from an ExperimentConfig."""

import torch
import torch.nn as nn

from .config import ExperimentConfig

ACTIVATIONS = {
    "relu": lambda: nn.ReLU(),
    "tanh": lambda: nn.Tanh(),
    "gelu": lambda: nn.GELU(),
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
}


class MLP(nn.Module):
    """Flatten -> [Linear -> BatchNorm1d? -> Activation -> Dropout?] x N -> Linear(10)."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        layers = [nn.Flatten()]
        in_features = 28 * 28
        for hidden in config.hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(ACTIVATIONS[config.activation]())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_features = hidden
        layers.append(nn.Linear(in_features, 10))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
