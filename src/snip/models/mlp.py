"""Create a simple mlp."""

from typing import Callable, List

import torch
from torch import nn


class MLP(nn.Module):
    """
    Attributes:
        layers (List[int]): List of layer sizes.
        activation (Callable): Activation function. Defaults to nn.functional.relu.
        _layers (nn.ModuleList): List of layers.
    """

    def __init__(self, layers: List[int], activation: Callable = nn.functional.relu):
        """Create a simple mlp.

        Args:
            layers (List[int]): List of layer sizes.
            activation (Callable): Activation function. Defaults to nn.functional.relu.
        """
        super().__init__()
        self.layers = layers
        self.activation = activation

        self._layers = nn.ModuleList()
        for i, layer_size in enumerate(layers[:-1]):
            self._layers.append(nn.Linear(layer_size, layers[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = self.activation(layer(x))
        return x
