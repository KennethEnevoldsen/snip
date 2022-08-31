"""Create a simple mlp."""


from typing import Callable, List, Union

import pytorch_lightning as pl
import torch
from torch import nn

from .utils import fetch_activation_function


class MLP(pl.LightningModule):
    """
    Attributes:
        layers_size (List[int]): List of layer sizes.
        activation (Callable): Activation function. Defaults to nn.functional.relu.
        _layers (nn.ModuleList): List of layers.
    """

    def __init__(
        self,
        layers: List[int],
        input_size: int,
        activation: Union[str, Callable] = nn.functional.relu,
    ):
        """Create a simple mlp.

        Args:
            layers (List[int]): List of layer sizes.
            activation (Callable): Activation function. Defaults to nn.functional.relu.
        """
        super().__init__()
        self.layers_size = [input_size] + layers
        if isinstance(activation, str):
            self.activation = fetch_activation_function(activation)
        else:
            self.activation = activation

        self._layers = nn.ModuleList()
        input_dim = input_size
        for output_dim in layers:
            self._layers.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
            x = self.activation(x)
        return x
