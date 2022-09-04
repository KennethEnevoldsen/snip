"""Utilities for models."""
from typing import Callable

import torch
from torch import nn


def fetch_activation_function(activation: str) -> Callable:
    """Fetch the activation function."""
    if activation == "relu":
        return nn.functional.relu
    elif activation == "elu":
        return nn.functional.elu
    elif activation == "selu":
        return nn.functional.selu
    elif activation == "leaky_relu":
        return nn.functional.leaky_relu
    elif activation == "tanh":
        return nn.functional.tanh
    elif activation == "sigmoid":
        return nn.functional.sigmoid
    elif activation == "linear":

        def linear(x):
            return x

        return linear
    else:
        raise ValueError(f"Unknown activation function: {activation}")


def fetch_optimizer(optimizer: str) -> Callable:
    """Fetch the optimizer."""
    optimizers_dict = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "rmsprop": torch.optim.RMSprop,
        "adamw": torch.optim.AdamW,
        "sparseadam": torch.optim.SparseAdam,
        "adamax": torch.optim.Adamax,
        "lbfgs": torch.optim.LBFGS,
        "rprop": torch.optim.Rprop,
    }
    if optimizer not in optimizers_dict:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented.")
    return optimizers_dict[optimizer]
