"""Tests for models/mlp.py."""
import pytest  # noqa
import torch

from snip.models import MLP


def test_MLP():

    model = MLP(layers=[10, 6], input_size=10)

    x = torch.randn(3, 10)
    y = model.forward(x)

    assert y.shape == torch.Size([3, 6])
