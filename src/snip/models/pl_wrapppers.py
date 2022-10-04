"""PyTorch lightning wrappers for nn.Modules."""

import time
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import fetch_optimizer


class PlAEWrapper(pl.LightningModule):
    """
    Attributes:
        encoder (nn.Module): Encoder module.
        decoder (nn.Module): Decoder module.
        optimizer_name (str): Name of the optimizer to use.
        learning_rate (float): Learning rate.
        optimizer_params (Optional[dict], optional): Optimizer parameters.
        val_loader (DataLoader): Validation dataloader.
        train_loader (DataLoader): Training dataloader.
        loss (nn.Module): Loss function.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: str,
        learning_rate: float,
        optimizer_params: Optional[dict] = None,
    ):
        """Create a pl wrapper for a module.

        Args:
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
            optimizer (str): Name of the optimizer to use.
            learning_rate (float): Learning rate.
            optimizer_params (Optional[dict], optional): Optimizer parameters. Defaults
                to None.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer_name = optimizer.lower()
        self.learning_rate = learning_rate
        self.optimizer_params = optimizer_params or {}

        # model datasets
        self.val_loader = None
        self.train_loader = None

        # rmse loss and metrics
        self.loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.decoder(self.encoder(x))

    def encode(self, dataloader: DataLoader) -> torch.Tensor:
        """Encode the data in the dataloader."""
        self.encoder.eval()
        with torch.no_grad():
            x = torch.cat([d for d in dataloader])
            compressed = self.encoder(x)
        self.encoder.train()
        return compressed

    def configure_optimizers(self):
        """Configure the optimizer."""
        optimizer = fetch_optimizer(self.optimizer_name)

        return optimizer(
            self.parameters(), lr=self.learning_rate, **self.optimizer_params
        )

    def training_step(
        self,
        train_batch,
        batch_idx,  # noqa E501
    ):
        """A single training step."""
        s = time.time()
        x = train_batch

        x_hat = self.forward(x)

        # calculate metrics
        loss = self.loss(x_hat, x)

        # check loss is not nan
        if torch.isnan(loss):
            raise ValueError("Loss is nan")

        # log metrics
        self.log("Training loss", loss)
        self.log("Training step/sec", time.time() - s)
        return loss

    def validation_step(
        self,
        val_batch,
        batch_idx,  # noqa E501
    ):
        """A single validation step."""
        s = time.time()
        x = val_batch

        x_hat = self.forward(x)

        # calculate metrics
        loss = self.loss(x_hat, x)

        # check loss is not nan
        if torch.isnan(loss):
            raise ValueError("Loss is nan")

        # log metrics
        self.log("Validation loss", loss)
        self.log("Validation step/sec", time.time() - s)

    def train_dataloader(self) -> DataLoader:
        """Extract the train dataloader."""
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Extract the val dataloader."""
        return self.val_loader
