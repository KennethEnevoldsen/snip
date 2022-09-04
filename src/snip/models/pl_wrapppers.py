"""PyTorch lightning wrappers for nn.Modules."""

import time

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import fetch_optimizer


class PlAEWrapper(pl.LightningModule):
    """
    Attributes:
        module (nn.Module): Module to wrap.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer: str,
        learning_rate: float,
        optimizer_params: dict = {},
    ):
        """Create a pl wrapper for a module.

        Args:
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
            optimizer (str): Name of the optimizer to use.
            learning_rate (float): Learning rate.
            optimizer_params (dict): Parameters for the optimizer.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer_name = optimizer.lower()
        self.learning_rate = learning_rate
        self.optimizer_params = optimizer_params

        # model datasets
        self.val_loader = None
        self.train_loader = None

        # rmse loss and metrics
        self.loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        optimizer = fetch_optimizer(self.optimizer_name)

        return optimizer(
            self.parameters(), lr=self.learning_rate, **self.optimizer_params
        )

    def training_step(self, train_batch, batch_idx):
        s = time.time()
        x = train_batch

        x_hat = self.forward(x)

        # calculate metrics
        loss = self.loss(x_hat, x)

        # log metrics
        self.log("Training loss", loss)
        self.log("Training step/sec", time.time() - s)
        return loss

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader
