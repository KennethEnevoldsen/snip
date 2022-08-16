from typing import Optional

import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader

from snip.models.denoising_autoencoder import DenoisingAutoencoder
from snip.models.pl_wrapper import PlOnehotWrapper
from snip.models.snpnet import SNPDecoder, SNPEncoder


def create_model(
    config,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
) -> pl.LightningModule:
    if config.architecture.lower() == "snpnet":
        enc_filters = [int(f * config.filter_factor) for f in [64, 128, 256]]
        dec_filters = [int(f * config.filter_factor) for f in [128, 128, 64, 32]]
        enc_layers = [int(n * config.layers_factor) for n in [1, 3, 4, 6]]
        dec_layers = [int(n * config.layers_factor) for n in [6, 4, 3, 3]]
        enc = SNPEncoder(width=config.width, filters=enc_filters, layers=enc_layers)
        dec = SNPDecoder(width=config.width, filters=dec_filters, layers=dec_layers)
        dae = DenoisingAutoencoder(
            enc,
            dec,
            dropout_p=config.dropout_p,
            fc_layer_size=config.dropout_p,
        )
    else:
        raise NotImplementedError(
            f"Model architecture: {config.architecture} is not implemented",
        )
    if config.snp_encoding.lower() == "one-hot":
        model = PlOnehotWrapper(
            model=dae,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            log_slow=config.log_slow,
        )
    else:
        raise NotImplementedError(
            f"SNP encoding: {config.snp_encoding} is not implemented",
        )
    if config.snp_location_feature:
        raise NotImplementedError("No SNP location features are implemented")

    # log number of parameters
    config.n_of_parameters = sum(p.numel() for p in model.parameters())
    config.n_trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    if config.watch is True:
        wandb.mdl_watch(model, log_freq=config.log_step)
    return model
