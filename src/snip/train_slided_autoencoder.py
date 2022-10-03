"""A script for training and evaluating a slided autoencoder."""

import shutil
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import dask
import hydra
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
from wasabi import msg

from snip.data.dataloaders import PLINKIterableDataset, combine_plinkdatasets
from snip.models import MLP, PlAEWrapper
from snip.utils import flatten_nested_dict

CONFIG_PATH = Path(__file__).parent / "configs"


dask.config.set(scheduler="synchronous")
# speeds up data loading:
# https://discuss.pytorch.org/t/problems-using-dataloader-for-dask-xarray-netcdf-data/108270/3


def create_autoencoder(cfg: Namespace) -> nn.Module:
    """Create a autoencoder."""

    if cfg.model.architecture.lower() == "mlp":
        assert (
            cfg.data.width == cfg.model.input_size
        ), "training data width  must match the input size of the model."
        assert (
            cfg.data.width == cfg.model.decode_layers[-1]
        ), "last layer of the decoder must match input size of the model."

        encoder = MLP(
            layers=cfg.model.encode_layers,
            input_size=cfg.model.input_size,
            activation=cfg.model.activation,
        )
        decoder = MLP(
            layers=cfg.model.decode_layers,
            input_size=cfg.model.encode_layers[-1],
            activation=cfg.model.activation,
        )
        # wrap in PlWrapper
        model = PlAEWrapper(
            encoder=encoder,
            decoder=decoder,
            optimizer=cfg.training.optimizer,
            learning_rate=cfg.training.learning_rate,
        )
    else:
        raise NotImplementedError("Model not implemented")
    return model


def create_datasets(
    cfg: Namespace,
) -> Tuple[
    Sequence[PLINKIterableDataset],
    Sequence[PLINKIterableDataset],
    Sequence[Optional[PLINKIterableDataset]],
]:
    """Create datasets for training, validation and test."""
    train = PLINKIterableDataset(
        cfg.data.train_path,
        impute_missing=cfg.data.impute_missing,
    )
    if not train.is_missing_imputed():
        if cfg.project.verbose:
            msg.info("Missing values are not imputed. Imputing missing values.")
        train.impute_missing()
        train.update_on_disk()
        if cfg.project.verbose:
            msg.good("Missing values are imputed. Saved missing imputation to disk.")

    train_datasets = train.split_into_strides(stride=cfg.data.stride)

    validation = PLINKIterableDataset(
        cfg.data.validation_path,
        impute_missing=cfg.data.impute_missing,
    )
    validation.impute_missing_from(train)  # transfer the missing imputes.
    validation_datasets = validation.split_into_strides(stride=cfg.data.stride)

    if cfg.data.test_path:
        test = PLINKIterableDataset(
            cfg.data.test_path,
            impute_missing=cfg.data.impute_missing,
        )
        test.impute_missing_from(train)
        test_datasets: Sequence[
            Optional[PLINKIterableDataset]
        ] = test.split_into_strides(
            stride=cfg.data.stride,
        )
    else:
        test_datasets = [None] * len(
            train_datasets,
        )

    return train_datasets, validation_datasets, test_datasets


def create_trainer(cfg) -> Trainer:
    wandb_logger = WandbLogger()
    callbacks = [ModelCheckpoint(monitor="Validation loss", mode="min")]
    if cfg.training.patience:
        early_stopping = EarlyStopping(
            "Validation loss",
            patience=cfg.training.patience,
        )
        if callbacks is None:
            callbacks = []
        callbacks.append(early_stopping)

    default_root_dir = Path(cfg.training.default_root_dir)
    weight_save_path = default_root_dir / wandb.config.run_name

    trainer = Trainer(
        logger=wandb_logger,
        log_every_n_steps=cfg.training.log_step,
        val_check_interval=cfg.training.val_check_interval,
        callbacks=callbacks,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        profiler=cfg.training.profiler,
        max_epochs=cfg.training.max_epochs,
        max_steps=cfg.training.max_steps,
        default_root_dir=default_root_dir,
        weights_save_path=weight_save_path,
        precision=cfg.training.precision,
        auto_lr_find=cfg.training.auto_lr_find,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
    )
    return trainer


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="default_config_train_slided_autoencoder",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    """Train and apply model based on config."""

    wandb.init(
        project=cfg.project.name,
        reinit=True,
        config=flatten_nested_dict(cfg, sep="."),
        mode=cfg.project.wandb_mode,
        allow_val_change=True,
    )

    wandb.config.run_name = (
        f"{cfg.project.run_name_prefix}_{wandb.run.name}_"
        + f"{datetime.today().strftime('%Y-%m-%d')}"
    )

    interim_path = Path(cfg.data.interim_path) / wandb.config.run_name
    result_path = Path(cfg.data.result_path) / wandb.config.run_name

    datasets = create_datasets(cfg)
    wandb.log({"N models": len(datasets[0])})
    trainer = create_trainer(cfg)

    # train the local MLPs
    for i, dataset_splits in enumerate(zip(*datasets)):
        model = create_autoencoder(cfg)
        wandb.log({"Model number": i})

        train, validation, test = dataset_splits

        train_loader = DataLoader(
            train,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
        )
        val_loader = DataLoader(
            validation,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
        )
        if test:
            test_loader = DataLoader(
                test,
                batch_size=cfg.data.batch_size,
                num_workers=cfg.data.num_workers,
            )

        # attach loaders to the model to allow for auto lr find
        model.train_loader, model.val_loader = train_loader, val_loader

        # train
        if cfg.training.auto_lr_find:  # and i == 0:
            lr_finder = trainer.tuner.lr_find(model)
            wandb.config.update(
                {"learning_rate": lr_finder.suggestion()},
                allow_val_change=True,
            )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # # apply
        # apply train
        c_snps = model.encode(dataloader=train_loader)
        c_train = PLINKIterableDataset.from_array(c_snps, metadata_from=train)
        c_train.to_disk(interim_path / f"train_{i}.zarr", mode="w")
        # apply val
        c_snps = model.encode(dataloader=val_loader)
        c_val = PLINKIterableDataset.from_array(c_snps, metadata_from=validation)
        c_val.to_disk(interim_path / f"validation_{i}.zarr", mode="w")
        # apply test
        if test:
            c_snps = model.encode(dataloader=test_loader)
            c_test = PLINKIterableDataset.from_array(c_snps, metadata_from=test)
            c_test.to_disk(interim_path / f"test_{i}.zarr", mode="w")

    # merge apply files
    for split in ["train", "validation", "test"]:
        if split == "test" and not cfg.data.test_path:
            continue
        compressed_snps_paths = list(interim_path.glob(f"{split}_*.zarr"))
        genotypes = [PLINKIterableDataset(path) for path in compressed_snps_paths]

        if cfg.project.verbose:
            msg.info(f"Merging {split}, consisting of {len(genotypes)} files")

        assert len(genotypes) > 0, "invalid number of genotypes"
        dataset = combine_plinkdatasets(
            genotypes,
            along_dim="variant",
            rewrite_variants=True,
        )
        dataset.to_disk(result_path / f"c_snps_{split}.zarr", mode="w")

    # remove interim files
    shutil.rmtree(interim_path)

    # evaluate
    # raise NotImplementedError


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
