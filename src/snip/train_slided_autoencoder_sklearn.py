"""A script for training and evaluating a slided autoencoder."""

from argparse import Namespace
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence, Tuple

import dask
import hydra
import numpy as np
import wandb
from omegaconf import DictConfig
from sklearn.neural_network import MLPRegressor
from wasabi import msg

from snip.data.dataloaders import PLINKIterableDataset, combine_plinkdatasets
from snip.utils import flatten_nested_dict

CONFIG_PATH = Path(__file__).parent / "configs"


# speeds up data loading
# https://discuss.pytorch.org/t/problems-using-dataloader-for-dask-xarray-netcdf-data/108270/3
dask.config.set(scheduler="synchronous")


def create_encoder(clf):
    """Create a encoder from a sklearn MLPRegressor."""

    def encoder(X):
        # find the bottleneck in layers
        bottleneck = np.argmin([layer.shape[1] for layer in clf.coefs_])
        # extract activation function
        if clf.activation == "relu":
            activation = lambda x: np.maximum(0, x)
        elif clf.activation == "identity":
            activation = lambda x: x

        for weight, bias in zip(
            clf.coefs_[: bottleneck + 1],
            clf.intercepts_[: bottleneck + 1],
        ):
            X = X @ weight + bias
            # activation
            X = activation(X)
        return X

    return encoder


def create_autoencoder(cfg: Namespace) -> MLPRegressor:
    """Create a autoencoder."""

    if cfg.model.architecture == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=cfg.model.hidden_layer_sizes,
            activation=cfg.model.activation,
            solver=cfg.model.solver,
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


def train(
    dataset_splits: Tuple[PLINKIterableDataset, ...],
    n: int,
    cfg: Namespace,
):
    """Train a autoencoder.

    Args:
        dataset_splits: A tuple of datasets for training, validation and test.
        n: The index of the dataset split.
        cfg: The configuration.
    """
    interim_path = Path(cfg.data.interim_path)

    model = create_autoencoder(cfg)
    msg.info(f"Training autoencoder {n}")

    train, validation, test = dataset_splits

    train_X = train.get_X().compute()
    # check that it does not contain data
    assert np.isnan(train_X).sum() == 0

    model.fit(train_X, train_X)

    # # apply
    encoder = create_encoder(model)
    c_snps = encoder(train_X)
    c_train = PLINKIterableDataset.from_array(c_snps, metadata_from=train)
    c_train.to_disk(interim_path / f"train_{n}.zarr", mode="w")
    # apply val
    validation_X = validation.get_X().compute()
    c_snps = encoder(validation_X)
    c_val = PLINKIterableDataset.from_array(c_snps, metadata_from=validation)
    c_val.to_disk(interim_path / f"validation_{n}.zarr", mode="w")
    # apply test
    if test:
        test_X = test.get_X().compute()
        c_snps = encoder(test_X)
        c_test = PLINKIterableDataset.from_array(c_snps, metadata_from=test)
        c_test.to_disk(interim_path / f"test_{n}.zarr", mode="w")


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="default_config_train_slided_autoencoder_sklearn",
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

    interim_path = Path(cfg.data.interim_path)
    result_path = Path(cfg.data.result_path)

    datasets = create_datasets(cfg)
    wandb.log({"N models": len(datasets[0])})
    msg.info(f"Training {len(datasets[0])} models")
    dataset_splits = zip(*datasets)

    # train the local MLPs in parallel
    with Pool(cfg.project.n_jobs) as pool:
        pool.starmap(
            train,
            zip(dataset_splits, range(len(datasets[0])), repeat(cfg)),
        )

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

    # evaluate
    # raise NotImplementedError


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
