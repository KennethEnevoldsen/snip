"""A script for training and evaluating a slided autoencoder."""

import shutil
from argparse import Namespace
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

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


def create_encoder(clf):
    """Create a encoder from a sklearn MLPRegressor."""

    def encoder(X):
        # find the bottleneck in layers
        bottleneck = np.argmin([layer.shape[1] for layer in clf.coefs_])
        # extract activation function
        if clf.activation == "relu":

            def activation(x):
                return np.maximum(0, x)

        elif clf.activation == "identity":

            def activation(x):
                return x

        for weight, bias in zip(
            clf.coefs_[: bottleneck + 1],
            clf.intercepts_[: bottleneck + 1],
        ):
            X = X @ weight + bias
            # activation
            X = activation(X)
        return X

    return encoder


def create_decoder(clf):
    """Create a decoder from a sklearn MLPRegressor."""

    def decoder(X):
        # find the bottleneck in layers
        bottleneck = np.argmin([layer.shape[1] for layer in clf.coefs_])
        # extract activation function
        if clf.activation == "relu":

            def activation(x):
                return np.maximum(0, x)

        elif clf.activation == "identity":

            def activation(x):
                return x

        for weight, bias in zip(
            clf.coefs_[bottleneck + 1 :],
            clf.intercepts_[bottleneck + 1 :],
        ):
            X = X @ weight + bias
            # activation
            X = activation(X)
        return X

    return decoder


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
        limit=cfg.data.limit,
        chromosome=cfg.data.chromosome,
    )
    if not train.is_missing_imputed():
        if cfg.project.verbose:
            msg.info("Missing values are not imputed. Imputing missing values.")
        train.impute_missing()
        # train.update_on_disk()
        # if cfg.project.verbose:
        #     msg.good("Missing values are imputed. Saved missing imputation to disk.")

    train_datasets = train.split_into_strides(stride=cfg.data.stride)

    validation = PLINKIterableDataset(
        cfg.data.validation_path,
        impute_missing=cfg.data.impute_missing,
        limit=cfg.data.limit,
        chromosome=cfg.data.chromosome,
    )
    validation.impute_missing_from(train)  # transfer the missing imputes.
    validation_datasets = validation.split_into_strides(stride=cfg.data.stride)

    if cfg.data.test_path:
        test = PLINKIterableDataset(
            cfg.data.test_path,
            impute_missing=cfg.data.impute_missing,
            limit=cfg.data.limit,
            chromosome=cfg.data.chromosome,
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
) -> tuple:
    """Train a autoencoder.

    Args:
        dataset_splits: A tuple of datasets for training, validation and test.
        n: The index of the dataset split.
        cfg: The configuration.

    Returns:
        tuple: A tuple of the index of the dataset split, the trained autoencoder and
            the evaluation metrics.
    """
    # interim_path = Path(cfg.data.interim_path) / wandb.config.run_name

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
    decoder = create_decoder(model)
    yhat = decoder(c_snps)
    # calculate reconstruction error
    train_reconstruction_error = np.mean((train_X - yhat) ** 2)

    c_train = PLINKIterableDataset.from_array(c_snps, metadata_from=train)
    # c_train.to_disk(interim_path / f"train_{n}.zarr", mode="w")
    # apply val
    validation_X = validation.get_X().compute()
    c_snps = encoder(validation_X)
    c_val = PLINKIterableDataset.from_array(c_snps, metadata_from=validation)
    yhat = decoder(c_snps)
    # calculate reconstruction error
    val_reconstruction_error = np.mean((validation_X - yhat) ** 2)

    # c_val.to_disk(interim_path / f"validation_{n}.zarr", mode="w")
    # apply test
    if test:
        test_X = test.get_X().compute()
        c_snps = encoder(test_X)
        c_test = PLINKIterableDataset.from_array(c_snps, metadata_from=test)
        # c_test.to_disk(interim_path / f"test_{n}.zarr", mode="w")
    else:
        c_test = None
    return c_train, c_val, c_test, train_reconstruction_error, val_reconstruction_error


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
        f"{cfg.project.run_name_prefix}{wandb.run.name}_"
        + f"{datetime.today().strftime('%Y-%m-%d')}"
    )

    interim_path = Path(cfg.data.interim_path) / wandb.config.run_name
    result_path = Path(cfg.data.result_path) / wandb.config.run_name

    datasets = create_datasets(cfg)
    wandb.log({"N models": len(datasets[0])})
    msg.info(f"Training {len(datasets[0])} models")
    dataset_splits = zip(*datasets)

    # train the local MLPs in parallel
    with Pool(cfg.project.n_jobs) as pool:
        results = pool.starmap(
            train,
            zip(dataset_splits, range(len(datasets[0])), repeat(cfg)),
        )

    # log avg, min, max reconstruction errors
    train_reconstruction_errors = [r[3] for r in results]
    val_reconstruction_errors = [r[4] for r in results]
    wandb.log(
        {
            "Mean(training reconstruction error)": np.mean(train_reconstruction_errors),
            "Mean(valilidation reconstruction error)": np.mean(
                val_reconstruction_errors,
            ),
            "Min(training reconstruction error)": np.min(train_reconstruction_errors),
            "Min(validation reconstruction error)": np.min(val_reconstruction_errors),
            "Max(training reconstruction error)": np.max(train_reconstruction_errors),
            "Max(validation reconstruction error)": np.max(val_reconstruction_errors),
        },
    )

    # convert reuslts to dict
    compressions: Dict[str, list] = {"train": [], "validation": [], "test": []}
    for result in results:
        compressions["train"].append(result[0])
        compressions["validation"].append(result[1])
        compressions["test"].append(result[2])

    # merge apply files
    for split in ["train", "validation", "test"]:
        if split == "test" and not cfg.data.test_path:
            continue
        if cfg.project.create_interim:
            compressed_snps_paths = list(interim_path.glob(f"{split}_*.zarr"))
            genotypes = [PLINKIterableDataset(path) for path in compressed_snps_paths]
        else:
            genotypes = compressions[split]

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
    if cfg.project.create_interim:
        shutil.rmtree(interim_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
