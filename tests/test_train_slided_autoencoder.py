"""Test for the train_slided_autoencoder.py script."""
import shutil
from pathlib import Path

from hydra import compose, initialize

from snip.data import PLINKIterableDataset
from snip.train_slided_autoencoder import main

from .utils import data_path  # noqa


def test_train_slided_autoencoder(data_path: Path):  # noqa
    """test whether the autoencoder trains as intended."""

    # create train and test set
    ds = PLINKIterableDataset(data_path / "human.bed")
    _train, test = ds.train_test_split(test_size=0.1)
    train, val = _train.train_test_split(test_size=0.1)

    train_path = data_path / "sample_train.zarr"
    val_path = data_path / "sample_validation.zarr"
    test_path = data_path / "sample_test.zarr"

    train.to_disk(train_path, mode="w")
    val.to_disk(val_path, mode="w")
    test.to_disk(test_path, mode="w")

    with initialize(version_base=None, config_path="../src/snip/configs/"):
        cfg = compose(
            config_name="test_config_train_slided_autoencoder.yaml",
        )
        main(cfg)

    # Delete train and test data
    shutil.rmtree(train_path)
    shutil.rmtree(val_path)
    shutil.rmtree(test_path)
    shutil.rmtree(data_path / "interim")
