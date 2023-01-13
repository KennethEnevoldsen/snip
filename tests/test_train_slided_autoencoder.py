"""Test for the train_slided_autoencoder.py script."""

from pathlib import Path

from hydra import compose, initialize

from snip.data import PLINKIterableDataset
from snip.train_slided_autoencoder_sklearn import main as sklearn_main

from .utils import data_path  # noqa


def test_train_slided_autoencoder_sklearn(data_path: Path):  # noqa
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
            config_name="default_config_train_slided_autoencoder_sklearn.yaml",
        )

        # change configs
        # data
        cfg.data.train_path = str(train_path)
        cfg.data.validation_path = str(val_path)
        cfg.data.test_path = str(test_path)
        cfg.data.result_path = "tests/data/compressed/"
        cfg.data.interim_path = "tests/data/interim/"  # temporary data path
        # limit to a small number of samples
        cfg.data.limit = 50

        # limit training time
        cfg.model.max_iter = 2

        # dont log to wandb
        cfg.project.wandb_mode = "dryrun"

        # train
        sklearn_main(cfg)
