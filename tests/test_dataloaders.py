from pathlib import Path

import numpy as np
import pytest
import torch
from pandas_plink import get_data_folder
from xarray import DataArray

from snip.data import PLINKIterableDataset

from .utils import data_path  # noqa


class TestPlinkIterableDataset:
    """Unit tests for the PlinkIterableDataset class."""

    @pytest.fixture(scope="class")
    def bed_path(self) -> Path:
        return Path(get_data_folder()) / "chr12.bed"

    @pytest.fixture(scope="class")
    def zarr_path(self, bed_path: Path) -> Path:
        zarr_path = Path(__file__).parent / "data" / "test.zarr"
        if not zarr_path.exists():
            PLINKIterableDataset(bed_path).to_disk(zarr_path)
        return zarr_path

    @pytest.fixture(scope="class")
    def bed_dataset(self, bed_path):  # noqa
        ds = PLINKIterableDataset(bed_path)
        return ds

    @pytest.fixture(scope="class")
    def zarr_dataset(self, zarr_path):  # noqa
        ds = PLINKIterableDataset(zarr_path)
        return ds

    @pytest.mark.parametrize(
        "from_format,to_format",
        [("bed", "zarr"), ("zarr", "bed")],
    )
    def test_to_disk(
        self,
        from_format: str,
        to_format: str,
        zarr_dataset: PLINKIterableDataset,
        bed_dataset: PLINKIterableDataset,
        data_path: Path,  # noqa
    ):
        if from_format == "zarr":
            ds = zarr_dataset
        elif from_format == "bed":
            ds = bed_dataset
        else:
            raise ValueError("Dataset format {from_format} not available.")

        ds.to_disk(data_path / f"test.{to_format}", overwrite=True)

    @pytest.mark.parametrize(
        "from_format",
        ["bed", "zarr"],
    )
    def test_from_disk(
        self,
        from_format: str,
        zarr_dataset: PLINKIterableDataset,
        bed_dataset: PLINKIterableDataset,
    ):
        if from_format == "zarr":
            ds = zarr_dataset
        elif from_format == "bed":
            ds = bed_dataset
        else:
            raise ValueError("Dataset format {from_format} not available.")
        assert isinstance(ds._genotype, DataArray)

    def test_set_chromosome(
        self,
        zarr_dataset: PLINKIterableDataset,
    ):
        ds = zarr_dataset
        ds.set_chromosome(chromosome=12)
        np.unique(ds.genotype.chrom.compute()) == "12"

    def test_tensor_iter(
        self,
        zarr_dataset: PLINKIterableDataset,
    ):
        ds = zarr_dataset
        X = next(iter(ds))
        assert isinstance(X, torch.Tensor)
        assert len(X.shape) == 1
        assert ds.genotype.shape[0] == len(list(iter(ds)))

        # with limited buffer size
        ds.buffer_size = 10
        X = next(iter(ds))
        assert isinstance(X, torch.Tensor)
        assert len(X.shape) == 1
        assert ds.genotype.shape[0] == len(list(iter(ds)))

    def test_batch_iter(
        self,
        zarr_dataset: PLINKIterableDataset,
    ):
        ds = zarr_dataset
        # torch iter
        X = next(ds.batch_iter(2))
        assert isinstance(X, torch.Tensor)
        assert len(X.shape) == 2
        assert X.shape[0] == 2

        # xarray iter
        ds.convert_to_tensor = False
        X = next(ds.batch_iter(2))
        assert isinstance(X, DataArray)
        assert len(X.shape) == 2
        assert X.shape[0] == 2
