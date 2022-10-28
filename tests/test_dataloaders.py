from pathlib import Path

import numpy as np
import pytest
import torch
from pandas_plink import get_data_folder
from torch.utils.data import DataLoader
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
        assert ds.genotype.shape[0] != 0
        assert ds.genotype.shape[1] != 0
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

        ds.to_disk(data_path / f"test.{to_format}", mode="w")

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
        assert isinstance(ds.genotype, DataArray)
        assert ds.genotype.shape[0] != 0
        assert ds.genotype.shape[1] != 0

    def test_split_into_strides(self, zarr_dataset: PLINKIterableDataset):
        ds = zarr_dataset
        datasets = ds.split_into_strides(stride=10)

        # test that it is assigned with correct shape
        for dataset in datasets:
            assert dataset.genotype.shape[1] == 10
            assert dataset.genotype.shape[0] == ds.genotype.shape[0]

    def test_set_chromosome(
        self,
        zarr_path: Path,
    ):
        ds = PLINKIterableDataset(zarr_path)
        ds.set_chromosome(chromosome=12)
        assert np.unique(ds.genotype.chrom.compute()) == "12"

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

    def test_train_test_split(self, zarr_dataset: PLINKIterableDataset):
        ds = zarr_dataset
        samples = ds.genotype.shape[0]
        train, test = ds.train_test_split(test_size=0.20)
        assert samples == train.genotype.shape[0] + test.genotype.shape[0]

    @pytest.mark.parametrize(
        "impute_method",
        [("mean"), ("mode")],
    )
    def test_impute_missing_mean(
        self,
        zarr_dataset: PLINKIterableDataset,
        impute_method: str,
    ):
        ds = zarr_dataset
        ds.impute_missing(impute_method)
        coords = ds.genotype.coords

        # test that it is assigned with correct shape
        assert f"{impute_method}_snp" in coords
        assert coords[f"{impute_method}_snp"].shape == coords["snp"].shape

        # test iter with limited buffer size
        ds.buffer_size = 10
        ds.convert_to_tensor = True
        ds.impute_missing_method = impute_method
        X = next(iter(ds))
        assert torch.isnan(X).sum() == 0  # pylint: disable=no-member

    def test_read_and_write_sped(
        self,
        zarr_dataset: PLINKIterableDataset,
    ):
        """Test that we can read and write sped files."""
        # copy
        ds = zarr_dataset
        test_data = Path(__file__).parent / "data" / "test.sped"
        ds.to_disk(test_data, mode="w")
        # read sped
        ds_ = PLINKIterableDataset(path=test_data)

        # test that it is assigned with correct shape
        assert ds_.genotype.shape == ds.genotype.shape
        # check that coords are the same
        coords = [
            "a0",
            "a1",
            "chrom",
            "cm",
            "father",
            "fid",
            "gender",
            "iid",
            "mother",
            "pos",
            "sample",
            "snp",
            "trait",
            "variant",
        ]
        for coord in coords:
            assert coord in ds_.genotype.coords
            assert coord in ds.genotype.coords
            assert np.all(ds_.genotype.coords[coord] == ds.genotype.coords[coord])

        # check values are the same, allow for nan
        assert np.allclose(ds_.genotype.values, ds.genotype.values, equal_nan=True)

    def test_dataloader_integration(
        self,
        zarr_dataset: PLINKIterableDataset,
    ):
        """Test that we can read and write sped files."""
        ds = zarr_dataset

        n = ds.genotype.shape
        dataloader = DataLoader(ds, batch_size=10)

        total = 0
        for batch in dataloader:
            assert isinstance(batch, torch.Tensor)
            assert batch.shape[0] <= 10
            total += batch.shape[0]
            assert batch.shape[1] == n[1]
        assert total == n[0]

        # w. split into strides
        strided_datasets = ds.split_into_strides(stride=9)
        assert len(strided_datasets) == ds.genotype.shape[1] // 9

        total_variants = 0
        for strided_ds in strided_datasets:
            dataloader = DataLoader(strided_ds, batch_size=10, drop_last=False)
            total_samples = 0
            for batch in dataloader:
                assert isinstance(batch, torch.Tensor)
                assert batch.shape[0] <= 10
                assert batch.shape[1] == 9, "batch shape should be 9"
                total_samples += batch.shape[0]
            assert total_samples == strided_ds.genotype.shape[0]
            total_variants += batch.shape[1]
            assert (
                strided_ds.genotype.shape[1] == batch.shape[1]
            ), f"Expected {strided_ds.genotype.shape[1]} but got {batch.shape[1]}"
            assert (
                ds.genotype.shape[1] > total_variants
            ), "Not all variants were loaded."
        assert (
            ds.genotype.shape[1] - total_variants < 9
        ), "All variants except the last should be in a stride"
