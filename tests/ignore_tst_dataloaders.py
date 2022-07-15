from pathlib import Path

import pytest
from snip.data import PLINKIterableDataset
from xarray import DataArray

from .test_utils import bed_path, data_path, zarr_path  # noqa


class TestPlinkIterableDataset:
    @pytest.fixture(scope="class")
    def bed_dataset(self, bed_path):  # noqa
        ds = PLINKIterableDataset(bed_path)
        assert isinstance(ds._genotype, DataArray)

    @pytest.fixture(scope="class")
    def zarr_dataset(self, zarr_path):  # noqa
        ds = PLINKIterableDataset(zarr_path)
        assert isinstance(ds._genotype, DataArray)

    @pytest.mark.parametrize("from_format,to_format", [("bed", "zarr"), ("zarr","bed")])
    def test_to_disk(
        self,
        from_format: str,
        to_format: str,
        zarr_dataset: PLINKIterableDataset,
        bed_dataset: PLINKIterableDataset,
        data_path: Path, # noqa
    ):
        if from_format == "zarr":
            ds = zarr_dataset
        elif from_format == "bed":
            ds = bed_dataset
        else:
            raise ValueError("Dataset format {from_format} not available.")

        ds.to_disk(data_path / "test." + to_format)
        assert isinstance(ds._genotype, DataArray)
