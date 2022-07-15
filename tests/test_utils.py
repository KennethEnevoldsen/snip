from pathlib import Path

import pytest
from pandas_plink import get_data_folder
from snip.data import PLINKIterableDataset


@pytest.fixture()
def bed_path():
    return Path(get_data_folder()) / "chr12.bed"


@pytest.fixture()
def data_path() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture()
def zarr_path(data_path):  # noqa
    return Path(__file__).parent / "data" / "test.zarr"


@pytest.fixture()
def zarr_tmp_path():
    return Path("tmp.zarr")


@pytest.fixture(autouse=True)
def run_around_tests(zarr_tmp_path, zarr_path, bed_path):  # noqa
    # Code that will run before your test, for example:
    # if test data is empty create it
    if not zarr_path.exists():
        PLINKIterableDataset(bed_path).to_disk(zarr_path)
    if not zarr_path.exists():
        PLINKIterableDataset(bed_path).to_disk(zarr_path.ex)

    # A test function will be run at this point
    yield
    # clean up test files
    import shutil

    shutil.rmtree(zarr_tmp_path)
