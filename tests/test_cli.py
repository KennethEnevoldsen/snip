from pathlib import Path

import pytest
from pandas_plink import get_data_folder, read_plink1_bin  # noqa

from snip.cli.convert import convert


@pytest.fixture()
def bed_path():
    yield Path(get_data_folder()) / "chr12.bed"


@pytest.fixture()
def zarr_path():
    yield Path("tmp.zarr")


def test_convert(bed_path, zarr_path):
    # test convert to zarr
    convert(bed_path, zarr_path)
    assert zarr_path.is_dir()


@pytest.fixture(autouse=True)
def run_around_tests(zarr_path):
    # Code that will run before your test, for example:
    # A test function will be run at this point
    yield
    # clean up test files
    import shutil

    shutil.rmtree(zarr_path)
