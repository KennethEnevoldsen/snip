from pathlib import Path

import pytest
from pandas_plink import get_data_folder


@pytest.fixture()
def bed_path():
    return Path(get_data_folder()) / "chr12.bed"


@pytest.fixture()
def data_path() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture()
def zarr_path(data_path):  # noqa
    return data_path / "test.zarr"


@pytest.fixture()
def zarr_tmp_path():
    return Path("tmp.zarr")
