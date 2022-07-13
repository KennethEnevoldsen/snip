from pathlib import Path

import pytest
from pandas_plink import get_data_folder
from typer.testing import CliRunner

from snip.cli.convert import convert

runner = CliRunner()


@pytest.fixture()
def bed_path():
    yield Path(get_data_folder()) / "chr12.bed"


@pytest.fixture()
def zarr_path():
    yield Path("tmp.zarr")


@pytest.fixture()
def app():
    from snip.cli._util import app, setup_cli

    setup_cli()
    return app


def test_convert(bed_path, zarr_path):
    # test convert to zarr
    convert(bed_path, zarr_path)
    assert zarr_path.is_dir()


def test_cli_convert(app, bed_path, zarr_path):
    result = runner.invoke(app, ["convert", f"{bed_path}", f"{zarr_path}"])  # noqa
    assert zarr_path.is_dir()


@pytest.fixture(autouse=True)
def run_around_tests(zarr_path):
    # Code that will run before your test, for example:
    # A test function will be run at this point
    yield
    # clean up test files
    import shutil

    shutil.rmtree(zarr_path)
