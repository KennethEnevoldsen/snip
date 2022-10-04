"""Tests for the command line interface."""
import shutil

import pytest
from typer.testing import CliRunner

from snip.cli.convert import convert

from .utils import bed_path, zarr_tmp_path  # noqa

runner = CliRunner()


@pytest.fixture()
def app():
    # setup app
    from snip.cli.convert import convert_cli  # noqa
    from snip.cli.create_test_data import create_test_data_cli  # noqa
    from snip.cli.setup import app
    from snip.cli.train_test_split import train_test_split_cli  # noqa

    return app


def test_convert(bed_path, zarr_tmp_path):  # noqa
    """Test the convert function."""
    convert(bed_path, zarr_tmp_path)
    assert zarr_tmp_path.is_dir()


def test_cli_convert(app, bed_path, zarr_tmp_path):  # noqa
    """Test the CLI convert command."""
    result = runner.invoke(app, ["convert", f"{bed_path}", f"{zarr_tmp_path}"])

    assert result.exit_code == 0
    assert zarr_tmp_path.is_dir()


@pytest.fixture(autouse=True)
def run_around_tests(zarr_tmp_path):  # noqa
    """Code that will run before tests."""
    # Before tests

    # A test function will be run at this point
    yield
    # After tests
    # clean up test files
    shutil.rmtree(zarr_tmp_path)
