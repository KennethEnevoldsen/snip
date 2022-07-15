import pytest
from typer.testing import CliRunner

from snip.cli.convert import convert

from .utils import bed_path, zarr_tmp_path  # noqa

runner = CliRunner()


@pytest.fixture()
def app():
    from snip.cli._util import app  # , setup_cli

    # setup_cli()
    return app


def test_convert(bed_path, zarr_tmp_path):  # noqa
    # test convert to zarr
    convert(bed_path, zarr_tmp_path)
    assert zarr_tmp_path.is_dir()


def test_cli_convert(app, bed_path, zarr_tmp_path):  # noqa
    result = runner.invoke(app, ["convert", f"{bed_path}", f"{zarr_tmp_path}"])
    print(bed_path)
    print(zarr_tmp_path)
    import os

    print(os.listdir())
    assert result.exit_code == 0
    assert zarr_tmp_path.is_dir()


@pytest.fixture(autouse=True)
def run_around_tests(zarr_tmp_path):  # noqa
    # Code that will run before your test, for example:
    # if test data is empty create it

    # A test function will be run at this point
    yield
    # clean up test files
    import shutil

    shutil.rmtree(zarr_tmp_path)
