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
    # test convert to zarr
    convert(bed_path, zarr_tmp_path)
    assert zarr_tmp_path.is_dir()


def test_cli_convert(app, bed_path, zarr_tmp_path):  # noqa
    result = runner.invoke(app, ["convert", f"{bed_path}", f"{zarr_tmp_path}"])

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
