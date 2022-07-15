import pytest
from snip.cli.convert import convert
from typer.testing import CliRunner

from .test_utils import bed_path, zarr_tmp_path  # noqa

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


# def test_cli_convert(app, bed_path, zarr_tmp_path):  # noqa
#     result = runner.invoke(app, ["convert", f"{bed_path}", f"{zarr_tmp_path}"])
#     print(bed_path)
#     print(zarr_tmp_path)
#     import os

#     print(os.listdir())
#     assert result.exit_code == 0
#     assert zarr_tmp_path.is_dir()
