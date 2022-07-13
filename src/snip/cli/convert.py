from pathlib import Path
from typing import Optional, Union

from typer import Argument, Option

from snip.data.dataloaders import PLINKIterableDataset

from ._util import app


@app.command("convert")
def convert_cli(
    input_path: Union[str, Path] = Argument(
        ...,
        help="Input file or directory",
        exists=True,
    ),
    output_path: Union[str, Path] = Argument(
        ...,
        help="Output file or directory",
        exists=True,
    ),
    format: Optional[str] = Option(
        None,
        "--format",
        "-f",
        help="Format of output file",
    ),
    overwrite: bool = False,
):
    """Intended use snip convert path/to/plink_files save/location.zarr.

    --format zarr.

    Where --format zarr can be left out, in which case it will be
    extracted from the save file extension.
    """
    convert(input_path, output_path, format, overwrite)


def convert(
    load_path: Union[str, Path],
    save_path: Union[str, Path],
    format: Optional[str] = None,
    overwrite: bool = False,
):
    load_path = Path(load_path)
    save_path = Path(save_path)
    # ensure filepaths exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if format is None:
        format = save_path.suffix
    format = format.strip(".")
    if format == "zarr":
        ds = PLINKIterableDataset(load_path)
        ds.to_disk(save_path, overwrite=overwrite)
