"""CLI for creating test data."""

from pathlib import Path
from typing import Optional

import typer
from pandas_plink import get_data_folder
from rich import print
from typer import Abort, Argument, Option

from snip.data.dataloaders import PLINKIterableDataset

from .setup import app


@app.command("create_test_data")
def create_test_data_cli(
    output_path: str = Argument(
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
    r"""CLI for for creating test data

    Intended usage:

    \b
    snip create_test_data data/test.zarr --format zarr

    Where ``--format zarr`` can be left out, in which case it will be
    extracted from the save file extension.

    Args:
        output_path (str): Output file or directory.
        format (Optional[str]): Format of output file.
        overwrite (bool): Should it overwrite the dataset.
    """
    save_path = Path(output_path)

    # ensure filepaths exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if format is None:
        format = save_path.suffix
    if format[0] != ".":
        format = f".{format}"

    if save_path.exists() and overwrite is False:
        print(
            f"[yellow] ⚠ A file already exists[/yellow] at {str(save_path.resolve())}",
        )
        overwrite = typer.confirm("are you sure you want to overwrite it?")
        if not overwrite:
            raise Abort()

    save_path = save_path.with_suffix(format)

    load_path = Path(get_data_folder()) / "chr12.bed"

    if format == ".zarr":
        ds = PLINKIterableDataset(load_path, verbose=False)
        ds.to_disk(save_path, overwrite=overwrite)
    else:
        raise ValueError(f"Format {format} is not supported")

    print(
        f"[green]✔ Finished [/green]: Test data created at:\n{save_path.resolve()}",
    )
