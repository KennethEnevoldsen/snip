from pathlib import Path
from typing import Optional, Union

import typer
from rich import print
from typer import Abort, Argument, Option

from snip.data.dataloaders import PLINKIterableDataset

from ._util import app


@app.command("convert")
def convert_cli(
    input_path: str = Argument(
        ...,
        help="Input file or directory",
        exists=True,
    ),
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
    chromosome: Optional[int] = Option(
        None,
        "--chromosome",
        "-c",
        help="chromosome to filter the dataset by.",
    ),
    overwrite: bool = False,
):
    """Intended usage:

    \b
    snip convert path/to/plink_files save/location.zarr --format zarr

    Where ``--format zarr`` can be left out, in which case it will be
    extracted from the save file extension.

    Args:
        input_path (str): Input file or directory.
        output_path (str): Output file or directory.
        format (Optional[str]): Format of output file.
        chromosome (Optional[int]): Chromosome to filter the dataset by.
        overwrite (bool): Should it overwrite the dataset.
    """

    output_path_ = Path(output_path)
    if output_path_.exists() and overwrite is False:
        print(
            f"[yellow] ⚠ A file already exists[/yellow] at {str(output_path_.resolve())}",
        )
        overwrite = typer.confirm("are you sure you want to overwrite it?")
        if not overwrite:
            raise Abort()

    save_path = convert(input_path, output_path_, format, overwrite)
    print(
        f"[green]✔ Finished [/green]: Converted {input_path} to {save_path.suffix}. "
        + f"Saved at:\n{save_path.resolve()}",
    )


def convert(
    load_path: Union[str, Path],
    save_path: Union[str, Path],
    format: Optional[str] = None,
    chromosome: Optional[int] = None,
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
        ds = PLINKIterableDataset(load_path, chromosome=chromosome, verbose=False)
        ds.to_disk(save_path, overwrite=overwrite)
    else:
        raise ValueError(f"Format {format} is not supported")

    return save_path
