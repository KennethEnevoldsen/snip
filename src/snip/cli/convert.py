"""CLI for converting between file formats."""

from pathlib import Path
from typing import Optional, Union

import typer
from rich import print  # pylint: disable=redefined-builtin
from typer import Abort, Argument, Option

from snip.data.dataloaders import PLINKIterableDataset

from .setup import app


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
    file_format: Optional[str] = Option(
        None,
        "--file_format",
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
    r"""CLI for converting between file types.

    Intended usage:

    \b
    snip convert path/to/plink_files save/location.zarr --format zarr

    Where ``--format zarr`` can be left out, in which case it will be
    extracted from the save file extension.

    Args:
        input_path (str): Input file or directory.
        output_path (str): Output file or directory.
        file_format (Optional[str]): Format of output file.
        chromosome (Optional[int]): Chromosome to filter the dataset by.
        overwrite (bool): Should it overwrite the dataset.
    """
    output_path_ = Path(output_path)
    if output_path_.exists() and overwrite is False:
        print(
            "[yellow] ⚠ A file already exists[/yellow] at"
            + f" {str(output_path_.resolve())}",
        )
        overwrite = typer.confirm("are you sure you want to overwrite it?")
        if not overwrite:
            raise Abort()

    save_path = convert(
        load_path=input_path,
        save_path=output_path_,
        file_format=file_format,
        chromosome=chromosome,
        overwrite=overwrite,
    )
    print(
        f"[green]✔ Finished [/green]: Converted {input_path} to {save_path.suffix}. "
        + f"Saved at:\n{save_path.resolve()}",
    )


def convert(
    load_path: Union[str, Path],
    save_path: Union[str, Path],
    file_format: Optional[str] = None,
    chromosome: Optional[int] = None,
    overwrite: bool = False,
) -> Path:
    """Convert between file types.

    Args:
        load_path (Union[str, Path]): Where to load the data from.
        save_path (Union[str, Path]): Where to save the data to.
        file_format (Optional[str]): The format of the data. Defaults to None.
        chromosome (Optional[int]): The chromosome to filter by. Defaults to None.
        overwrite (bool): Overwrite existing file at save path? Defaults to False.

    Returns:
        Path: The final output path.
    """
    load_path = Path(load_path)
    save_path = Path(save_path)

    # ensure filepaths exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format is None:
        file_format = save_path.suffix
        file_format = file_format.strip(".")

    mode = None
    if overwrite:
        mode = "w"
    if file_format in ["zarr", "sped", "bed"]:
        ds = PLINKIterableDataset(load_path, chromosome=chromosome, verbose=False)
        ds.to_disk(save_path, mode=mode)
    else:
        raise ValueError(f"Format {file_format} is not supported")

    return save_path
