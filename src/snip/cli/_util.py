from pathlib import Path
from typing import Optional

import typer
from rich import print
from typer import Abort, Argument, Option

from .convert import convert

HELP = "Snip Command-line Interface"
NAME = "snip"

app = typer.Typer(name=NAME, help=HELP)


@app.command("delete")
def delete(user: str):
    # THIS is only as otherwise convert is no longer a command.
    typer.echo(f"Deleting user: {user}")


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
    overwrite: bool = False,
):
    """Intended usage:

    .. code::

        snip convert path/to/plink_files save/location.zarr --format zarr

    Where ``--format zarr`` can be left out, in which case it will be
    extracted from the save file extension.
    """

    output_path = Path(output_path)
    if output_path.exists() and overwrite is False:
        print(
            f"[yellow] ⚠ A file already exists[/yellow] at {str(output_path.resolve())}",
        )
        overwrite = typer.confirm("are you sure you want to overwrite it?")
        if not overwrite:
            raise Abort()

    save_path = convert(input_path, output_path, format, overwrite)
    print(
        f"[green]✔ Finished [/green]: Converted {input_path} to {save_path.suffix}. "
        + f"Saved at:\n{save_path.resolve()}",
    )


# def setup_cli() -> None:
#     # Ensure that all app.commands are run
#     # from .convert import convert_cli  # noqa

#     command = typer.main.get_command(app)
#     # command(prog_name=NAME)
