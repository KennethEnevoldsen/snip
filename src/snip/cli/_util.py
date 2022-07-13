import typer

COMMAND = "python -m snip"
HELP = "Snip Command-line Interface"
NAME = "snip"

app = typer.Typer(name=NAME, help=HELP)


@app.command()
def create(user: str):
    typer.echo(f"Creating user: {user}")


@app.command()
def delete(user: str):
    typer.echo(f"Deleting user: {user}")


def setup_cli() -> None:
    # Ensure that all app.commands are run
    from .convert import convert_cli  # noqa

    # Ensure that the help messages always display the correct prompt
    command = typer.main.get_command(app)
    command(prog_name=COMMAND)
