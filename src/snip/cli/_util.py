import typer

COMMAND = "python -m snip"
HELP = "Snip Command-line Interface"
NAME = "snip"

app = typer.Typer(name=NAME, help=HELP)


@app.command("delete")
def delete(user: str):
    # THIS is only as otherwise convert is no longer a command.
    typer.echo(f"Deleting user: {user}")


def setup_cli() -> None:
    # Ensure that all app.commands are run
    from .convert import convert_cli  # noqa

    command = typer.main.get_command(app)
    command(prog_name=COMMAND)
    return app()
