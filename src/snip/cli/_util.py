import typer

HELP = "Snip Command-line Interface"
NAME = "snip"

app = typer.Typer(name=NAME, help=HELP)


def setup_cli() -> None:
    # Ensure that all app.commands are run
    from .convert import convert_cli  # noqa
