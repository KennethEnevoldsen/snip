import typer

COMMAND = "python -m snip"
HELP = "Snip Command-line Interface"
NAME = "snip"

app = typer.Typer(name=NAME, help=HELP)


def setup_cli() -> None:
    """Setups command line interface."""
    # Ensure that all app.commands are run
    from .convert import convert_cli  # noqa
    from .create_test_data import create_test_data_cli  # noqa
    from .train_test_split import train_test_split_cli  # noqa

    command = typer.main.get_command(app)
    command(prog_name=COMMAND)
    return app()
