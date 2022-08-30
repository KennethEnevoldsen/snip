from pathlib import Path
from typing import Optional, Tuple, Union

import typer
from rich import print
from typer import Abort, Argument, Option

from snip.data.dataloaders import PLINKIterableDataset

from .setup import app


def __check_and_normalize_train_test_size(
    train_size: Union[float, int, None],
    test_size: Union[float, int, None],
) -> Tuple[Union[float, int, None], Union[float, int, None]]:
    if train_size and test_size:
        raise ValueError("Cannot specify both train_size and test_size")
    if train_size is None and test_size is None:
        raise ValueError("Must specify either --train_size or --test_size")
    # check if float is actually an int
    if train_size:
        if train_size.is_integer():
            train_size = int(train_size)
    if test_size:
        if test_size.is_integer():
            test_size = int(test_size)

    return train_size, test_size


@app.command("train_test_split")
def train_test_split_cli(
    input_path: str = Argument(
        ...,
        help="Input file or directory",
    ),
    train_path: str = Argument(
        ...,
        help="Output file or directory for training set",
    ),
    test_path: str = Argument(
        ...,
        help="Output file or directory for test set",
    ),
    test_size: Optional[float] = Option(
        None,
        "--test_size",
        help="Size of the test set. If int, then it is the number of samples. If float"
        + ", then it is the proportion of the dataset.",
    ),
    train_size: Optional[float] = Option(
        None,
        "--train_size",
        help="Size of the test set. If int, then it is the number of samples. If float"
        + ", then it is the proportion of the dataset.",
    ),
    overwrite: bool = False,
):
    """CLI for creating a train and test set from a dataset.

    Args:
        input_path (str): Input file or directory.
        train_path (str): Output file or directory for training set.
        test_path (str): Output file or directory for test set.
        test_size (Optional[float]): Size of the test set. If int, then it
            is the number of samples. If float, then it is the proportion of the
            dataset.
        train_size (Optional[float]): Size of the train set. If int, then it
            is the number of samples. If float, then it is the proportion of the
            dataset.
        overwrite (bool): Should it overwrite the dataset.
    """

    _train_size, _test_size = __check_and_normalize_train_test_size(
        train_size,
        test_size,
    )

    _input_path: Path = Path(input_path)
    _train_path: Path = Path(train_path)
    _test_path: Path = Path(test_path)

    if (_train_path.exists() or _test_path.exists()) and overwrite is False:
        if _train_path.exists():
            path = _train_path
        else:
            path = _test_path
        print(
            f"[yellow] ⚠ A file already exists[/yellow] at {str(path.resolve())}",
        )
        overwrite = typer.confirm("are you sure you want to overwrite it?")
        if not overwrite:
            raise Abort()

    # ensure filepaths exist
    _train_path.parent.mkdir(parents=True, exist_ok=True)
    _test_path.parent.mkdir(parents=True, exist_ok=True)

    if _input_path.suffix == ".zarr":
        ds = PLINKIterableDataset(_input_path, verbose=False)
        ds.train_test_split(test_size=_test_size, train_size=_train_size)
        ds.to_disk(_train_path, overwrite=overwrite)
        ds.to_disk(_test_path, overwrite=overwrite)
    else:
        raise ValueError(f"Format {_input_path.suffix} is not supported")

    print(
        f"[green]✔ Finished [/green]: Train data created at:\n{_train_path.resolve()}",
    )
    print(
        f"[green]✔ Finished [/green]: Test data created at:\n{_test_path.resolve()}",
    )
