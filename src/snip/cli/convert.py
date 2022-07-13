from pathlib import Path
from typing import Optional, Union

from snip.data.dataloaders import PLINKIterableDataset


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
    else:
        raise ValueError(f"Format {format} is not supported")

    return save_path
