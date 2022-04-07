from ctypes import Union
from pathlib import Path
from snip.data.dataloaders import PLINKIterableDataset


def save_to_zarr(
    save_path: Union[str, Path], read_path: Union[str, Path], overwrite: bool = False
) -> None:
    ds = PLINKIterableDataset(read_path)
    ds.to_disk(save_path, overwrite=True)