"""Functionality for converting a .zarr file to a .sped file."""
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
from dask import array as da
from pandas_plink._read import _read_bim, _read_fam
from pandas_plink._write import _fill_sample, _fill_variant, _write_bim, _write_fam


def write_sped(
    G: xr.DataArray,
    sped: Union[str, Path],
    bim: Union[str, Path, None] = None,
    fam: Union[str, Path, None] = None,
) -> None:
    """Writes dataarray to .sped file.

    A .sped binary file set consists of three files:

    - .sped: containing the genotype.
    - .bim: containing variant information.
    - .fam: containing sample information.

    The user must provide the genotype (dosage) via a :class:`xarray.DataArray`.
    That matrix must have two named dimensions: **sample** and **variant**.

    Args:
        G (xr.DataArray): The genotype matrix.
        sped (str): The path to the .sped file.
        bim (Union[str, Path, None], optional): The path to the .bim file. If None, the
            .bim file will be written to the same directory as the .sped file.
        fam (Union[str, Path, None], optional): The path to the .fam file. If None, the
            .fam file will be written to the same directory as the .sped file.

    Example:
        >>> from xarray import DataArray
        >>> from src.data.write_to_sped import write_to_sped
        >>>
        >>> G = DataArray(
        ...     [[3.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
        ...     dims=["sample", "variant"],
        ...     coords = dict(
        ...         sample  = ["boffy", "jolly"],
        ...         fid     = ("sample", ["humin"] * 2 ),
        ...
        ...         variant = ["not", "sure", "what"],
        ...         snp     = ("variant", ["rs1", "rs2", "rs3"]),
        ...         chrom   = ("variant", ["1", "1", "2"]),
        ...         a0      = ("variant", ['A', 'T', 'G']),
        ...         a1      = ("variant", ['C', 'A', 'T']),
        ...     )
        ... )
        >>> write_to_sped(G, "./test.sped")
    """
    if G.ndim != 2:
        raise ValueError("G has to be bidimensional")

    if set(list(G.dims)) != {"sample", "variant"}:
        raise ValueError("G has to have both `sample` and `variant` dimensions.")

    G = G.transpose("sample", "variant")

    sped = Path(sped)

    if bim is None:
        bim = sped.with_suffix(".bim")

    if fam is None:
        fam = sped.with_suffix(".fam")

    bim = Path(bim)
    fam = Path(fam)

    G = _fill_sample(G)
    G = _fill_variant(G)
    _write_bim(bim, G)
    _write_fam(fam, G)
    _write_sped(G, sped)


def _write_sped(G: xr.DataArray, sped: Union[str, Path]) -> None:
    """Writes the .sped file.

    Args:
        G (xr.DataArray): The genotype matrix.
        sped (str): The path to the .sped file.
    """
    sped = Path(sped)
    arr = G.data
    if isinstance(arr, da.Array):
        arr = arr.compute()
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Array type: {type(arr)} is not handled")
    arr.astype(np.float32).T.tofile(sped)


def read_sped(sped: Union[str, Path]) -> xr.DataArray:
    """Reads a .sped file.

    Args:
        sped (Union[str, Path]): The path to the .sped file.

    Returns:
        xr.DataArray: The genotype matrix.
    """
    sped = Path(sped)
    bim = sped.with_suffix(".bim")
    fam = sped.with_suffix(".fam")

    arr = np.fromfile(sped, dtype=np.float32)

    bim = _read_bim(bim)
    fam = _read_fam(fam)

    # reshape based on fam and bim
    arr = arr.reshape((len(bim), len(fam)))
    arr = arr.T

    coords = {
        "variant": bim.index,
        "sample": fam.index,
    }

    # add variant, samples coordinates
    for col in bim.columns:
        coords[col] = ("variant", bim[col].values)
    for col in fam.columns:
        coords[col] = ("sample", fam[col].values)

    return xr.DataArray(arr, dims=["sample", "variant"], coords=coords)
