"""Script for combining output of 007.1 into a single file concating them along
teh snp axis."""
import time
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
from wasabi import msg

from snip.data import PLINKIterableDataset


def write_to_sped(act, width, split):
    """Write the compressed files to sped file."""
    msg.info(f"Processing act: {act}, width: {width}, split: {split}")
    start_time = time.time()

    chromosomes = list(range(1, 23))
    data_path = "/home/kce/NLPPred/github/snip/data/compressed/whole_geno"
    path = (
        data_path + "/chr{chrom}_20k_{act}_{width}None_2022-11-14/c_snps_{split}.zarr"
    )
    # get all the paths
    paths = [
        path.format(chrom=chrom, act=act, width=width, split=split)
        for chrom in chromosomes
    ]

    def load_and_fix_dataset(path: Union[str, Path]) -> PLINKIterableDataset:
        """Load dataset and fix the coords:

        This include:
            - setting the chrom coord
            - setting the snp coord
            - creating a unique variant
        """
        chrom = int(path.split("/")[-2].split("_")[0][3:])
        assert chrom in range(1, 23)
        # create string for chrom where length stable i.e. 1 -> "01"
        chrom_str = f"{chrom:02d}"
        ds = PLINKIterableDataset(path)
        genotype = ds.genotype

        n_variants = genotype.shape[1]
        chrom_coords = np.repeat(chrom, n_variants)
        snp_coords = np.array([f"c{chrom_str}_{i}" for i in range(n_variants)])

        coords = {
            "chrom": ("variant", chrom_coords),
            "snp": ("variant", snp_coords),
            "variant": ("variant", np.arange(n_variants)),
        }
        genotype = genotype.assign_coords(coords)
        ds.genotype = genotype
        return ds

    # load all the datasets
    datasets = [load_and_fix_dataset(path) for path in paths]
    genotypes = [ds.genotype for ds in datasets]

    # combine them
    genotype = xr.concat(genotypes, dim="variant")

    assert genotype.shape[0] == genotypes[0].shape[0], "variants are not the same"
    assert len(np.unique(genotype.chrom.compute().values)) == 22
    assert len(np.unique(genotype.snp.compute().values)) == genotype.shape[1]

    ds = datasets[0]
    ds.genotype = genotype
    ds.path = ""

    msg.info("Saving to sped file")
    save_path = data_path + f"/combined/chr1-22_20k_{act}_{width}_c_snps_{split}.sped"
    ds.to_disk(save_path)
    msg.good(
        f"Saved to sped file (time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}), saved to:\n {save_path}",
    )


if __name__ == "__main__":
    act = "identity"
    width = 512
    split = "train"
    for split in ["train", "validation", "test"]:
        for width in [16, 512]:
            for act in ["identity", "relu"]:
                write_to_sped(act, width, split)
