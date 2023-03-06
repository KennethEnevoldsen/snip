"""Script for combining output of 007.1 into a single file concating them along
the snp axis."""

import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from wasabi import msg

from snip.data import PLINKIterableDataset


def extract_info(f):
    match = re.search(
        r"chr(\d+)_(\d+)k_(\w+)_(\d+)_compression(\d+[,.]*\d*)(.*)_(\d{4}-\d{2}-\d{2})",
        f.name,
    )

    if match:
        chr, n_indiv, act, width, compression, name, date = match.groups()
        return chr, n_indiv, act, width, compression, name, date, f


def dataframe_from_folders(data_path):
    folders = [f for f in data_path.iterdir() if f.is_dir()]
    folders = [f for f in folders if f.name.startswith("chr")]
    # remove folders which are empty
    for f in folders:
        if not list(f.iterdir()):
            # delete the folder
            print(f"Deleting empty folder {f}")
            f.rmdir()

    folders = [f for f in folders if (f / "metadata.jsonl").exists()]

    table = [extract_info(f) for f in folders]
    table = list(filter(lambda x: x is not None, table))
    table = pd.DataFrame(
        table,
        columns=[
            "chr",
            "n_indiv",
            "act",
            "width",
            "compression",
            "name",
            "date",
            "folder",
        ],
    )

    # remove if there is duplicates which have the same values in all of ["n_indiv", "act", "width", "compression", "chr"]
    n_row = table.shape[0]
    table = table.drop_duplicates(["n_indiv", "act", "width", "compression", "chr"])
    print(f"Removed {n_row - table.shape[0]} duplicates")

    # # print filenames not in the table
    # print("Files not in the table:")
    # for f in [f for f in data_path.iterdir() if f.is_dir()]:
    #     if f.name not in table["folder"].values:
    #         print(f.name)

    return table


def write_to_sped(df, act, width, compression):
    data_path = "/home/kce/NLPPred/github/snip/data/compressed/whole_geno"
    start_time = time.time()
    paths = df["folder"].tolist()
    chroms = df["chr"].tolist()
    splits = ["train", "validation", "test"]
    for split in splits:
        msg.info(f"Combining {split} data")
        save_path = (
            str(data_path)
            + f"/combined/chr1-22_20k_{act}_{width}_compression_{compression}_c_snps_{split}.sped"
        )
        if Path(save_path).exists():
            msg.info("Save path already exists, skipping")
            print(f"{save_path}")
            continue

        datasets = []
        for path, chrom in zip(paths, chroms):
            # list files
            ds = PLINKIterableDataset(path / f"c_snps_{split}.zarr")

            # run checks
            assert np.unique(ds.genotype["chrom"].data.compute()) == chrom
            variants = list(ds.genotype["variant"].data)
            assert len(variants) != len(
                set(variants),
            ), "There are duplicates in the variant column"

            datasets.append(ds)

        # combine datasets
        genotypes = [ds.genotype for ds in datasets]
        genotype = xr.concat(genotypes, dim="variant")

        ds = datasets[0]
        ds.genotype = genotype
        ds.path = ""

        msg.info("Saving to sped file")
        # ensure path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_disk(save_path)
        msg.good(
            f"Saved to sped file (time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}), saved to:\n {save_path}",
        )


if __name__ == "__main__":
    repo_dir = Path(__file__).resolve().parents[1]
    data_path = repo_dir / "data" / "compressed" / "whole_geno"

    table = dataframe_from_folders(data_path)

    for group, df in table.groupby(["n_indiv", "act", "width", "compression"]):
        assert (
            max(Counter(df["chr"]).values()) == 1
        ), "There are duplicates in the chr column"
        n, act, width, compression = group

        if n != "20":
            continue

        # check for missing compressions
        print(group, end=" ")
        missing_a_chrom = False
        for i in range(1, 23):
            if str(i) not in df["chr"].values:
                print(f"\n - Missing chr {i}")
                missing_a_chrom = True
        if missing_a_chrom:
            continue

        write_to_sped(df, act, width, compression)
