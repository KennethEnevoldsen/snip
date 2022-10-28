"""train a linear model from compression."""


import pandas as pd

from snip.data import PLINKIterableDataset

ds = PLINKIterableDataset(
    "/home/kce/NLPPred/github/snip/data/compressed/c_snps_train.zarr",
)


path = "/home/kce/dsmwpred/data/ukbb/height.train"

# load df and set columns
df = pd.read_csv(path, sep=" ")
df.columns = ["ID", "ID2", "height"]
df.drop("ID2", axis=1, inplace=True)

# extract xarray
geno = ds.genotype
# filter ds to only include variang iid equal to IDs in df
geno.where(geno.sample.isin(df.ID), drop=True)
