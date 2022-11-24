"""Short script to split a dataset into train and test sets."""

from snip.data import PLINKIterableDataset

ds = PLINKIterableDataset("/home/kce/dsmwpred/data/ukbb/geno.bed")

train, test = ds.train_test_split(test_size=0.1)
train, val = train.train_test_split(test_size=0.1)

train.to_disk("data/raw/ukbb_geno_train.zarr", mode="w")
val.to_disk("data/raw/ukbb_geno_validation.zarr", mode="w")
test.to_disk("data/raw/ukbb_geno_test.zarr", mode="w")
