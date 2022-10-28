"""Short script to split a dataset into train and test sets."""

from snip.data import PLINKIterableDataset

# ds = PLINKIterableDataset("/home/kce/dsmwpred/data/ukbb/geno.bed", chromosome=22)

# train, test = ds.train_test_split(test_size=0.1)
# train, val = train.train_test_split(test_size=0.1)

# train.to_disk("data/processed/chrom_22_train.zarr", mode="w")
# val.to_disk("data/processed/chrom_22_validation.zarr", mode="w")
# test.to_disk("data/processed/chrom_22_test.zarr", mode="w")


ds = PLINKIterableDataset("/home/kce/dsmwpred/data/ukbb/geno.bed")

train, test = ds.train_test_split(test_size=0.1)
train, val = train.train_test_split(test_size=0.1)

train.to_disk("data/processed/chrom_1_train.zarr", mode="w")
val.to_disk("data/processed/chrom_1_validation.zarr", mode="w")
test.to_disk("data/processed/chrom_1_test.zarr", mode="w")

ds = PLINKIterableDataset("data/processed/chrom_1_train.zarr")
