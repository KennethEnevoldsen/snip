from snip.data import PLINKIterableDataset

ds = PLINKIterableDataset("/home/kce/dsmwpred/data/ukbb/geno.bed", chromosome=22)
train, test = ds.train_test_split(test_size=0.1)
train, val = train.train_test_split(test_size=0.1)

train.to_disk("data/processed/chrom_22_train.zarr")
val.to_disk("data/processed/chrom_22_validation.zarr")
test.to_disk("data/processed/chrom_22_test.zarr")
