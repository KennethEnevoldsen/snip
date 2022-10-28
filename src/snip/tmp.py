from snip.data import PLINKIterableDataset

ds = PLINKIterableDataset(
    "/home/kce/NLPPred/github/snip/data/compressed/c_snps_train.zarr",
    limit=10,
)

geno = ds.genotype

geno.sample
ds.to_disk("tmp.sped")

geno.fid.compute()

ds = PLINKIterableDataset(
    "/home/kce/NLPPred/github/snip/data/processed/chrom_22_test.zarr",
    limit=10,
)
ds.genotype.fid.compute()


ds = PLINKIterableDataset(
    "/home/kce/NLPPred/github/snip/data/processed/chrom_21.zarr",
    limit=10,
)
train, test = ds.train_test_split()
train.genotype.fid.compute()
