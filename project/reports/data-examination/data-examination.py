"""Checks that the compressed SNPs is in the correct format."""
from snip.data import PLINKIterableDataset

data_path = "/home/kce/NLPPred/github/snip/data/compressed/c_snps_train.zarr"
compressed_geno = PLINKIterableDataset(data_path)

data_path = "/home/kce/NLPPred/github/snip/data/processed/chrom_22_train.zarr"
geno = PLINKIterableDataset(data_path)

assert geno.genotype.shape[0] == compressed_geno.genotype.shape[0]
assert compressed_geno.genotype.shape[1] * 2 <= geno.genotype.shape[1]
