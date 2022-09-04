#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --mem 64g
#SBATCH -c 16
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred

# snip convert /home/kce/dsmwpred/data/ukbb/geno.bed data/processed/chrom_6.zarr --format zarr --chromosome 6 --overwrite
# snip convert /home/kce/dsmwpred/data/ukbb/geno.bed data/processed/chrom_21.zarr --format zarr --chromosome 21 --overwrite
snip convert /home/kce/dsmwpred/data/ukbb/geno.bed data/processed/chrom_22.zarr --format zarr --chromosome 22 --overwrite

# split into train test:
snip train_test_split data/processed/chrom_22.zarr data/processed/chrom_22_train.zarr data/processed/chrom_22_test.zarr --test_size 0.1 --overwrite
snip train_test_split data/processed/chrom_22.zarr data/processed/chrom_22_train.zarr data/processed/chrom_22_val.zarr --test_size 0.1 --overwrite
