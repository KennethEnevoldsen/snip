#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --mem 64g
#SBATCH -c 16
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred

snip convert /home/kce/dsmwpred/data/ukbb/geno.bed data/processed/chrom_21.zarr --format zarr --chromosome 21