#!/bin/bash
#SBATCH --time=11:00:00
#SBATCH --mem 32g
#SBATCH -c 16
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred
#SBATCH --mail-type=END
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com

GENO_PATH=/home/kce/dsmwpred/data/ukbb/geno.bed
DATA_FOLDER=/home/kce/NLPPred/github/snip/data

# convert the .bed data to .zarr
snip convert $GENO_PATH $DATA_FOLDER/raw/ukbb_geno.zarr