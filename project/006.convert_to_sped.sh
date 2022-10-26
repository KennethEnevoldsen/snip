#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem 32g
#SBATCH -c 16
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com

snip convert data/compressed/c_snps_train.zarr data/compressed/c_snps_train.sped