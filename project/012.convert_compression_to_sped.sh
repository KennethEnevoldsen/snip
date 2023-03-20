#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem 380g
#SBATCH -c 4
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com

python project/012.convert_compression_to_sped.py