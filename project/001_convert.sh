#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --mem 64g
#SBATCH -c 16
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred
#SBATCH --mail-type=END
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com


python project/002_train_test_split.py 