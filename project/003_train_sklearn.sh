#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem 64g
#SBATCH -c 16
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com
 
python src/snip/train_slided_autoencoder_sklearn.py project.wandb_mode=run project.n_jobs=16