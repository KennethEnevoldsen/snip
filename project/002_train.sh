#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem 64g
#SBATCH -c 8
#SBATCH --gres=gpu:1 
#SBATCH -p gpu
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com

python src/snip/train_slided_autoencoder.py accelerator=gpu wandb_mode=run --config-name default_config_train_slided_autoencoder.yaml