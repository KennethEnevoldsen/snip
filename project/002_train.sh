#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem 128g
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred

python src/snip/train_slided_autoencoder.py --config-name default_config_train_slided_autoencoder.yaml