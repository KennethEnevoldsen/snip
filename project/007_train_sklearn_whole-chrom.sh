#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem 64g
#SBATCH -c 16
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com

python src/snip/train_slided_autoencoder_sklearn.py \
    project.wandb_mode=run \
    project.n_jobs=16 \
    data.result_path=data/compressed/relu \
    data.train_path=data/raw/ukbb_geno.zarr \
    data.limit=20000 \
    project.run_name_prefix=whole-chrom_20k_relu512

python src/snip/train_slided_autoencoder_sklearn.py \
    project.wandb_mode=run \
    project.n_jobs=16 \
    data.result_path=data/compressed/linear \
    data.train_path=data/raw/ukbb_geno.zarr \
    data.limit=20000 \
    model.activation=identity \
    project.run_name_prefix=whole-chrom_20k_linear512
