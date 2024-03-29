"""Script for creating slurm runs and calling the training of the sklearn model
with different training hyperparameters."""
import os
from itertools import product

outline = """#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem 64g
#SBATCH -c {cores}
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com

python src/snip/train_slided_autoencoder_sklearn.py \\
    project.wandb_mode=run \\
    project.n_jobs={cores} \\
    data.stride={stride} \\
    data.width={stride} \\
    model.hidden_layer_sizes='{hidden_size}' \\
    data.result_path=data/compressed/whole_geno \\
    data.train_path=/home/kce/NLPPred/github/snip/data/raw/ukbb_geno_train.zarr \\
    data.validation_path=/home/kce/NLPPred/github/snip/data/raw/ukbb_geno_validation.zarr \\
    data.test_path=/home/kce/NLPPred/github/snip/data/raw/ukbb_geno_test.zarr \\
    model.activation={activation} \\
    data.limit={limit} \\
    data.chromosome={chromosome} \\
    project.run_name_prefix={prefix}
"""

variations = {
    "limit": [
        # 20000,
        # 50000,
        100000,
        # 200000,
        # "null",
    ],
    "activation": [
        "relu",
        # "identity",
    ],
    "cores": [12],
    # "chromosome": list(range(1, 23)),
    "chromosome": [18],
    "stride": [
        # 512,
        16,
    ],
    "compression": [
        # 2,
        4,
        # 1.5,
    ],
}
# _slurm_chr18_100k_identity_16_compression4.

# create a list of all combinations of the parameters
combinations = list(product(*variations.values()))

# create slurm commands
for i, combination in enumerate(combinations):
    # create a dictionary of the parameters
    params = dict(zip(variations.keys(), combination))
    if params["limit"] != "null":
        limit_in_k = params["limit"] // 1000
    else:
        limit_in_k = "null"
    params[
        "prefix"
    ] = f"chr{params['chromosome']}_{limit_in_k}k_{params['activation']}_{params['stride']}_compression{params['compression']}"
    # calculate the hidden size
    stride = params["stride"]
    hidden_size = int(stride / params["compression"])
    intermediate_hidden = stride - int((stride - hidden_size) / params["compression"])
    params["hidden_size"] = [intermediate_hidden, hidden_size, intermediate_hidden]
    # create the slurm command
    slurm_command = outline.format(**params)
    # create the slurm file
    filename = f"007.1_{i}_slurm_{params['prefix']}.sh"
    path = f"project/{filename}"
    with open(path, "w") as f:
        f.write(slurm_command)

    # sbatch the slurm file
    os.system(f"sbatch {path}")
    # # clean up the slurm file
    os.system(f"rm {path}")
