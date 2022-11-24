"""Script for creating slurm runs and calling the training of the sklearn model
with different training hyperparameters."""
import os
from itertools import product

outline = """#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem 32g
#SBATCH -c {cores}
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com

python src/snip/train_slided_autoencoder_sklearn.py \\
    project.wandb_mode=dryrun \\
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
    "limit": [20000],
    "activation": ["relu", "identity"],
    "cores": [8],
    "chromosome": list(range(1, 23)),
    "stride": [512, 16],
}


# create a list of all combinations of the parameters
combinations = list(product(*variations.values()))

# create slurm commands
for i, combination in enumerate(combinations):
    # create a dictionary of the parameters
    params = dict(zip(variations.keys(), combination))
    params[
        "prefix"
    ] = f"chr{params['chromosome']}_20k_{params['activation']}_{params['stride']}"
    # calculate the hidden size
    stride = params["stride"]
    hidden_size = int(stride / 2)
    intermediate_hidden = stride - int((stride - hidden_size) / 2)
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
    # clean up the slurm file
    os.system(f"rm {path}")
