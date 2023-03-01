"""Script for creating slurm runs."""
import os
from itertools import product

outline = """#!/bin/bash
#SBATCH --mem 32G
#SBATCH -t 10:0:0
#SBATCH -c 4
#SBATCH -A NLPPred
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out


# variables
PROJECT_PATH=$HOME/NLPPred/github/snip
LDAK=$PROJECT_PATH/ldak/ldak5.2.linux
KEEP=$PROJECT_PATH/data/compressed/whole_geno/combined_sped/chr1-22_20k_identity_16_c_snps_train.fam
BFILE=$HOME/dsmwpred/data/ukbb/geno
R2={pruning-r2}
N_CORES=4

OUTPUT_PATH='/home/kce/NLPPred/github/snip/data/ldak_results'
PHENO_PATH='/home/kce/NLPPred/phenos'
PHENO="{phenotype}.pheno"
DATA_PATH="/home/kce/dsmwpred/data/ukbb"
GENO_PATH='geno'

mkdir -p $OUTPUT_PATH/pruning

# create kinship matrix

# check if it exists
if [ -f $OUTPUT_PATH/pruning/geno.GCTA.grm.details ]; then
    echo "Kinship matrix already exists"
else
    $LDAK --calc-kins-direct $OUTPUT_PATH/pruning/geno.GCTA \\
        --bfile $DATA_PATH/$GENO_PATH \\
        --ignore-weights YES \\
        --power -1 \\
        --max-threads $N_CORES \\
        --keep $OUTPUT_PATH/individuals_intersect.txt
fi


# prune
if [ -f $OUTPUT_PATH/pruning/prune.$R2.in ]; then
    echo "Pruning already done"
else
    $LDAK --thin $OUTPUT_PATH/pruning/prune.$R2 \\
            --bfile $BFILE \\
            --max-threads $N_CORES \\
            --keep $KEEP \\
            --window-cm 1 \\
            --window-prune $R2
fi


# Perform single SNP analysis

SS_ANALYSIS_DONE=false
# check if single SNP analysis already done by checking
# 1) if $OUTPUT_PATH/pruning/"$PHENO.prune.$R2.quant.progress" exists
if [ -f $OUTPUT_PATH/pruning/"$PHENO.prune.$R2.quant.progress" ]; then
    # and 2) if the last line in the file states that X is equal to Y. The last line is on the form: "Performing single-SNP analysis for Chunk X of Y"
    echo "Single SNP analysis already started"
    LAST_LINE=$(tail -n 1 $OUTPUT_PATH/pruning/"$PHENO.prune.$R2.quant.progress")
    # extract X and Y from the last line
    X=$(echo $LAST_LINE | cut -d' ' -f 6)
    Y=$(echo $LAST_LINE | cut -d' ' -f 8)
    if [ $X -eq $Y ]; then
        echo "Single SNP analysis already done"
        SS_ANALYSIS_DONE=true
    else
        echo "but Single SNP analysis is not done"
    fi
fi

if [ $SS_ANALYSIS_DONE = false ]; then
    echo "Performing single SNP analysis for $PHENO"
    $LDAK --linear $OUTPUT_PATH/pruning/"$PHENO.prune.$R2.quant" \\
        --bfile $DATA_PATH/$GENO_PATH \\
        --pheno $PHENO_PATH/$PHENO \\
        --mpheno 1 \\
        --keep $OUTPUT_PATH/individuals_intersect.txt \\
        --extract $OUTPUT_PATH/pruning/prune.$R2.in \\
        --max-threads $N_CORES
fi

# perform REML analysis
if [ -f $OUTPUT_PATH/pruning/"$PHENO.prune.$R2.reml1.reml" ]; then
    echo "REML analysis already done"
else
    echo "Performing REML analysis for $PHENO"
    $LDAK --reml $OUTPUT_PATH/pruning/"$PHENO.prune.$R2.reml1" \\
        --bfile $OUTPUT_PATH/pruning/prune.$R2.in \\
        --pheno $PHENO_PATH/$PHENO \\
        --mpheno 1 \\
        --grm $OUTPUT_PATH/pruning/geno.GCTA \\
        --extract $OUTPUT_PATH/pruning/prune.$R2.in \\
        --keep $OUTPUT_PATH/individuals_intersect.txt \\
        --max-threads $N_CORES
fi
"""

variations = {
    "phenotype": ["alkaline", "bilirubin", "cholesterol", "hba1c", "height", "urate"],
    "pruning-r2": [0.05, 0.1, 0.2, 0.5, 0.8],
}


# create a list of all combinations of the parameters
combinations = list(product(*variations.values()))

# create slurm commands
for i, combination in enumerate(combinations):
    # create a dictionary of the parameters
    params = dict(zip(variations.keys(), combination))

    # create the run name
    filename = os.path.basename(__file__)[:-3]  # no .py
    filename += "_" + "_".join([f"{k}_{v}" for k, v in params.items()])
    filename += ".sh"

    # create the slurm command
    slurm_command = outline.format(**params)
    # create the slurm file
    path = f"project/{filename}"
    with open(path, "w") as f:
        f.write(slurm_command)

    # sbatch the slurm file
    os.system(f"sbatch {path}")
    # # clean up the slurm file
    os.system(f"rm {path}")
