#!/bin/bash
#SBATCH --mem 32G
#SBATCH -t 10:0:0
#SBATCH -c 4
#SBATCH -A NLPPred
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out


# variables
PROJECT_PATH=$HOME/NLPPred/github/snip
LDAK=$PROJECT_PATH/ldak/ldak5.2.linux
KEEP=$PROJECT_PATH/data/compressed/whole_geno/combined_sped/chr1-22_20k_identity_16_c_snps_train.fam
BFILE=$HOME/dsmwpred/data/ukbb/geno
R2=0.8
N_CORES=4

OUTPUT_PATH='/home/kce/NLPPred/github/snip/data/ldak_results'
PHENO_PATH='/home/kce/NLPPred/phenos'
PHENO="height.pheno"
DATA_PATH="/home/kce/dsmwpred/data/ukbb"
GENO_PATH='geno' 

mkdir -p $OUTPUT_PATH/pruning

# create kinship matrix

# check if it exists
if [ -f $OUTPUT_PATH/pruning/geno.GCTA ]; then
    echo "Kinship matrix already exists"
else
    $LDAK --calc-kins-direct $OUTPUT_PATH/pruning/geno.GCTA \
        --bfile $DATA_PATH/$GENO_PATH \
        --ignore-weights YES \
        --power -1 \
        --max-threads $N_CORES \
        --keep $OUTPUT_PATH/individuals_intersect.txt
fi


# prune
$LDAK --thin $OUTPUT_PATH/pruning/prune.$R2 \
        --bfile $BFILE \
        --max-threads $N_CORES \
        --keep $KEEP \
        --window-cm 1 \
        --window-prune $R2

echo "Performing single SNP analysis for $PHENO"

# Perform single SNP analysis
$LDAK --linear $OUTPUT_PATH/pruning/"$PHENO.prune.$R2.quant" \
    # --bfile $DATA_PATH/$GENO_PATH \
    --bfile $DATA_PATH/$GENO_PATH \
    --pheno $PHENO_PATH/$PHENO \
    --mpheno 1 \
    --keep $OUTPUT_PATH/individuals_intersect.txt \
    --extract $OUTPUT_PATH/pruning/prune.$R2.in \
    --max-threads $N_CORES

# perform REML analysis
$LDAK --reml $OUTPUT_PATH/pruning/"$PHENO.prune.$R2.reml1" \
    --bfile $OUTPUT_PATH/pruning/prune.$R2.in \
    --pheno $PHENO_PATH/$PHENO \
    --mpheno 1 \
    --grm $OUTPUT_PATH/pruning/"
    --keep $OUTPUT_PATH/individuals_intersect.txt \
    --max-threads $N_CORES
"""