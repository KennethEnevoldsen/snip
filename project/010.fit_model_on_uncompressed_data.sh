#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem 32g
#SBATCH -c 4
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com

# perform single SNP analysis
# https://dougspeed.com/single-predictor-analysis/



# VARIABLES
LDAK='/home/kce/NLPPred/github/snip/ldak/ldak5.2.linux'
OUTPUT_PATH='/home/kce/NLPPred/github/snip/data/ldak_results'
PHENO_PATH='/home/kce/dsmwpred/data/ukbb/height.train'
N_CORES=4
DATA_PATH="/home/kce/dsmwpred/data/ukbb"
GENO_PATH='geno' 

# create subset based on intersection between phenotype and compressed genotype data (CSNP)
CSNP_PATH='/home/kce/NLPPred/github/snip/data/compressed/whole_geno/combined_sped/chr1-22_20k_relu_512_c_snps_train.fam'
# 1. get the individual IDs from the sped path
cut -f 1,2 $CSNP_PATH > $OUTPUT_PATH/individuals.txt
# 2. get the individual IDs from the phenotype path
cut -f 1,2 $PHENO_PATH > $OUTPUT_PATH/individuals_pheno.txt
# 3. get the intersection
join -1 1 -2 1 $OUTPUT_PATH/individuals.txt $OUTPUT_PATH/individuals_pheno.txt > $OUTPUT_PATH/individuals_intersect.txt


# Calculate kinship matrix
# https://dougspeed.com/calculate-kinships/
$LDAK --calc-kins-direct $OUTPUT_PATH/$GENO_PATH.GCTA \
    --bfile $DATA_PATH/$GENO_PATH \
    --ignore-weights YES \
    --power -1 \
    --max-threads $N_CORES \
    --keep $OUTPUT_PATH/individuals_intersect.txt

# perform REML analysis
$LDAK --reml $OUTPUT_PATH/$GENO_PATH.reml1 \
    --bfile $DATA_PATH/$GENO_PATH \
    --pheno $PHENO_PATH \
    --grm $OUTPUT_PATH/$GENO_PATH.GCTA \
    --max-threads $N_CORES \
    --keep $OUTPUT_PATH/individuals_intersect.txt
