#!/bin/bash
#SBATCH --time=24:00:00
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
PHENO_PATH='/home/kce/NLPPred/phenos'
PHENOS=("alkaline.pheno" "bilirubin.pheno" "cholesterol.pheno" "hba1c.pheno" "height.pheno" "urate.pheno")
N_CORES=4
DATA_PATH="/home/kce/dsmwpred/data/ukbb"
GENO_PATH='geno' 

# create subset based on intersection between phenotype and compressed genotype data (CSNP)
CSNP_PATH='/home/kce/NLPPred/github/snip/data/compressed/whole_geno/combined_sped/chr1-22_20k_relu_512_c_snps_train.fam'
# 1. get the individual IDs from the sped path
cut -f 1,2 $CSNP_PATH > $OUTPUT_PATH/individuals.txt
# 2. get the individual IDs from the phenotype path
cut -f 1,2 /home/kce/dsmwpred/data/ukbb/height.train > $OUTPUT_PATH/individuals_pheno.txt
# 3. get the intersection
join -1 1 -2 1 $OUTPUT_PATH/individuals.txt $OUTPUT_PATH/individuals_pheno.txt > $OUTPUT_PATH/individuals_intersect.txt

# 

for PHENO in "${PHENOS[@]}"
    do
    # Perform single SNP analysis
    mkdir -p $OUTPUT_PATH/uncompressed

    echo "Performing single SNP analysis for $PHENO"

    $LDAK --linear $OUTPUT_PATH/uncompressed/"$PHENO.uncompressed.quant" \
        --bfile $DATA_PATH/$GENO_PATH \
        --pheno $PHENO_PATH/$PHENO \
        --mpheno 1 \
        --keep $OUTPUT_PATH/individuals_intersect.txt \
        --max-threads $N_CORES
done

# Calculate kinship matrix
# https://dougspeed.com/calculate-kinships/
$LDAK --calc-kins-direct $OUTPUT_PATH/uncompressed/geno.GCTA \
    --bfile $DATA_PATH/$GENO_PATH \
    --ignore-weights YES \
    --power -1 \
    --max-threads $N_CORES \
    --keep $OUTPUT_PATH/individuals_intersect.txt

for PHENO in "${PHENOS[@]}"
    do
    # perform REML analysis
    $LDAK --reml $OUTPUT_PATH/uncompressed/"$PHENO.uncompressed.reml1" \
        --bfile $DATA_PATH/$GENO_PATH \
        --pheno $PHENO_PATH/$PHENO \
        --grm $OUTPUT_PATH/uncompressed/geno.GCTA \
        --max-threads $N_CORES \
        --keep $OUTPUT_PATH/individuals_intersect.txt
done


/home/kce/NLPPred/github/snip/data/ldak_results/uncompressed/height.pheno/uncompressed.quant.coeff