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

/home/kce/NLPPred/github/snip/data/compressed/whole_geno/combined_sped/chr1-22_20k_identity_16_c_snps_test.sped

DATA_PATH='/home/kce/NLPPred/github/snip/data/compressed/whole_geno/combined_sped'
GENO_PATHS=("chr1-22_20k_relu_512_c_snps_train" "chr1-22_20k_relu_16_c_snps_train" "chr1-22_20k_identity_512_c_snps_train" "chr1-22_20k_identity_16_c_snps_train")
 

# perform single SNP analysis
for GENO_PATH in "${GENO_PATHS[@]}"
do

    echo "Processing: $DATA_PATH/$GENO_PATH.sped"
    echo "output path: $OUTPUT_PATH/$GENO_PATH.quant"

    # check if the file exists

    
    # # check if output path exists if so then skip
    # if [ -f "$OUTPUT_PATH/$GENO_PATH.quant.assoc" ]; then
    #     echo "Output path exists, skipping..."
    #     continue
    # fi


    $LDAK --linear $OUTPUT_PATH/$GENO_PATH.quant \
        --sped $DATA_PATH/$GENO_PATH \
        --SNP-data NO \
        --pheno $PHENO_PATH \
        --mpheno 1 \
        --max-threads $N_CORES
done

# # perform Per-Predictor Heritabilities asuming the GCTA model
# # https://dougspeed.com/per-predictor-heritabilities/
# for GENO_PATH in "${GENO_PATHS[@]}"
# do
#     if [ -f "$OUTPUT_PATH/$GENO_PATH.gcta" ]; then
#         echo "Output path exists, skipping..."
#         continue
#     fi

#     echo "Processing: $DATA_PATH/$GENO_PATH.sped"
#     echo "Output path: $OUTPUT_PATH/$GENO_PATH.gcta"
#     $LDAK --calc-tagging $OUTPUT_PATH/$GENO_PATH.gcta \
#         --sped $DATA_PATH/$GENO_PATH \
#         --ignore-weights YES \
#         --power -1 \
#         --window-cm 1 \
#         --SNP-data NO \
#         --save-matrix YES

# done

# Calculate kinship matrix
# https://dougspeed.com/per-predictor-heritabilities/
for GENO_PATH in "${GENO_PATHS[@]}"
do

    # check if output path exists if so then skip
    if [ -f "$OUTPUT_PATH/$GENO_PATH.grm.id" ]; then
        echo "Output path exists, skipping..."
        continue
    fi

    echo "Processing: $DATA_PATH/$GENO_PATH.sped"
    echo "Output path: $OUTPUT_PATH/$GENO_PATH.GCTA"
    $LDAK --calc-kins-direct $OUTPUT_PATH/$GENO_PATH.GCTA \
        --sped $DATA_PATH/$GENO_PATH \
        --ignore-weights YES \
        --power -1 \
        --SNP-data NO \
        --max-threads $N_CORES

done

# perform REML analysis
for GENO_PATH in "${GENO_PATHS[@]}"
do
    # if [ -f "$OUTPUT_PATH/$GENO_PATH.reml1" ]; then
    #     echo "Output path exists, skipping..."
    #     continue
    # fi

    echo "Processing: $DATA_PATH/$GENO_PATH.sped"
    echo "Output path: $OUTPUT_PATH/$GENO_PATH.reml1"
    $LDAK --reml $OUTPUT_PATH/$GENO_PATH.reml1 \
        --sped $DATA_PATH/$GENO_PATH \
        --pheno $PHENO_PATH \
        --SNP-data NO \
        --grm $OUTPUT_PATH/$GENO_PATH.GCTA \
        --max-threads $N_CORES 
done
