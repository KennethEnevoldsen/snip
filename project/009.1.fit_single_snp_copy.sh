#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem 32g
#SBATCH -c 4
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com

# this script is the same as the 009.fit_single_snp.sh script but it is
# modified to run using multiple phenotypes

# VARIABLES
LDAK='/home/kce/NLPPred/github/snip/ldak/ldak5.2.linux'
OUTPUT_PATH='/home/kce/NLPPred/github/snip/data/ldak_results'
PHENO_PATH='/home/kce/NLPPred/phenos'
PHENOS=("alkaline.pheno" "bilirubin.pheno" "cholesterol.pheno" "hba1c.pheno" "height.pheno" "urate.pheno")
N_CORES=4

DATA_PATH='/home/kce/NLPPred/github/snip/data/compressed/whole_geno/combined_sped'
GENO_PATHS=("chr1-22_20k_relu_512_c_snps_train" "chr1-22_20k_relu_16_c_snps_train" "chr1-22_20k_identity_512_c_snps_train" "chr1-22_20k_identity_16_c_snps_train")

# # perform single SNP analysis
# for GENO_PATH in "${GENO_PATHS[@]}"
#     do
#     for PHENO in "${PHENOS[@]}"
#         do

#             echo "Processing: $DATA_PATH/$GENO_PATH.sped"
#             echo "output path: $OUTPUT_PATH/$GENO_PATH/$PHENO.quant"

#             # create folder if it doesn't exist
#             mkdir -p $OUTPUT_PATH/$GENO_PATH
#             mkdir -p $OUTPUT_PATH/$GENO_PATH/$PHENO

#             $LDAK --linear $OUTPUT_PATH/$GENO_PATH/$PHENO.quant \
#                 --sped $DATA_PATH/$GENO_PATH \
#                 --SNP-data NO \
#                 --pheno $PHENO_PATH/$PHENO \
#                 --mpheno 1 \
#                 --max-threads $N_CORES
#     done
# done 

# # Calculate kinship matrix
# # https://dougspeed.com/per-predictor-heritabilities/
# for GENO_PATH in "${GENO_PATHS[@]}"
#     do
#     for PHENO in "${PHENOS[@]}"
#         do

#             echo "Processing: $DATA_PATH/$GENO_PATH.sped"
#             echo "Output path: $OUTPUT_PATH/$GENO_PATH/$PHENO.GCTA"
#             $LDAK --calc-kins-direct $OUTPUT_PATH/$GENO_PATH/$PHENO.GCTA \
#                 --sped $DATA_PATH/$GENO_PATH \
#                 --ignore-weights YES \
#                 --power -1 \
#                 --SNP-data NO \
#                 --max-threads $N_CORES

#     done
# done

# # perform REML analysis
for GENO_PATH in "${GENO_PATHS[@]}"
    do
    for PHENO in "${PHENOS[@]}"
        do

            echo "Processing: $DATA_PATH/$GENO_PATH.sped"
            echo "Output path: $OUTPUT_PATH/$GENO_PATH/$PHENO.reml1"
            $LDAK --reml $OUTPUT_PATH/$GENO_PATH/$PHENO.reml1 \
                --sped $DATA_PATH/$GENO_PATH \
                --pheno $PHENO_PATH/$PHENO \
                --SNP-data NO \
                --grm $OUTPUT_PATH/$GENO_PATH/$PHENO.GCTA \
                --max-threads $N_CORES 
    done
done