#!/bin/bash
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
N_CORES=4

OUTPUT_PATH='/home/kce/NLPPred/github/snip/data/ldak_results'
PHENO_PATH='/home/kce/NLPPred/phenos'
PHENO="alkaline.pheno"
SPED_FILE='/home/kce/NLPPred/github/snip/data/compressed/whole_geno/combined/chr1-22_20k_relu_16_compression_4_c_snps_train'
SAVE_FOLDER_NAME='chr1-22_20k_relu_16_compression_4_c_snps_train'

mkdir -p $OUTPUT_PATH/$SAVE_FOLDER_NAME

# create kinship matrix

# check if it exists
if [ -f $OUTPUT_PATH/$SAVE_FOLDER_NAME/geno.GCTA.grm.details ]; then
    echo "Kinship matrix already exists"
else
    $LDAK --calc-kins-direct $OUTPUT_PATH/$SAVE_FOLDER_NAME/geno.GCTA \
        --sped $SPED_FILE \
        --SNP-data NO \
        --ignore-weights YES \
        --power -1 \
        --max-threads $N_CORES \
        --keep $OUTPUT_PATH/individuals_intersect.txt
fi



# Perform single SNP analysis

SS_ANALYSIS_DONE=false
# check if single SNP analysis already done by checking
# 1) if $OUTPUT_PATH/$SAVE_FOLDER_NAME/"$PHENO.quant.progress" exists
if [ -f $OUTPUT_PATH/$SAVE_FOLDER_NAME/"$PHENO.quant.progress" ]; then
    # and 2) if the last line in the file states that X is equal to Y. The last line is on the form: "Performing single-SNP analysis for Chunk X of Y"
    echo "Single SNP analysis already started"
    LAST_LINE=$(tail -n 1 $OUTPUT_PATH/$SAVE_FOLDER_NAME/"$PHENO.quant.progress")
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

    $LDAK --linear $OUTPUT_PATH/$SAVE_FOLDER_NAME/"$PHENO.quant" \
        --sped $SPED_FILE \
        --pheno $PHENO_PATH/$PHENO \
        --mpheno 1 \
        --keep $OUTPUT_PATH/individuals_intersect.txt \
        --max-threads $N_CORES
fi

# perform REML analysis
if [ -f $OUTPUT_PATH/$SAVE_FOLDER_NAME/"$PHENO.reml1.reml" ]; then
    echo "REML analysis already done"
else
    echo "Performing REML analysis for $PHENO"
    $LDAK --reml $OUTPUT_PATH/$SAVE_FOLDER_NAME/"$PHENO.reml1" \
        --sped $SPED_FILE \
        --pheno $PHENO_PATH/$PHENO \
        --mpheno 1 \
        --grm $OUTPUT_PATH/$SAVE_FOLDER_NAME/geno.GCTA \
        --keep $OUTPUT_PATH/individuals_intersect.txt \
        --max-threads $N_CORES
fi
