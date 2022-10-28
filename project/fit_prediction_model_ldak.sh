# fit a prediction model using LDAK

# VARIABLES
LDAK='/home/kce/NLPPred/github/snip/ldak/ldak5.2.linux'
OUTPUT_PATH='/home/kce/NLPPred/github/snip/models/ldak_outputs'
GENO_PATH='/home/kce/dsmwpred/data/ukbb/geno'
PHENO_PATH='/home/kce/dsmwpred/data/ukbb/height.train'
N_CORES=16

# select 20k IDs from the training set height data
head -n 20000 $PHENO_PATH > $OUTPUT_PATH/20k_subset.txt

# perform single snp analysis
# https://dougspeed.com/single-predictor-analysis/

$LDAK --linear $OUTPUT_PATH/quant \
    --bfile $GENO_PATH \
    --pheno $PHENO_PATH \
    --keep $OUTPUT_PATH/20k_subset.txt \
    --mpheno 1 \
    --max-threads $N_CORES

# perform Per-Predictor Heritabilities asuming the GCTA model
# https://dougspeed.com/per-predictor-heritabilities/
$LDAK --calc-tagging gcta \
    --bfile $GENO_PATH \
    --ignore-weights YES \
    --power -1 \
    --window-cm 1 \
    --save-matrix YES \

$LDAK --sum-hers gcta \
    --tagfile $OUTPUT_PATH/gcta.tagging \
    --summmary $OUTPUT_PATH/quant.summaries \
    --matrix $OUTPUT_PATH/gcta.matrix

# potentially remove highly correlated SNPs

# Construct prediction model
$LDAK --mega-prs megabayesr \
    --model bayesr \
    --ind.hers $OUTPUT_PATH/gcta.ind.hers \
    --summmary $OUTPUT_PATH/quant.summaries \
    --cv.proportion 0.1 \
    # --cors  # neglecting to remove correlated variables for now.
    # --high-LD # no high LD in chrom 22
    # --allow-ambiguous YES \ 
    --window-cm 1 
