
# Estimate the heritability given the GCTA model using the LDAK package.
# Assuming it is run from /home/kce/NLPPred/github/snip

output_path=/home/kce/NLPPred/github/snip/models/ldak_outputs
pheno_path=/home/kce/dsmwpred/data/ukbb/height.train
sped_path=/home/kce/NLPPred/github/snip/data/compressed/c_snps_train

# check overlap in individuals between sped path .fam and pheno path
# 1. get the individual IDs from the sped path
cut -f 1,2 $sped_path.fam > $output_path/individuals_sped.txt
# 2. get the individual IDs from the pheno path
cut -f 1,2 $pheno_path > $output_path/individuals_pheno.txt
# 3. get the intersection of the two
comm -12 $output_path/individuals_sped.txt $output_path/individuals_pheno.txt > $output_path/individuals_overlap.txt
# 4. get the number of common IDs
echo "Number of common IDs:"
wc -l $output_path/individuals_overlap.txt
# 5. select 20k common IDs (dont shuffle)
shuf -n 20000 $output_path/individuals_ovxerlap.txt > $output_path/individuals_overlap_20k.txt


# Calculate kinship matrix assuming GCTA model
# http://dougspeed.com/calculate-kinships/
ldak/ldak5.2.linux --calc-kins-direct $output_path/gcta_kinship \
    --sped $sped_path \
    --ignore-weights YES \
    --SNP-data NO \
    --power -1 \
    --keep common_ids_20k.txt


# fit GCTA model using REML
# Follows this guide:
# https://dougspeed.com/reml-analysis/
ldak/ldak5.2.linux --reml $output_path/gcta_reml \
    --pheno $pheno_path \
    --grm $output_path/gcta_kinship