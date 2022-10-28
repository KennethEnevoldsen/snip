#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem 32g
#SBATCH -c 16
#SBATCH --output ./project/reports/slurm-output/%x-%u-%j.out
#SBATCH -A NLPPred
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com

DATA_FOLDER=/home/kce/NLPPred/github/snip/data/compressed

# loop trought each subfolder
for folder in $DATA_FOLDER/act*; do
    # loop through each .zarr file in the subfolder
    for file in $folder/*.zarr; do
        # get the file name without the extension
        file_name=$(basename $file .zarr)
        echo "Calling function:"
        echo "snip convert $file $folder/$file_name.sped"
        # convert the .zarr file to .sped
        snip convert $file $folder/$file_name.sped --overwrite
    done
done
