#!/bin/bash
#SBATCH --job-name=filter_pref_tuning
#SBATCH --partition=nlprx-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --qos="short"
#SBATCH --output=logs/filter_pref_tuning_%j.out
#SBATCH --error=logs/filter_pref_tuning_%j.err


FILTERED_PATH="/srv/nlprx-lab/share6/jpark3272/arxiv_src/ml_domain"
PREPROCESSING_DS_PATH="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/pref_tuning_filtered"

mkdir -p logs
mkdir -p "$PREPROCESSING_DS_PATH"


# Run the script
# PREPROCESSING_DS_PATH=YOUR_PREPROCESSING_DS_PATH

python3 extractor/preprocess/filter_preference.py \
    --extracted_ml_path $FILTERED_PATH \
    --ml_table_ds $PREPROCESSING_DS_PATH