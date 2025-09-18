#!/bin/bash
#SBATCH --job-name=filter_pref_tuning
#SBATCH --partition=nlprx-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:8
#SBATCH --mem=512GB
#SBATCH --qos="short"
#SBATCH --output=logs/filter_pref_tuning_%j.out
#SBATCH --error=logs/filter_pref_tuning_%j.err

FILTERED_PATH="/srv/nlprx-lab/share6/jpark3272/arxiv_src/ml_domain"
PREPROCESSING_DS_PATH="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/databases2/pref_tuning_filtered"

mkdir -p "$PREPROCESSING_DS_PATH"
mkdir -p "logs"


source /nethome/kkocherla3/miniconda3/etc/profile.d/conda.sh
conda activate pipeline_env

python extractor/preprocess/broad_pref_filter.py \
    --extracted_ml_path $FILTERED_PATH \
    --ml_table_ds $PREPROCESSING_DS_PATH