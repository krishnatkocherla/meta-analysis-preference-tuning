#!/bin/bash
#SBATCH --job-name=references
#SBATCH --partition=nlprx-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=512GB
#SBATCH --qos="debug"
#SBATCH --output=logs_%A/references%a.out
#SBATCH --error=logs_%A/references%a.err
#SBATCH --array=0-0 # total 2 shards, adjust as needed

TOTAL_SHARDS=1
SHARD_IDX=$SLURM_ARRAY_TASK_ID

TARGET_MODEL_EXTRACTED_AUGMENTED_FILTERED_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/augmented_schema"
TARGET_MODEL_EXTRACTED_BIBTEX_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/bibtex_ds"
ARXIV_CITED_DOWNLOAD_PATH="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/arxiv_cited"
DOWNLOADED_ARXIV_SOURCE_PATH="/srv/nlprx-lab/share6/jpark3272/arxiv_src/ml_domain/"

mkdir -p "$TARGET_MODEL_EXTRACTED_BIBTEX_DS"
mkdir -p "$ARXIV_CITED_DOWNLOAD_PATH"

source /nethome/kkocherla3/miniconda3/etc/profile.d/conda.sh
conda activate pipeline_env

python extractor/generate_description/get_bibtex_paper.py \
    --hf_ds_path $TARGET_MODEL_EXTRACTED_AUGMENTED_FILTERED_DS \
    --hf_ds_output_path $TARGET_MODEL_EXTRACTED_BIBTEX_DS \
    --source_path $DOWNLOADED_ARXIV_SOURCE_PATH \
    --source_save_path $ARXIV_CITED_DOWNLOAD_PATH