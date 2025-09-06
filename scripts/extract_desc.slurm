#!/bin/bash
#SBATCH --job-name=extract_description
#SBATCH --partition=nlprx-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=512GB
#SBATCH --qos="debug"
#SBATCH --output=logs_%A/extract_description%a.out
#SBATCH --error=logs_%A/extract_description%a.err
#SBATCH --array=0-0

TOTAL_SHARDS=1
SHARD_IDX=$SLURM_ARRAY_TASK_ID

CACHE_DIR="/srv/nlprx-lab/share6/kkocherla3/.cache/huggingface"
TARGET_MODEL_EXTRACTED_DESCRIPTION_GENERATED_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/databases/desc_ds"
TARGET_MODEL_EXTRACTED_DESCRIPTION_EXTRACTED_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/databases/extract_desc_ds"
ARXIV_CITED_DOWNLOAD_PATH="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/arxiv_cited"
DOWNLOADED_ARXIV_SOURCE_PATH="/srv/nlprx-lab/share6/jpark3272/arxiv_src/ml_domain/"


mkdir -p "$CACHE_DIR"
mkdir -p "$TARGET_MODEL_EXTRACTED_DESCRIPTION_EXTRACTED_DS"

VLLM_USE_V1=0
export VLLM_USE_V1

source /nethome/kkocherla3/miniconda3/etc/profile.d/conda.sh
conda activate pipeline_env

python -m extractor.generate_description.extract_description \
    --hf_ds_path $TARGET_MODEL_EXTRACTED_DESCRIPTION_GENERATED_DS \
    --hf_ds_output_path $TARGET_MODEL_EXTRACTED_DESCRIPTION_EXTRACTED_DS \
    --orig_source_path $DOWNLOADED_ARXIV_SOURCE_PATH \
    --citation_source_path $ARXIV_CITED_DOWNLOAD_PATH \
    --cache_dir "$CACHE_DIR" \
    --shard_idx "$SHARD_IDX" \
    --total_shards "$TOTAL_SHARDS"