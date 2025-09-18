#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --partition=nlprx-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=512GB
#SBATCH --qos="short"
#SBATCH --output=logs_%A/extract%a.out
#SBATCH --error=logs_%A/extract%a.err
#SBATCH --array=0-3 # total 2 shards, adjust as needed

TOTAL_SHARDS=4
SHARD_IDX=$SLURM_ARRAY_TASK_ID

TARGET_MODEL_LEADERBOARD_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/databases2/pref_tables_qwen"
CACHE_DIR="/srv/nlprx-lab/share6/kkocherla3/.cache/huggingface"
TARGET_MODEL_EXTRACTED_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/databases2/table_schema_think"

mkdir -p "$CACHE_DIR"
mkdir -p "$TARGET_MODEL_EXTRACTED_DS"

VLLM_USE_V1=0
export VLLM_USE_V1

source /nethome/kkocherla3/miniconda3/etc/profile.d/conda.sh
conda activate pipeline_env

python extractor/extract/schema_extract_target_model.py \
    --hf_ds_path "$TARGET_MODEL_LEADERBOARD_DS" \
    --hf_ds_output_path "$TARGET_MODEL_EXTRACTED_DS/shard_${SHARD_IDX}" \
    --cache_dir "$CACHE_DIR" \
    --shard_idx "$SHARD_IDX" \
    --total_shards "$TOTAL_SHARDS"