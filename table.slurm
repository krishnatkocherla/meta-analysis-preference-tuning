#!/bin/bash
#SBATCH --job-name=filter_pref_tuning
#SBATCH --partition=nlprx-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:8
#SBATCH --mem=512GB
#SBATCH --qos="long"
#SBATCH --output=logs_%A/table_filter_%a.out
#SBATCH --error=logs_%A/table_filter_%a.err
#SBATCH --array=0-7  # total 2 shards, adjust as needed

TOTAL_SHARDS=8
SHARD_IDX=$SLURM_ARRAY_TASK_ID

LEADERBOARD_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/databases2/pref_tables_qwen"
CACHE_DIR="/srv/nlprx-lab/share6/kkocherla3/.cache/huggingface"
PREPROCESSING_DS_PATH="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/databases2/pref_tuning_filtered"

mkdir -p "$CACHE_DIR"
mkdir -p "$LEADERBOARD_DS"

source /nethome/kkocherla3/miniconda3/etc/profile.d/conda.sh
conda activate pipeline_env

python extractor/filter/filter_leaderboard_table.py \
    --ml_table_ds "$PREPROCESSING_DS_PATH" \
    --ml_leaderboard_table_ds "$LEADERBOARD_DS/shard_${SHARD_IDX}" \
    --cache_dir "$CACHE_DIR" \
    --shard_idx "$SHARD_IDX" \
    --total_shards "$TOTAL_SHARDS"