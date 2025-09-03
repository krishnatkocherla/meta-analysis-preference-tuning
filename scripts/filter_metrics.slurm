#!/bin/bash
#SBATCH --job-name=filter_metrics
#SBATCH --partition=nlprx-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=512GB
#SBATCH --qos="short"
#SBATCH --output=logs_%A/filter_metrics%a.out
#SBATCH --error=logs_%A/filter_metrics%a.err
#SBATCH --array=0-3 # total 2 shards, adjust as needed

TOTAL_SHARDS=4
SHARD_IDX=$SLURM_ARRAY_TASK_ID

INPUT_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/augmented_schema"
CACHE_DIR="/srv/nlprx-lab/share6/kkocherla3/.cache/huggingface"
EXTRACTED_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/filtered_schema"

mkdir -p "$CACHE_DIR"
mkdir -p "$EXTRACTED_DS"

VLLM_USE_V1=0
export VLLM_USE_V1

source /nethome/kkocherla3/miniconda3/etc/profile.d/conda.sh
conda activate pipeline_env

python extractor/extract/filter_and_adjust_metrics.py \
    --hf_ds_path "$INPUT_DS" \
    --hf_ds_output_path "$EXTRACTED_DS/shard_${SHARD_IDX}" \
    --cache_dir "$CACHE_DIR" \
    --shard_idx "$SHARD_IDX" \
    --total_shards "$TOTAL_SHARDS"