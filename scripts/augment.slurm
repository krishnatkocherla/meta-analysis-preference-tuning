#!/bin/bash
#SBATCH --job-name=augment
#SBATCH --partition=nlprx-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=512GB
#SBATCH --qos="debug"
#SBATCH --output=logs_%A/augment%a.out
#SBATCH --error=logs_%A/augment%a.err
#SBATCH --array=0-0 # total 2 shards, adjust as needed

TOTAL_SHARDS=1
SHARD_IDX=$SLURM_ARRAY_TASK_ID

INPUT_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/table_schema_think"
CACHE_DIR="/srv/nlprx-lab/share6/kkocherla3/.cache/huggingface"
EXTRACTED_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/augmented_schema"

mkdir -p "$CACHE_DIR"
mkdir -p "$EXTRACTED_DS"

VLLM_USE_V1=0
export VLLM_USE_V1

source /nethome/kkocherla3/miniconda3/etc/profile.d/conda.sh
conda activate pipeline_env

python extractor/extract/context_augment.py \
    --hf_ds_path "$INPUT_DS" \
    --hf_ds_output_path "$EXTRACTED_DS/shard_${SHARD_IDX}" \
    --cache_dir "$CACHE_DIR" \
    --shard_idx "$SHARD_IDX" \
    --total_shards "$TOTAL_SHARDS"