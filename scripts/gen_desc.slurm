#!/bin/bash
#SBATCH --job-name=generate_description
#SBATCH --partition=nlprx-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=512GB
#SBATCH --qos="debug"
#SBATCH --output=logs_%A/generate_description%a.out
#SBATCH --error=logs_%A/generate_description%a.err
#SBATCH --array=0-0

TOTAL_SHARDS=1
SHARD_IDX=$SLURM_ARRAY_TASK_ID

CACHE_DIR="/srv/nlprx-lab/share6/kkocherla3/.cache/huggingface"
TARGET_MODEL_EXTRACTED_BIBTEX_CANONICALIZED_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/bibtex_ds"
TARGET_MODEL_EXTRACTED_DESCRIPTION_GENERATED_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/desc_ds"

mkdir -p "$CACHE_DIR"
mkdir -p "$TARGET_MODEL_EXTRACTED_DESCRIPTION_GENERATED_DS"

VLLM_USE_V1=0
export VLLM_USE_V1

source /nethome/kkocherla3/miniconda3/etc/profile.d/conda.sh
conda activate pipeline_env

python extractor/generate_description/generate_description.py \
    --hf_ds_path $TARGET_MODEL_EXTRACTED_BIBTEX_CANONICALIZED_DS \
    --hf_ds_output_path $TARGET_MODEL_EXTRACTED_DESCRIPTION_GENERATED_DS \
    --cache_dir "$CACHE_DIR" \
    --shard_idx "$SHARD_IDX" \
    --total_shards "$TOTAL_SHARDS"