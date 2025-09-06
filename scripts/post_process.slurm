#!/bin/bash
#SBATCH --job-name=post_process
#SBATCH --partition=nlprx-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=512GB
#SBATCH --qos="debug"
#SBATCH --output=logs_%A/post_process%a.out
#SBATCH --error=logs_%A/post_process%a.err
#SBATCH --array=0-0

TOTAL_SHARDS=1
SHARD_IDX=$SLURM_ARRAY_TASK_ID

CACHE_DIR="/srv/nlprx-lab/share6/kkocherla3/.cache/huggingface"
TARGET_MODEL_FINAL_EXTRACTED_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/databases/final_ds"
TARGET_MODEL_EXTRACTED_DESCRIPTION_EXTRACTED_DS="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/databases/extract_desc_ds"

mkdir -p "$CACHE_DIR"
mkdir -p "$TARGET_MODEL_EXTRACTED_DESCRIPTION_EXTRACTED_DS"

source /nethome/kkocherla3/miniconda3/etc/profile.d/conda.sh
conda activate pipeline_env

python -m extractor.generate_description.post_process \
    --hf_ds_path $TARGET_MODEL_EXTRACTED_DESCRIPTION_EXTRACTED_DS \
    --hf_ds_output_path $TARGET_MODEL_FINAL_EXTRACTED_DS \