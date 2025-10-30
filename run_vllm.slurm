#!/bin/bash
#SBATCH --job-name=vllm_qwen
#SBATCH --partition=nlprx-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:8
#SBATCH --mem=128GB
#SBATCH --qos=debug
#SBATCH --output=logs/vllm_qwen_%j.out
#SBATCH --error=logs/vllm_qwen_%j.err

mkdir -p logs

# Load your conda environment
source /nethome/kkocherla3/miniconda3/etc/profile.d/conda.sh
conda activate pipeline_env

# Run vLLM serve with explicit device
VLLM_LOGGING_LEVEL=DEBUG vllm serve Qwen/Qwen3-32B \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --tensor-parallel-size 8 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.85 \
    --download-dir /nethome/kkocherla3/flash/.cache \
    --device cuda