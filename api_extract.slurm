#!/bin/bash
#SBATCH --job-name=filter_pref_tuning
#SBATCH --partition=nlprx-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=512GB
#SBATCH --qos="debug"
#SBATCH --output=logs/filter_pref_tuning_%j.out
#SBATCH --error=logs/filter_pref_tuning_%j.err

mkdir -p "logs"


source /nethome/kkocherla3/miniconda3/etc/profile.d/conda.sh
conda activate pipeline_env

python extractor/preprocess/bulk_api_filter.py