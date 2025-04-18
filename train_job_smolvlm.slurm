#!/bin/bash
#SBATCH --job-name=train_ColSmolVLM
#SBATCH --output=logs/train_ColSmolVLM/%A_%a.out
#SBATCH --error=logs/train_ColSmolVLM/%A_%a.err
#SBATCH --gpus=2
#SBATCH --partition=hopper-prod
#SBATCH --time=5:00:00
#SBATCH --qos=normal

source /admin/home/arthur_bresnu/.bashrc

cd /fsx/arthur_bresnu/projects/colpali
source .venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

CONFIG_FILE="scripts/configs/idefics/train_colsmolvlm-256_reproduction.yaml"

# Run training with accelerate
accelerate launch scripts/train/train_colbert.py ${CONFIG_FILE} 
