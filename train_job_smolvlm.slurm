#!/bin/bash
#SBATCH --job-name=train_ColSmolVLM
#SBATCH --output=logs/train_ColSmolVLM/%A_%a.out
#SBATCH --error=logs/train_ColSmolVLM/%A_%a.err
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --gpus=2
#SBATCH --time=5:00:00

cd /fsx/arthur_bresnu/projects/colpali
source .venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs/train_ColSmolVLM

CONFIG_FILE="scripts/configs/idefics/train_colsmolvlm-256_reproduction.yaml"

# Run training with accelerate
accelerate launch --main_process_port 0 scripts/train/train_colbert.py ${CONFIG_FILE} 