#!/bin/bash
#SBATCH --job-name=train_ColSmolVLM
#SBATCH --output=logs/train_ColSmolVLM%A_%a.out
#SBATCH --error=logs/train_ColSmolVLM%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:2  # Request 2 GPUs

source /admin/home/arthur_bresnu/.bashrc

cd /fsx/arthur_bresnu/projects/colpali
source .venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Get number of GPUs from SLURM
N_GPU=$SLURM_GPUS_ON_NODE

CONFIG_FILE="scripts/configs/idefics/train_colsmolvlm_model.yaml"

# Run training with accelerate
accelerate launch --multi_gpu --num_processes=${N_GPU} \
    scripts/train/train_colbert.py \
    ${CONFIG_FILE} 
