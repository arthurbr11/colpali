#!/bin/bash
#SBATCH --job-name=vidore_eval
#SBATCH --array=0-3
#SBATCH --output=logs/vidore_eval/%A_%a.out
#SBATCH --error=logs/vidore_eval/%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1

cd /fsx/arthur_bresnu/projects/colpali
source .venv/bin/activate

mkdir -p logs/vidore_eval

# Define model and processor directories for each array task
declare -A MODEL_DIRS=(
    [0]="/models/ColSmolVLM-Instruct-256M_exp_bs64_lr5e-4_gpu2"
    [1]="/models/ColSmolDocling-256M-preview_exp_bs64_lr1e-3_gpu4"
    [2]="/models/ColSmolDocling-256M-preview_exp_bs64_lr3e-4_gpu4"
    [3]="/models/ColSmolDocling-256M-preview_exp_bs64_lr5e-4_gpu4"
)

declare -A PROCESSOR_DIRS=(
    [0]="vidore/ColSmolVLM-Instruct-256M-base"
    [1]="arthurbresnu/ColSmolDocling-256M-preview-base"
    [2]="arthurbresnu/ColSmolDocling-256M-preview-base"
    [3]="arthurbresnu/ColSmolDocling-256M-preview-base"
)

# Get the model and processor directories for this array task
MODEL_DIR=${MODEL_DIRS[$SLURM_ARRAY_TASK_ID]}
PROCESSOR_DIR=${PROCESSOR_DIRS[$SLURM_ARRAY_TASK_ID]}

# Set output directory
OUTPUT_DIR="${MODEL_DIR}/metrics"

# Run the evaluation script
python evaluate_vidore2.py \
    --model_dir "${MODEL_DIR}" \
    --processor_dir "${PROCESSOR_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_query 128 \
    --batch_passage 128 \
    --batch_score 128 