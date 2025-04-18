#!/bin/bash
#SBATCH --job-name=train_ColSmolDocling
#SBATCH --output=logs/train_ColSmolDocling/%A_%a.out
#SBATCH --error=logs/train_ColSmolDocling/%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --gpus=4
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --array=0-2  # Array of 3 tasks (0,1,2)


source /admin/home/arthur_bresnu/.bashrc

cd /fsx/arthur_bresnu/projects/colpali
source .venv/bin/activate


# Create logs directory if it doesn't exist
mkdir -p logs

# Get number of GPUs from SLURM
N_GPU=$SLURM_GPUS_ON_NODE

# Define arrays for batch sizes and learning rates
BATCH_SIZES=(64 64 64)
LEARNING_RATES=(5e-4 1e-3 3e-4) 

# Get the parameters for this array task
BATCH_SIZE=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}
LR=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}

# Create run name and output directory based on parameters
RUN_NAME="ColSmolDocling-256M-preview_exp_bs${BATCH_SIZE}_lr${LR}_gpu${N_GPU}"
OUTPUT_DIR="models/ColSmolDocling-256M-preview_exp_bs${BATCH_SIZE}_lr${LR}_gpu${N_GPU}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Base config file
CONFIG_FILE="scripts/configs/idefics/train_colsmoldocling_model_original.yaml"

# Create temporary config file
TEMP_CONFIG="scripts/configs/idefics/train_colsmoldocling_model_bs${BATCH_SIZE}_lr${LR}_gpu${N_GPU}.yaml"

# Use sed to replace parameters in config
sed -e "s/per_device_train_batch_size: [0-9]*/per_device_train_batch_size: ${BATCH_SIZE}/" \
    -e "s/per_device_eval_batch_size: [0-9]*/per_device_eval_batch_size: ${BATCH_SIZE}/" \
    -e "s/learning_rate: [0-9.e-]*/learning_rate: ${LR}/" \
    -e "s/output_dir: !path [^ ]*/output_dir: !path ${OUTPUT_DIR}/" \
    -e "s/run_name: [^ ]*/run_name: ${RUN_NAME}/" \
    ${CONFIG_FILE} > ${TEMP_CONFIG}

# Run training with accelerate
accelerate launch scripts/train/train_colbert.py ${TEMP_CONFIG} 

# Clean up temporary config
rm ${TEMP_CONFIG} 