#!/bin/bash
#SBATCH --job-name=mpralegnet
#SBATCH --output=out/mpralegnet_%j.out
#SBATCH --error=out/mpralegnet_%j.err
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=96G
#SBATCH --qos=slow_nice
#SBATCH --partition=gpuq
#SBATCH --time=48:00:00


# Custom command to send notifications to Slack
source "$(which job_notify_slurm)"
notify_job_start

set -e

CONFIG_IDX=$1
if [ -z "$CONFIG_IDX" ]; then
    echo "Usage: $0 <config_index>"
    exit 1
fi
CONFIG_FILE="./configs/config${CONFIG_IDX}.json"

PROJ_DIR=/grid/koo/home/nagai/projects/continual_learning/insilico_mpra
PYTHON=${PROJ_DIR}/.venv/bin/python

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Project directory: $PROJ_DIR"

# Load modules or activate conda environment
# Adjust this to match your environment setup
# module load cuda/11.7  # Adjust CUDA version as needed

# Navigate to your project directory
cd "$PROJ_DIR"

$PYTHON train.py --config "$CONFIG_FILE"

# Custom command to send notifications to Slack
notify_job_end
