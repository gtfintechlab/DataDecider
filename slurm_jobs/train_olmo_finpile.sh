#!/bin/bash

#SBATCH --job-name=train_olmo_finpile
#SBATCH --output=logs/train_olmo_finpile_%j.out
#SBATCH --error=logs/train_olmo_finpile_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --qos=inferno
#SBATCH --account=gts-schava6-fy20phase3

# Job description: Train OLMo model with FinPile data

# Parse command line arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <data-path> <model-size> <max-steps>"
    echo "Example: $0 data/finpile_subsamples/finpile_small 4M 1000"
    exit 1
fi

DATA_PATH="$1"
MODEL_SIZE="$2"
MAX_STEPS="$3"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Data path: $DATA_PATH"
echo "Model size: $MODEL_SIZE"
echo "Max steps: $MAX_STEPS"
echo "GPU info:"
nvidia-smi

# Load modules and activate environment
module purge
module load anaconda3

# Navigate to project directory
cd /storage/coda1/p-schava6/0/shared/DataDecider

# Check if data exists
if [ ! -f "${DATA_PATH}.bin" ]; then
    echo "ERROR: Data not found at ${DATA_PATH}.bin"
    exit 1
fi

echo "Starting OLMo training with FinPile data..."

# Create a training config for this run
CONFIG_FILE="configs/training/finpile_${MODEL_SIZE}_slurm.yaml"
mkdir -p configs/training

cat > "$CONFIG_FILE" << EOF
# OLMo training config for FinPile data on SLURM
model_size: "$MODEL_SIZE"
sequence_length: 2048
batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1e-4
warmup_steps: 100
max_steps: $MAX_STEPS
eval_interval: 200
save_interval: 500
fp16: true
optimizer: "adamw"
weight_decay: 0.1
beta1: 0.9
beta2: 0.95
EOF

echo "Created config file: $CONFIG_FILE"

# Run training
uv run python data_decide/scripts/train.py \\
    --data-path "$DATA_PATH" \\
    --training-config "$CONFIG_FILE" \\
    --output-dir "outputs/finpile_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)" \\
    --resume-from-checkpoint

echo "Job completed at: $(date)"