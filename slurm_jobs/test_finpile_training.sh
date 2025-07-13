#!/bin/bash

#SBATCH --job-name=test_finpile_training
#SBATCH --output=logs/test_finpile_training_%j.out
#SBATCH --error=logs/test_finpile_training_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --qos=inferno
#SBATCH --account=gts-schava6-fy20phase3

# Job description: Test model training with FinPile data

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "GPU info:"
nvidia-smi

# Load modules and activate environment
module purge
module load anaconda3

# Navigate to project directory
cd /storage/coda1/p-schava6/0/shared/DataDecider

# Ensure we have our tiny subsample
if [ ! -f "data/finpile_subsamples/finpile_tiny.bin" ]; then
    echo "Creating tiny subsample first..."
    uv run python create_tiny_subsample.py
fi

# Run the training test
echo "Testing FinPile model training..."
uv run python test_finpile_training.py

echo "Job completed at: $(date)"