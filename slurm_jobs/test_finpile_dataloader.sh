#!/bin/bash

#SBATCH --job-name=test_finpile_loader
#SBATCH --output=logs/test_finpile_loader_%j.out
#SBATCH --error=logs/test_finpile_loader_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --qos=inferno
#SBATCH --account=gts-schava6-fy20phase3

# Job description: Test FinPile data loader functionality

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

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

# Run the data loader test
echo "Testing FinPile data loader..."
uv run python test_finpile_dataloader.py

echo "Job completed at: $(date)"