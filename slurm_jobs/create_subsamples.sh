#!/bin/bash

#SBATCH --job-name=create_finpile_subsamples
#SBATCH --output=logs/create_subsamples_%j.out
#SBATCH --error=logs/create_subsamples_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --qos=inferno
#SBATCH --account=gts-schava6-fy20phase3

# Job description: Create larger subsamples from FinPile dataset

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

# Load modules and activate environment
module purge
module load anaconda3

# Navigate to project directory
cd /storage/coda1/p-schava6/0/shared/DataDecider

# Check if source data exists
SOURCE_DATA="/storage/coda1/p-schava6/0/shared/finpile/datamixes/0fp-100dolma"
if [ ! -f "${SOURCE_DATA}.bin" ]; then
    echo "ERROR: Source data not found at ${SOURCE_DATA}.bin"
    exit 1
fi

echo "Creating FinPile subsamples..."
echo "Source data: ${SOURCE_DATA}"
echo "Target directory: data/finpile_subsamples/"

# Create subsamples of different sizes
uv run python -c "
import sys
sys.path.append('.')
from data_decide.utils.subsample_finpile import create_subsample
from pathlib import Path

output_dir = Path('data/finpile_subsamples')
output_dir.mkdir(parents=True, exist_ok=True)

# Create progressively larger subsamples
configs = [
    ('small', 100),      # ~400K tokens
    ('medium', 1000),    # ~4M tokens  
    ('large', 10000),    # ~40M tokens
]

source_prefix = '${SOURCE_DATA}'

for name, num_chunks in configs:
    target_prefix = output_dir / f'finpile_{name}'
    print(f'Creating {name} subsample ({num_chunks:,} chunks)...')
    
    try:
        total_tokens, total_docs = create_subsample(
            source_prefix=source_prefix,
            target_prefix=str(target_prefix),
            num_chunks=num_chunks,
            chunk_size=4096,
            start_chunk=0,
            verbose=True
        )
        print(f'✓ {name}: {total_tokens:,} tokens, {total_docs:,} docs')
    except Exception as e:
        print(f'✗ {name}: Failed - {e}')
"

echo "Job completed at: $(date)"