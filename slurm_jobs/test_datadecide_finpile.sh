#!/bin/bash

#SBATCH --job-name=test_datadecide_finpile
#SBATCH --output=logs/test_datadecide_finpile_%j.out
#SBATCH --error=logs/test_datadecide_finpile_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --qos=inferno
#SBATCH --account=gts-schava6-fy20phase3

# Job description: Test DataDecide curation with FinPile data

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

# Load modules and activate environment
module purge
module load anaconda3

# Navigate to project directory
cd /storage/coda1/p-schava6/0/shared/DataDecider

# This would test DataDecide curation, but we need raw text data for curation
# The finpile data is already tokenized, so this is more of a placeholder
# for future work where we have raw text data to curate

echo "Note: DataDecide curation requires raw text data"
echo "FinPile data is already tokenized, so skipping curation test"
echo "This job template is ready for when we have raw text data to curate"

# For now, just test that our DataDecideCurator can be imported
uv run python -c "
from data_decide.olmo.data.data_curation import DataDecideCurator
print('✓ DataDecideCurator imports successfully')
print('✓ Ready for raw text data curation when available')
"

echo "Job completed at: $(date)"