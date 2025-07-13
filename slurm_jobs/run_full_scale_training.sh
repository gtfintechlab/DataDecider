#!/bin/bash
#SBATCH --job-name=full_scale_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-v100
#SBATCH --qos=inferno
#SBATCH --account=gts-schava6-fy20phase3
#SBATCH --output=logs/full_scale_training_%j.out
#SBATCH --error=logs/full_scale_training_%j.err

echo "---------------------------------------"
echo "Begin Slurm Prolog: $(date)"
echo "Job ID:    $SLURM_JOB_ID"
echo "User ID:   $USER" 
echo "Account:   $SLURM_JOB_ACCOUNT"
echo "Job name:  $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "QOS:       $SLURM_JOB_QOS"
echo "---------------------------------------"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

# GPU info
echo "GPU info:"
nvidia-smi

echo "Running DataDecide full-scale training comparison..."

# Run full-scale training with DataDecide methodology
uv run python train_with_best_recipe.py \
    --dataset-prefix data/finpile_subsamples/finpile_medium \
    --model-size 4M \
    --max-steps 1000 \
    --batch-size 8 \
    --output-dir results/full_scale_training

echo "Job completed at: $(date)"