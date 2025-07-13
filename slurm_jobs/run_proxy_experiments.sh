#!/bin/bash
#SBATCH --job-name=proxy_experiments
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --account=gts-schava6-fy20phase3
#SBATCH --output=logs/proxy_experiments_%j.out
#SBATCH --error=logs/proxy_experiments_%j.err

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

echo "Running DataDecide proxy experiments..."

uv run python create_proxy_training_pipeline.py

echo "Job completed at: $(date)"