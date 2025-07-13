#!/bin/bash

# Helper script to submit SLURM jobs for DataDecider FinPile integration

show_usage() {
    echo "Usage: $0 <job-type> [additional-args]"
    echo ""
    echo "Available job types:"
    echo "  test-dataloader    - Test FinPile data loader (30 min, CPU only)"
    echo "  test-training      - Test model training (1 hour, 1x V100)"
    echo "  create-subsamples  - Create larger subsamples (2 hours, CPU only)"
    echo "  train-olmo        - Train OLMo model (requires: data-path model-size max-steps)"
    echo "  test-datadecide   - Test DataDecide integration (2 hours, CPU only)"
    echo "  proxy-experiments - Run DataDecide proxy experiments (2 hours, CPU only)"
    echo "  full-scale        - Run full-scale training comparison (4 hours, 1x V100)"
    echo ""
    echo "Examples:"
    echo "  $0 test-dataloader"
    echo "  $0 test-training"
    echo "  $0 create-subsamples"
    echo "  $0 train-olmo data/finpile_subsamples/finpile_small 4M 1000"
    echo "  $0 test-datadecide"
    echo "  $0 proxy-experiments"
    echo "  $0 full-scale"
}

if [ "$#" -lt 1 ]; then
    show_usage
    exit 1
fi

JOB_TYPE="$1"
shift  # Remove job type from arguments

# Ensure logs directory exists
mkdir -p logs

case "$JOB_TYPE" in
    "test-dataloader")
        echo "Submitting FinPile data loader test job..."
        sbatch slurm_jobs/test_finpile_dataloader.sh
        ;;
    "test-training")
        echo "Submitting FinPile training test job..."
        sbatch slurm_jobs/test_finpile_training.sh
        ;;
    "create-subsamples")
        echo "Submitting subsample creation job..."
        sbatch slurm_jobs/create_subsamples.sh
        ;;
    "train-olmo")
        if [ "$#" -ne 3 ]; then
            echo "Error: train-olmo requires 3 arguments: data-path model-size max-steps"
            echo "Example: $0 train-olmo data/finpile_subsamples/finpile_small 4M 1000"
            exit 1
        fi
        echo "Submitting OLMo training job with args: $@"
        sbatch slurm_jobs/train_olmo_finpile.sh "$@"
        ;;
    "test-datadecide")
        echo "Submitting DataDecide test job..."
        sbatch slurm_jobs/test_datadecide_finpile.sh
        ;;
    "proxy-experiments")
        echo "Submitting DataDecide proxy experiments job..."
        sbatch slurm_jobs/run_proxy_experiments.sh
        ;;
    "full-scale")
        echo "Submitting full-scale training comparison job..."
        sbatch slurm_jobs/run_full_scale_training.sh
        ;;
    *)
        echo "Error: Unknown job type '$JOB_TYPE'"
        show_usage
        exit 1
        ;;
esac

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/<job_name>_<jobid>.out"