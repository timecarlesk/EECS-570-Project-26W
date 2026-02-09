#!/bin/bash
#SBATCH --job-name=cpasync_a100_mig
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

module purge
module load cuda
module load cmake

# NOTE:
# This script assumes the allocated A100 is configured as MIG 3g.40gb by the cluster setup.
# If your cluster uses a dedicated MIG partition/constraint, adjust SBATCH directives accordingly.

PROJECT_ROOT="/scratch/eecs570w26s001_class_root/eecs570w26s001_class/xxxchen/project2_cp_async_pipeline"
cd "${PROJECT_ROOT}"
mkdir -p outputs slurm

python3 scripts/run_cuda_benchmarks.py \
  --run-gemm --run-stencil --run-pointer \
  --gpu-name A100_MIG_3g40gb \
  --run-v0 \
  --gemm-output outputs/gemm_raw_a100_mig.csv \
  --stencil-output outputs/stencil_raw_a100_mig.csv \
  --pointer-output outputs/pointer_chase_raw_a100_mig.csv

python3 scripts/extract_l2_params.py --input outputs/pointer_chase_raw_a100_mig.csv --output-json outputs/l2_params_a100_mig.json
python3 scripts/run_model_validation.py \
  --gpu-list A100_MIG_3g40gb \
  --gemm-raw outputs/gemm_raw_a100_mig.csv \
  --stencil-raw outputs/stencil_raw_a100_mig.csv \
  --measured-out outputs/measured_speedup_a100_mig.csv \
  --compare-l2-modes \
  --leave-one-out
