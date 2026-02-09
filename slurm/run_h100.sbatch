#!/bin/bash
#SBATCH --job-name=cpasync_h100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

module purge
module load cuda
module load cmake

PROJECT_ROOT="/scratch/eecs570w26s001_class_root/eecs570w26s001_class/xxxchen/project2_cp_async_pipeline"
cd "${PROJECT_ROOT}"
mkdir -p outputs slurm

python3 scripts/run_cuda_benchmarks.py \
  --run-gemm --run-stencil --run-pointer \
  --gpu-name H100_SXM5 \
  --run-v0 \
  --gemm-output outputs/gemm_raw_h100.csv \
  --stencil-output outputs/stencil_raw_h100.csv \
  --pointer-output outputs/pointer_chase_raw_h100.csv

python3 scripts/extract_l2_params.py --input outputs/pointer_chase_raw_h100.csv --output-json outputs/l2_params_h100.json
python3 scripts/run_model_validation.py \
  --gpu-list H100_SXM5 \
  --gemm-raw outputs/gemm_raw_h100.csv \
  --stencil-raw outputs/stencil_raw_h100.csv \
  --measured-out outputs/measured_speedup_h100.csv \
  --compare-l2-modes \
  --leave-one-out
