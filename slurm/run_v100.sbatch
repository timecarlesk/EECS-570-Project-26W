#!/bin/bash
#SBATCH --job-name=cpasync_v100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
  --gpu-name V100 \
  --run-v0 \
  --gemm-output outputs/gemm_raw_v100.csv \
  --stencil-output outputs/stencil_raw_v100.csv \
  --pointer-output outputs/pointer_chase_raw_v100.csv

python3 scripts/extract_l2_params.py --input outputs/pointer_chase_raw_v100.csv --output-json outputs/l2_params_v100.json
python3 scripts/run_model_validation.py \
  --gpu-list V100 \
  --gemm-raw outputs/gemm_raw_v100.csv \
  --stencil-raw outputs/stencil_raw_v100.csv \
  --measured-out outputs/measured_speedup_v100.csv
