#!/bin/bash
#SBATCH --job-name=cpasync_l40s
#SBATCH --account=mcity_project
#SBATCH --partition=mcity_project
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

module purge
module load cuda
module load cmake

PROJECT_ROOT="/scratch/mcity_project_root/mcity_project/xxxchen/EECS_570/project2_cp_async_pipeline"
cd "${PROJECT_ROOT}"
mkdir -p outputs/l40s slurm

python3 scripts/run_cuda_benchmarks.py \
  --run-gemm --run-stencil --run-pointer \
  --gpu-name L40S \
  --run-v0 \
  --gemm-output outputs/l40s/gemm_raw.csv \
  --stencil-output outputs/l40s/stencil_raw.csv \
  --pointer-output outputs/l40s/pointer_chase_raw.csv

python3 scripts/extract_l2_params.py --input outputs/l40s/pointer_chase_raw.csv --output-json outputs/l40s/l2_params.json
python3 scripts/run_model_validation.py \
  --gpu-list L40S \
  --gemm-raw outputs/l40s/gemm_raw.csv \
  --stencil-raw outputs/l40s/stencil_raw.csv \
  --measured-out outputs/l40s/measured_speedup.csv \
  --compare-l2-modes \
  --leave-one-out
