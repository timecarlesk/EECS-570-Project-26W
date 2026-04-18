#!/bin/bash
#SBATCH --job-name=cpasync_h100
#SBATCH --account=mcity_project
#SBATCH --partition=mcity_project
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

module purge
module load cuda
module load cmake

PROJECT_ROOT="/scratch/mcity_project_root/mcity_project/xxxchen/EECS_570/project2_cp_async_pipeline"
cd "${PROJECT_ROOT}"
mkdir -p outputs/h100 slurm

python3 scripts/run_cuda_benchmarks.py \
  --run-gemm --run-stencil --run-pointer \
  --gpu-name H100_SXM5 \
  --run-v0 \
  --gemm-output outputs/h100/gemm_raw.csv \
  --stencil-output outputs/h100/stencil_raw.csv \
  --pointer-output outputs/h100/pointer_chase_raw.csv

python3 scripts/extract_l2_params.py --input outputs/h100/pointer_chase_raw.csv --output-json outputs/h100/l2_params.json
python3 scripts/run_model_validation.py \
  --gpu-list H100_SXM5 \
  --gemm-raw outputs/h100/gemm_raw.csv \
  --stencil-raw outputs/h100/stencil_raw.csv \
  --measured-out outputs/h100/measured_speedup.csv \
  --compare-l2-modes \
  --leave-one-out
