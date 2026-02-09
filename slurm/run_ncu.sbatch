#!/bin/bash
#SBATCH --job-name=cpasync_ncu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

module purge
module load cuda
module load cmake

PROJECT_ROOT="/scratch/eecs570w26s001_class_root/eecs570w26s001_class/xxxchen/project2_cp_async_pipeline"
cd "${PROJECT_ROOT}"
mkdir -p outputs/nsight slurm

# Build binaries and run Nsight Compute data collection for selected points.
python3 scripts/run_cuda_benchmarks.py --run-gemm --run-stencil
python3 scripts/run_nsight_profiling.py --build-dir build --output-dir outputs/nsight
