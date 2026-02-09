#!/bin/bash
#SBATCH --job-name=cpasync_project2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

# Example environment setup; adjust to your cluster.
module purge
module load cuda
module load cmake

PROJECT_ROOT="/scratch/eecs570w26s001_class_root/eecs570w26s001_class/xxxchen/project2_cp_async_pipeline"
cd "${PROJECT_ROOT}"

mkdir -p outputs slurm

python3 scripts/run_cuda_benchmarks.py --run-gemm --run-stencil --run-pointer --run-v0
python3 scripts/extract_l2_params.py --input outputs/pointer_chase_raw.csv --output-json outputs/l2_params.json
python3 scripts/run_model_validation.py --compare-l2-modes --leave-one-out
python3 scripts/plot_results.py --measured outputs/measured_speedup.csv --output-dir outputs/figures
python3 scripts/anomaly_analysis.py --measured outputs/measured_speedup.csv --output outputs/anomaly_topk.csv
