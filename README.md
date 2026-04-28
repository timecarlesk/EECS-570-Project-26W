# Project : cp.async Pipeline (Implementation)

This repo implements the full workflow described in `proposal_rq2.tex`:

1. CUDA benchmarks (GEMM / stencil / pointer-chasing)
2. Microbenchmark-based architecture constants
3. Parameter-free prediction model
4. Validation (MAPE, leave-one-out, nominal vs effective L2)
5. Visualization and anomaly analysis
6. Triton pre-filter integration scaffold

## Implemented Components

### CUDA benchmarks

- `cuda/src/benchmark_gemm.cu`
  - `V0` naive GEMM
  - `V1` shared-memory tiled GEMM
  - `V3` cp.async pipelined GEMM (`num_stages` 2/3/4)
  - `Ref` cuBLAS SGEMM baseline
  - Tile-size sweep support: `--tile-sizes 16,32,64,128`
- `cuda/src/benchmark_stencil.cu`
  - `V1`/`V3` 1D stencil with halo, `num_stages` 2/3/4
- `cuda/src/microbench_pointer_chase.cu`
  - Pointer-chasing latency sweep (`cycles/load`, `ns/load`)
  - Streaming bandwidth curve (`stream_bandwidth_gbps`)

### Python model / analysis

- `src/predictor.py`
  - overlap + occupancy model
  - `l2_mode` support: `effective` vs `nominal`
- `scripts/evaluate_mape.py`
  - per-GPU / per-workload MAPE
  - leave-one-out report
  - nominal-vs-effective L2 comparison
- `scripts/plot_results.py`
  - Speedup vs size
  - Speedup vs `W_conc/C_L2`
  - `S*(N)`
  - Speedup-rank vs ridge-rank
  - Predicted-vs-measured scatter
  - Pipeline benefit heatmap
- `scripts/anomaly_analysis.py`
  - `delta = predicted - measured`, top-k by `|delta|`
- `scripts/run_nsight_profiling.py`
  - Nsight Compute CLI collection for selected `(size, stage, tile)` points

### Triton integration

- `src/triton_prefilter.py`
  - stage pruning logic
  - Triton decorator integration via `autotune_with_cp_async_prefilter(...)`
- `triton/gemm_kernel.py`
  - minimal demo kernel scaffold using the pre-filter decorator

### SLURM

- `slurm/run_v100.sbatch`
- `slurm/run_a40.sbatch`
- `slurm/run_a100_mig.sbatch`
- `slurm/run_h100.sbatch`
- `slurm/run_ncu.sbatch`
- `slurm/run_full_pipeline.sbatch`

Per-GPU scripts write GPU-specific output filenames (for example `outputs/gemm_raw_a40.csv`) to avoid overwrite when collecting data across multiple machines.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run Benchmarks

### One-command run

```bash
python3 scripts/run_cuda_benchmarks.py --run-gemm --run-stencil --run-pointer --run-v0
```

Key options:

- `--gemm-tile-sizes 16,32,64,128`
- `--gemm-stages 2,3,4`
- `--gemm-n-values 1024,2048,...`
- `--stencil-lengths 65536,262144,...`
- `--gpu-name A40` (or custom label)

## Extract L2 / latency constants

```bash
python3 scripts/extract_l2_params.py \
  --input outputs/pointer_chase_raw.csv \
  --output-json outputs/l2_params.json
```

Use the suggested values to update `src/gpu_specs.py`:

- `l2_eff_mb`
- `l2_latency_cycles`
- `dram_latency_cycles`

## Convert raw benchmark data to measured speedup

```bash
python3 scripts/build_measured_speedup_csv.py \
  --raw outputs/gemm_raw.csv,outputs/stencil_raw.csv \
  --output outputs/measured_speedup.csv
```

## Validation

```bash
python3 scripts/run_model_validation.py --compare-l2-modes --leave-one-out
```

Equivalent direct command:

```bash
python3 scripts/evaluate_mape.py \
  --measured outputs/measured_speedup.csv \
  --compare-l2-modes \
  --leave-one-out
```

## Plotting

```bash
python3 scripts/plot_results.py \
  --measured outputs/measured_speedup.csv \
  --output-dir outputs/figures
```

## Nsight + anomaly workflow

```bash
python3 scripts/run_nsight_profiling.py --build-dir build --output-dir outputs/nsight
python3 scripts/anomaly_analysis.py --measured outputs/measured_speedup.csv --ncu-dir outputs/nsight
```

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

- `numpy`
- `pandas`
- `matplotlib`
- `triton` (optional, gated by Python version marker)

## Notes

- cp.async kernels are only meaningful on SM80+ (Ampere/Hopper).
- V100 naturally acts as boundary case (`V3` not run).
- In this current coding environment, `nvcc` is not available, so CUDA binaries were not compiled here; Python scripts/tests were validated.
