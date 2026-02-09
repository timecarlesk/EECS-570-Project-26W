# Project 2 Code Implementation (Based on `proposal_rq2.tex`)

This directory turns the engineering-ready parts of the proposal into runnable code, with focus on:

1. A parameter-free prediction model (Section 7)
2. `W_block / W_conc` and L2-normalized analysis (Section 5)
3. A Triton autotune `num_stages` pre-filter (Section 8)
4. GEMM / Stencil sweep export and MAPE evaluation (Section 4/7)

## Directory Structure

- `src/gpu_specs.py`: GPU parameter table (A40/A100_MIG/H100/V100)
- `src/workloads.py`: GEMM and 1D stencil working-set and AI definitions
- `src/predictor.py`: prediction model implementation
- `src/triton_prefilter.py`: `num_stages` pruning logic
- `scripts/generate_predictions.py`: generate sweep prediction CSV files
- `scripts/best_stage_report.py`: generate `S*(N)` report
- `scripts/evaluate_mape.py`: compute MAPE from measured speedup CSV
- `scripts/demo_prefilter.py`: demo script for pre-filter behavior
- `scripts/generate_heatmap_data.py`: generate pipeline benefit map data (AI vs `W_conc/L2`)
- `data/sample_measurements.csv`: sample measured-data format
- `tests/test_predictor.py`: basic unit tests

## Quick Start

Run the following inside `project2_cp_async_pipeline`:

```bash
python3 scripts/generate_predictions.py --workload gemm --gpus all --output outputs/gemm_predictions.csv
python3 scripts/generate_predictions.py --workload stencil --gpus all --output outputs/stencil_predictions.csv
python3 scripts/best_stage_report.py --input outputs/gemm_predictions.csv --output outputs/gemm_best_stage.csv
python3 scripts/generate_heatmap_data.py --gpus all --output outputs/heatmap_data.csv
python3 scripts/demo_prefilter.py --workload gemm --gpus all --epsilon 0.03
python3 scripts/evaluate_mape.py --measured data/sample_measurements.csv
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Prediction Model Equations

- `h = min(1, C_L2_eff / W_conc)`
- `L_eff = h * L_L2 + (1-h) * L_DRAM`
- `S_min = ceil(L_eff / T_compute)`
- `pred_speedup = overlap_term * Occ(S)/Occ(1)`

Implementation: `src/predictor.py`.

## Parameters You Should Replace with Real Measurements

The following values in `gpu_specs.py` are placeholders and should be replaced by your Week 1-2 microbenchmark results:

- `l2_eff_mb`
- `l2_latency_cycles`
- `dram_latency_cycles`
- (optional) `sm_clock_ghz`

After replacing them, rerun `generate_predictions.py` and `evaluate_mape.py`.

## How to Integrate the Triton Pre-Filter

`src/triton_prefilter.py` provides:

- `prune_num_stages(workload, gpu_name, problem_size, candidate_stages, epsilon, tile_size)`

It returns `kept/pruned/scores`. Call this before Triton autotune and pass only the kept `num_stages` candidates to real benchmarking.

## Notes

The current environment does not have `nvcc`, so this version delivers the runnable analysis/prediction pipeline first.
If needed, I can next add CUDA kernel skeletons (V0/V1/V3 GEMM + Stencil + pointer-chasing microbenchmark) and organize them with your cluster build scripts.
