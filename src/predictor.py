"""Parameter-free cp.async speedup predictor from the proposal."""

from __future__ import division

import math

from gpu_specs import get_gpu_spec
from workloads import make_workload_point


def smem_per_sm_bytes(gpu_spec):
    return float(gpu_spec["smem_per_sm_kb"]) * 1024.0


def l2_capacity_bytes(gpu_spec, l2_mode="effective"):
    mode = str(l2_mode).strip().lower()
    if mode == "nominal":
        return float(gpu_spec["l2_nominal_mb"]) * 1024.0 * 1024.0
    return float(gpu_spec["l2_eff_mb"]) * 1024.0 * 1024.0


def compute_blocks_per_sm(gpu_spec, w_block_bytes, stage):
    if stage < 1:
        return 0
    block_smem = float(w_block_bytes) * float(stage)
    if block_smem <= 0:
        return 0
    by_smem = int(smem_per_sm_bytes(gpu_spec) // block_smem)
    return min(int(gpu_spec.get("max_blocks_per_sm", 32)), by_smem)


def compute_w_conc(workload_point, gpu_spec, stage):
    w_block = float(workload_point["w_block_bytes"])
    blocks_per_sm = compute_blocks_per_sm(gpu_spec, w_block, stage)
    b_active = int(gpu_spec["n_sm"]) * int(blocks_per_sm)
    concurrent_blocks = min(int(workload_point["grid_blocks"]), b_active)
    w_conc = w_block * float(stage) * float(concurrent_blocks)
    return {
        "w_conc_bytes": float(w_conc),
        "blocks_per_sm": int(blocks_per_sm),
        "b_active": int(b_active),
        "concurrent_blocks": int(concurrent_blocks),
    }


def compute_compute_cycles(workload_point, gpu_spec):
    flops = float(workload_point["flops_per_tile"])
    clock_hz = float(gpu_spec["sm_clock_ghz"]) * 1e9
    n_sm = float(gpu_spec["n_sm"])
    flops_per_cycle_per_sm = (float(gpu_spec["fp32_tflops"]) * 1e12) / (clock_hz * n_sm)
    if flops_per_cycle_per_sm <= 0:
        return float("inf")
    full_tile_cycles = flops / flops_per_cycle_per_sm

    # cp.async overlap is bounded by per-stage compute window, not full tile lifetime.
    overlap_chunks = max(1, int(workload_point.get("overlap_chunks", 1)))
    return full_tile_cycles / float(overlap_chunks)


def predict_speedup(workload_point, gpu_spec, stage, l2_mode="effective"):
    stage = int(stage)

    if stage < 1:
        return {
            "valid": False,
            "reason": "stage_must_be_positive",
            "pred_speedup": 0.0,
        }

    if stage > 1 and not bool(gpu_spec["supports_cp_async"]):
        return {
            "valid": False,
            "reason": "gpu_has_no_cp_async",
            "pred_speedup": 1.0,
            "stage": stage,
            "w_conc_bytes": 0.0,
            "w_conc_over_l2": 0.0,
            "hit_fraction": 1.0,
            "l_eff_cycles": float(gpu_spec["l2_latency_cycles"]),
            "t_compute_cycles": 0.0,
            "s_min": 1,
            "occ_ratio": 1.0,
        }

    base = compute_w_conc(workload_point, gpu_spec, 1)
    cur = compute_w_conc(workload_point, gpu_spec, stage)

    if base["blocks_per_sm"] <= 0:
        return {
            "valid": False,
            "reason": "invalid_baseline_occupancy",
            "pred_speedup": 0.0,
        }

    if cur["blocks_per_sm"] <= 0:
        return {
            "valid": False,
            "reason": "stage_exceeds_smem",
            "pred_speedup": 0.0,
            "stage": stage,
        }

    w_conc = cur["w_conc_bytes"]
    l2_bytes = l2_capacity_bytes(gpu_spec, l2_mode=l2_mode)
    if w_conc <= 0:
        hit_fraction = 1.0
    else:
        hit_fraction = min(1.0, l2_bytes / w_conc)

    l2_latency = float(gpu_spec["l2_latency_cycles"])
    dram_latency = float(gpu_spec["dram_latency_cycles"])
    l_eff = hit_fraction * l2_latency + (1.0 - hit_fraction) * dram_latency

    t_compute = compute_compute_cycles(workload_point, gpu_spec)
    if t_compute <= 0:
        s_min = 1
    else:
        s_min = int(math.ceil(l_eff / t_compute))
        s_min = max(1, s_min)

    hidden_depth = min(stage, s_min)
    numerator = max(t_compute, l_eff)
    denominator = max(t_compute, l_eff / float(hidden_depth))

    overlap_term = numerator / denominator if denominator > 0 else 1.0
    occ_ratio_raw = cur["blocks_per_sm"] / float(base["blocks_per_sm"])
    occ_alpha = float(gpu_spec.get("occupancy_alpha", 1.0))
    occ_ratio = occ_ratio_raw ** occ_alpha

    pred_speedup = overlap_term * occ_ratio

    return {
        "valid": True,
        "reason": "ok",
        "pred_speedup": float(pred_speedup),
        "stage": stage,
        "w_conc_bytes": float(w_conc),
        "w_conc_over_l2": float(w_conc / l2_bytes) if l2_bytes > 0 else float("inf"),
        "hit_fraction": float(hit_fraction),
        "l_eff_cycles": float(l_eff),
        "t_compute_cycles": float(t_compute),
        "s_min": int(s_min),
        "occ_ratio": float(occ_ratio),
        "occ_ratio_raw": float(occ_ratio_raw),
        "blocks_per_sm": int(cur["blocks_per_sm"]),
        "concurrent_blocks": int(cur["concurrent_blocks"]),
        "b_active": int(cur["b_active"]),
        "l2_mode": str(l2_mode).strip().lower(),
    }


def predict_one(workload, gpu_name, problem_size, stage, tile_size=None, l2_mode="effective"):
    gpu_spec = get_gpu_spec(gpu_name)
    point = make_workload_point(workload, problem_size, tile_size=tile_size)
    out = predict_speedup(point, gpu_spec, stage, l2_mode=l2_mode)
    out["gpu"] = gpu_name
    out["workload"] = workload
    out["problem_size"] = int(problem_size)
    out["tile_size"] = int(point["tile_size"])
    out["arithmetic_intensity"] = float(point["arithmetic_intensity"])
    out["w_block_bytes"] = float(point["w_block_bytes"])
    out["grid_blocks"] = int(point["grid_blocks"])
    return out
