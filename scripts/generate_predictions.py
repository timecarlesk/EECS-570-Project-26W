#!/usr/bin/env python3
"""Generate prediction table for GEMM or stencil sweeps."""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from gpu_specs import DEFAULT_STAGE_CANDIDATES, list_gpu_names
from io_utils import write_csv
from predictor import predict_one
from workloads import default_problem_sizes


def parse_int_list(raw):
    if raw is None or raw.strip() == "":
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, choices=["gemm", "stencil"])
    parser.add_argument(
        "--gpus",
        default="all",
        help="Comma separated GPU names. Use 'all' for every GPU in table.",
    )
    parser.add_argument(
        "--stages",
        default=",".join(str(s) for s in DEFAULT_STAGE_CANDIDATES),
        help="Comma separated stage values.",
    )
    parser.add_argument(
        "--problem-sizes",
        default="",
        help="Comma separated problem sizes. Empty uses default sweep from proposal.",
    )
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--l2-mode", choices=["effective", "nominal"], default="effective")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpus.strip().lower() == "all":
        gpus = list_gpu_names()
    else:
        gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]

    stages = parse_int_list(args.stages)
    if not stages:
        stages = list(DEFAULT_STAGE_CANDIDATES)

    sizes = parse_int_list(args.problem_sizes)
    if not sizes:
        sizes = default_problem_sizes(args.workload)

    rows = []
    for gpu in gpus:
        for problem_size in sizes:
            for stage in stages:
                pred = predict_one(
                    args.workload,
                    gpu,
                    problem_size,
                    stage,
                    tile_size=args.tile_size,
                    l2_mode=args.l2_mode,
                )
                rows.append(
                    {
                        "workload": pred["workload"],
                        "gpu": pred["gpu"],
                        "problem_size": pred["problem_size"],
                        "tile_size": pred["tile_size"],
                        "stage": pred.get("stage", stage),
                        "valid": pred.get("valid", False),
                        "reason": pred.get("reason", ""),
                        "pred_speedup": "{:.6f}".format(float(pred.get("pred_speedup", 0.0))),
                        "arithmetic_intensity": "{:.6f}".format(float(pred.get("arithmetic_intensity", 0.0))),
                        "w_block_bytes": "{:.1f}".format(float(pred.get("w_block_bytes", 0.0))),
                        "w_conc_bytes": "{:.1f}".format(float(pred.get("w_conc_bytes", 0.0))),
                        "w_conc_over_l2": "{:.6f}".format(float(pred.get("w_conc_over_l2", 0.0))),
                        "hit_fraction": "{:.6f}".format(float(pred.get("hit_fraction", 0.0))),
                        "l_eff_cycles": "{:.3f}".format(float(pred.get("l_eff_cycles", 0.0))),
                        "t_compute_cycles": "{:.3f}".format(float(pred.get("t_compute_cycles", 0.0))),
                        "s_min": pred.get("s_min", 0),
                        "occ_ratio": "{:.6f}".format(float(pred.get("occ_ratio", 0.0))),
                        "occ_ratio_raw": "{:.6f}".format(float(pred.get("occ_ratio_raw", 0.0))),
                        "blocks_per_sm": pred.get("blocks_per_sm", 0),
                        "concurrent_blocks": pred.get("concurrent_blocks", 0),
                        "b_active": pred.get("b_active", 0),
                        "l2_mode": pred.get("l2_mode", args.l2_mode),
                    }
                )

    fieldnames = [
        "workload",
        "gpu",
        "problem_size",
        "tile_size",
        "stage",
        "valid",
        "reason",
        "pred_speedup",
        "arithmetic_intensity",
        "w_block_bytes",
        "w_conc_bytes",
        "w_conc_over_l2",
        "hit_fraction",
        "l_eff_cycles",
        "t_compute_cycles",
        "s_min",
        "occ_ratio",
        "occ_ratio_raw",
        "blocks_per_sm",
        "concurrent_blocks",
        "b_active",
        "l2_mode",
    ]
    write_csv(args.output, rows, fieldnames)

    print("Wrote {} rows to {}".format(len(rows), args.output))


if __name__ == "__main__":
    main()
