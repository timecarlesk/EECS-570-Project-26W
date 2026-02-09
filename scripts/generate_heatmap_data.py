#!/usr/bin/env python3
"""Generate pipeline benefit map data (AI vs Wconc/L2, colored by best speedup)."""

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
from workloads import GEMM_N_SWEEP, TILE_SWEEP


def parse_int_list(raw):
    if raw is None or raw.strip() == "":
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--problem-sizes", default="")
    parser.add_argument("--tile-sizes", default="")
    parser.add_argument(
        "--stages",
        default=",".join(str(s) for s in DEFAULT_STAGE_CANDIDATES),
    )
    parser.add_argument("--l2-mode", choices=["effective", "nominal"], default="effective")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpus.strip().lower() == "all":
        gpus = list_gpu_names()
    else:
        gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]

    sizes = parse_int_list(args.problem_sizes) or list(GEMM_N_SWEEP)
    tiles = parse_int_list(args.tile_sizes) or list(TILE_SWEEP)
    stages = parse_int_list(args.stages) or list(DEFAULT_STAGE_CANDIDATES)

    rows = []
    for gpu in gpus:
        for tile in tiles:
            for n in sizes:
                best = None
                for s in stages:
                    pred = predict_one("gemm", gpu, n, s, tile_size=tile, l2_mode=args.l2_mode)
                    if best is None or float(pred.get("pred_speedup", 0.0)) > float(best.get("pred_speedup", 0.0)):
                        best = pred

                rows.append(
                    {
                        "gpu": gpu,
                        "problem_size": n,
                        "tile_size": tile,
                        "arithmetic_intensity": "{:.6f}".format(float(best["arithmetic_intensity"])),
                        "w_conc_over_l2": "{:.6f}".format(float(best.get("w_conc_over_l2", 0.0))),
                        "best_stage": int(best.get("stage", 1)),
                        "best_pred_speedup": "{:.6f}".format(float(best.get("pred_speedup", 1.0))),
                        "l2_mode": best.get("l2_mode", args.l2_mode),
                    }
                )

    fieldnames = [
        "gpu",
        "problem_size",
        "tile_size",
        "arithmetic_intensity",
        "w_conc_over_l2",
        "best_stage",
        "best_pred_speedup",
        "l2_mode",
    ]
    write_csv(args.output, rows, fieldnames)
    print("Wrote {} rows to {}".format(len(rows), args.output))


if __name__ == "__main__":
    main()
