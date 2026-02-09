#!/usr/bin/env python3
"""Rank largest prediction-vs-measurement deviations and report top-k anomalies."""

from __future__ import division

import argparse
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from io_utils import read_csv
from predictor import predict_one


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--l2-mode", choices=["effective", "nominal"], default="effective")
    parser.add_argument("--ncu-dir", default="")
    parser.add_argument("--output", default=os.path.join(ROOT, "outputs", "anomaly_topk.csv"))
    return parser.parse_args()


def as_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def as_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def find_ncu_matches(ncu_dir, workload, problem_size, stage, tile_size):
    if not ncu_dir:
        return []
    if workload == "gemm":
        pattern = os.path.join(ncu_dir, "*gemm*N{}*T{}*S{}*.csv".format(problem_size, tile_size, stage))
    else:
        pattern = os.path.join(ncu_dir, "*stencil*L{}*S{}*.csv".format(problem_size, stage))
    return sorted(glob.glob(pattern))


def main():
    args = parse_args()
    rows = read_csv(args.measured)

    scored = []
    for row in rows:
        workload = str(row.get("workload", "")).strip().lower()
        gpu = str(row.get("gpu", "")).strip()
        problem_size = as_int(row.get("problem_size", "0"))
        stage = as_int(row.get("stage", "0"))
        measured = as_float(row.get("measured_speedup", "0"))
        tile_raw = str(row.get("tile_size", "")).strip()
        tile_size = as_int(tile_raw) if tile_raw else None

        pred = predict_one(
            workload,
            gpu,
            problem_size,
            stage,
            tile_size=tile_size,
            l2_mode=args.l2_mode,
        )
        predicted = as_float(pred.get("pred_speedup", 0.0))

        delta = predicted - measured
        abs_delta = abs(delta)

        matches = find_ncu_matches(
            args.ncu_dir,
            workload=workload,
            problem_size=problem_size,
            stage=stage,
            tile_size=tile_size if tile_size is not None else 0,
        )

        scored.append(
            {
                "workload": workload,
                "gpu": gpu,
                "problem_size": problem_size,
                "stage": stage,
                "tile_size": tile_size if tile_size is not None else "",
                "measured_speedup": measured,
                "pred_speedup": predicted,
                "delta": delta,
                "abs_delta": abs_delta,
                "ncu_matches": "|".join(matches),
            }
        )

    scored = sorted(scored, key=lambda x: x["abs_delta"], reverse=True)
    topk = scored[: max(1, int(args.top_k))]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        fieldnames = [
            "workload",
            "gpu",
            "problem_size",
            "stage",
            "tile_size",
            "measured_speedup",
            "pred_speedup",
            "delta",
            "abs_delta",
            "ncu_matches",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in topk:
            writer.writerow(row)

    print("Top-{} anomalies (l2_mode={}):".format(len(topk), args.l2_mode))
    for i, row in enumerate(topk, 1):
        print(
            "{}. {} {} size={} S={} tile={} measured={:.6f} pred={:.6f} delta={:.6f} abs={:.6f}".format(
                i,
                row["workload"],
                row["gpu"],
                row["problem_size"],
                row["stage"],
                row["tile_size"],
                row["measured_speedup"],
                row["pred_speedup"],
                row["delta"],
                row["abs_delta"],
            )
        )
        if row["ncu_matches"]:
            print("   ncu: {}".format(row["ncu_matches"]))

    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()
