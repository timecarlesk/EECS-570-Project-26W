#!/usr/bin/env python3
"""Evaluate prediction accuracy (MAPE) using measured speedup CSV."""

from __future__ import division

import argparse
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
    parser.add_argument(
        "--measured",
        required=True,
        help=(
            "CSV columns: workload,gpu,problem_size,stage,measured_speedup "
            "[,tile_size]"
        ),
    )
    return parser.parse_args()


def as_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def safe_mape(pred, meas):
    denom = meas if abs(meas) > 1e-9 else 1.0
    return abs(pred - meas) / abs(denom)


def print_group_mape(name, group):
    if not group:
        return
    mape = sum(group) / float(len(group)) * 100.0
    print("{}: n={} MAPE={:.3f}%".format(name, len(group), mape))


def main():
    args = parse_args()
    rows = read_csv(args.measured)

    all_errors = []
    by_gpu = {}
    by_workload = {}

    for row in rows:
        workload = row["workload"].strip()
        gpu = row["gpu"].strip()
        size = int(row["problem_size"])
        stage = int(row["stage"])
        measured = as_float(row["measured_speedup"])
        tile_size = row.get("tile_size", "")
        tile_size = int(tile_size) if str(tile_size).strip() else None

        pred = predict_one(workload, gpu, size, stage, tile_size=tile_size)
        pred_speedup = as_float(pred.get("pred_speedup", 0.0))
        err = safe_mape(pred_speedup, measured)

        all_errors.append(err)
        by_gpu.setdefault(gpu, []).append(err)
        by_workload.setdefault(workload, []).append(err)

        print(
            "{} {} size={} S={} measured={:.6f} pred={:.6f} APE={:.4f}".format(
                workload,
                gpu,
                size,
                stage,
                measured,
                pred_speedup,
                err,
            )
        )

    print_group_mape("Overall", all_errors)
    for gpu in sorted(by_gpu.keys()):
        print_group_mape("GPU=" + gpu, by_gpu[gpu])
    for workload in sorted(by_workload.keys()):
        print_group_mape("Workload=" + workload, by_workload[workload])


if __name__ == "__main__":
    main()
