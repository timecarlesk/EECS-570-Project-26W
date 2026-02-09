#!/usr/bin/env python3
"""Convert raw benchmark CSV (V1/V3 timings) into measured speedup CSV.

Input CSV columns (from CUDA runners):
  workload,gpu,problem_size,variant,stage,tile_size,time_ms,...

Output CSV columns (for evaluate_mape.py):
  workload,gpu,problem_size,stage,tile_size,measured_speedup
"""

import argparse
import csv
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        required=True,
        help="Comma-separated raw CSV paths (e.g., outputs/gemm_raw.csv,outputs/stencil_raw.csv)",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def as_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def as_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def main():
    args = parse_args()
    raw_paths = [x.strip() for x in args.raw.split(",") if x.strip()]

    rows = []
    for path in raw_paths:
        with open(path, "r") as f:
            rows.extend(list(csv.DictReader(f)))

    groups = defaultdict(list)
    for row in rows:
        key = (
            row.get("workload", "").strip().lower(),
            row.get("gpu", "").strip(),
            as_int(row.get("problem_size", "0")),
            as_int(row.get("tile_size", "0")),
        )
        groups[key].append(row)

    out_rows = []
    for key, group in groups.items():
        workload, gpu, problem_size, tile_size = key

        baseline = None
        for row in group:
            variant = row.get("variant", "").strip().upper()
            stage = as_int(row.get("stage", "0"))
            if variant == "V1" and stage == 1:
                baseline = row
                break

        if baseline is None:
            continue

        baseline_ms = as_float(baseline.get("time_ms", "0"))
        if baseline_ms <= 0:
            continue

        out_rows.append(
            {
                "workload": workload,
                "gpu": gpu,
                "problem_size": str(problem_size),
                "stage": "1",
                "tile_size": str(tile_size),
                "measured_speedup": "1.000000",
            }
        )

        for row in group:
            variant = row.get("variant", "").strip().upper()
            stage = as_int(row.get("stage", "0"))
            if variant != "V3" or stage < 2:
                continue

            t_ms = as_float(row.get("time_ms", "0"))
            if t_ms <= 0:
                continue

            speedup = baseline_ms / t_ms
            out_rows.append(
                {
                    "workload": workload,
                    "gpu": gpu,
                    "problem_size": str(problem_size),
                    "stage": str(stage),
                    "tile_size": str(tile_size),
                    "measured_speedup": "{:.6f}".format(speedup),
                }
            )

    out_rows = sorted(
        out_rows,
        key=lambda r: (
            r["workload"],
            r["gpu"],
            int(r["problem_size"]),
            int(r["stage"]),
        ),
    )

    dedup = {}
    for row in out_rows:
        key = (
            row["workload"],
            row["gpu"],
            row["problem_size"],
            row["stage"],
            row["tile_size"],
        )
        dedup[key] = row
    out_rows = [dedup[k] for k in sorted(dedup.keys())]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        fieldnames = [
            "workload",
            "gpu",
            "problem_size",
            "stage",
            "tile_size",
            "measured_speedup",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print("Wrote {} rows to {}".format(len(out_rows), args.output))


if __name__ == "__main__":
    main()
