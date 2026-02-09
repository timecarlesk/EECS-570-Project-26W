#!/usr/bin/env python3
"""Find best stage S* for each (workload, gpu, problem_size)."""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from io_utils import read_csv, write_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Prediction CSV from generate_predictions.py")
    parser.add_argument("--output", default="", help="Optional output CSV path")
    return parser.parse_args()


def as_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def main():
    args = parse_args()
    rows = read_csv(args.input)

    best = {}
    for row in rows:
        key = (row["workload"], row["gpu"], int(row["problem_size"]))
        score = as_float(row.get("pred_speedup", "0"))
        valid = str(row.get("valid", "False")).lower() in ("true", "1", "yes")
        stage = int(row["stage"])

        # Prefer valid rows; among equals choose higher score.
        rank = (1 if valid else 0, score)
        if key not in best or rank > best[key]["rank"]:
            best[key] = {
                "rank": rank,
                "workload": row["workload"],
                "gpu": row["gpu"],
                "problem_size": int(row["problem_size"]),
                "best_stage": stage,
                "pred_speedup": "{:.6f}".format(score),
                "w_conc_over_l2": row.get("w_conc_over_l2", ""),
                "arithmetic_intensity": row.get("arithmetic_intensity", ""),
            }

    out_rows = [v for v in best.values()]
    out_rows = sorted(out_rows, key=lambda x: (x["workload"], x["gpu"], x["problem_size"]))

    for r in out_rows:
        print(
            "{workload:7s} {gpu:18s} size={problem_size:8d} S*={best_stage} pred={pred_speedup}".format(
                **r
            )
        )

    if args.output:
        out_clean = []
        for row in out_rows:
            clean = dict(row)
            clean.pop("rank", None)
            out_clean.append(clean)
        fieldnames = [
            "workload",
            "gpu",
            "problem_size",
            "best_stage",
            "pred_speedup",
            "w_conc_over_l2",
            "arithmetic_intensity",
        ]
        write_csv(args.output, out_clean, fieldnames)
        print("Wrote {} rows to {}".format(len(out_clean), args.output))


if __name__ == "__main__":
    main()
