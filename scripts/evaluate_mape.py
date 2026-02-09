#!/usr/bin/env python3
"""Evaluate prediction accuracy (MAPE) using measured speedup CSV.

Supports:
- L2 mode selection: effective vs nominal
- nominal-vs-effective comparison
- leave-one-out (held-out GPU) reporting
"""

from __future__ import division

import argparse
import csv
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
    parser.add_argument("--l2-mode", choices=["effective", "nominal"], default="effective")
    parser.add_argument("--compare-l2-modes", action="store_true")
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument("--output-details", default="")
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


def safe_mape(pred, meas):
    denom = meas if abs(meas) > 1e-9 else 1.0
    return abs(pred - meas) / abs(denom)


def summarize_errors(errors):
    if not errors:
        return 0.0
    return (sum(errors) / float(len(errors))) * 100.0


def evaluate_rows(rows, l2_mode):
    details = []
    all_errors = []
    by_gpu = {}
    by_workload = {}

    for row in rows:
        workload = row["workload"].strip()
        gpu = row["gpu"].strip()
        size = as_int(row["problem_size"])
        stage = as_int(row["stage"])
        measured = as_float(row["measured_speedup"])
        tile_size_raw = row.get("tile_size", "")
        tile_size = as_int(tile_size_raw) if str(tile_size_raw).strip() else None

        pred = predict_one(workload, gpu, size, stage, tile_size=tile_size, l2_mode=l2_mode)
        pred_speedup = as_float(pred.get("pred_speedup", 0.0))
        err = safe_mape(pred_speedup, measured)

        all_errors.append(err)
        by_gpu.setdefault(gpu, []).append(err)
        by_workload.setdefault(workload, []).append(err)

        details.append(
            {
                "workload": workload,
                "gpu": gpu,
                "problem_size": size,
                "stage": stage,
                "tile_size": tile_size if tile_size is not None else "",
                "measured_speedup": measured,
                "pred_speedup": pred_speedup,
                "ape": err,
                "l2_mode": l2_mode,
            }
        )

    summary = {
        "overall_mape": summarize_errors(all_errors),
        "n": len(all_errors),
        "by_gpu": {g: summarize_errors(v) for g, v in sorted(by_gpu.items())},
        "by_workload": {w: summarize_errors(v) for w, v in sorted(by_workload.items())},
    }
    return details, summary


def print_summary(name, details, summary, verbose_rows=True):
    if verbose_rows:
        for d in details:
            print(
                "{} {} size={} S={} measured={:.6f} pred={:.6f} APE={:.4f} [{}]".format(
                    d["workload"],
                    d["gpu"],
                    d["problem_size"],
                    d["stage"],
                    d["measured_speedup"],
                    d["pred_speedup"],
                    d["ape"],
                    d["l2_mode"],
                )
            )

    print("{}: n={} MAPE={:.3f}%".format(name, summary["n"], summary["overall_mape"]))
    for gpu in sorted(summary["by_gpu"].keys()):
        print("GPU={} [{}] MAPE={:.3f}%".format(gpu, name, summary["by_gpu"][gpu]))
    for workload in sorted(summary["by_workload"].keys()):
        print(
            "Workload={} [{}] MAPE={:.3f}%".format(
                workload,
                name,
                summary["by_workload"][workload],
            )
        )


def run_leave_one_out(rows, l2_mode):
    # The predictor is parameter-free (no fitted params from kernel data),
    # so leave-one-out evaluates each GPU using only its arch constants.
    gpus = sorted(set(r["gpu"].strip() for r in rows if r.get("gpu", "").strip()))
    print("\nLeave-one-out report [{}]:".format(l2_mode))
    for held_gpu in gpus:
        held_rows = [r for r in rows if r["gpu"].strip() == held_gpu]

        _, summary = evaluate_rows(held_rows, l2_mode=l2_mode)
        print(
            "held_out_gpu={} test_n={} held_out_MAPE={:.3f}%".format(
                held_gpu,
                summary["n"],
                summary["overall_mape"],
            )
        )


def write_details(path, details):
    fieldnames = [
        "workload",
        "gpu",
        "problem_size",
        "stage",
        "tile_size",
        "measured_speedup",
        "pred_speedup",
        "ape",
        "l2_mode",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in details:
            writer.writerow(d)
    print("Wrote detail rows to {}".format(path))


def main():
    args = parse_args()
    rows = read_csv(args.measured)

    modes = [args.l2_mode]
    if args.compare_l2_modes:
        modes = ["effective", "nominal"]

    all_details = []
    all_summaries = {}

    for mode in modes:
        details, summary = evaluate_rows(rows, l2_mode=mode)
        all_details.extend(details)
        all_summaries[mode] = summary
        print_summary("mode=" + mode, details, summary, verbose_rows=not args.compare_l2_modes)

        if args.leave_one_out:
            run_leave_one_out(rows, l2_mode=mode)

    if args.compare_l2_modes:
        eff = all_summaries.get("effective", {}).get("overall_mape", 0.0)
        nom = all_summaries.get("nominal", {}).get("overall_mape", 0.0)
        delta = nom - eff
        print("\nL2 mode comparison: nominal-effective delta = {:.3f}%".format(delta))

    if args.output_details:
        write_details(args.output_details, all_details)


if __name__ == "__main__":
    main()
