#!/usr/bin/env python3
"""Estimate L2 effective capacity / L2 latency / DRAM latency from pointer-chasing CSV."""

import argparse
import csv
import json
import math
from statistics import median


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="pointer_chase_raw.csv")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Capacity knee threshold between L2 and DRAM plateaus (default=0.1)",
    )
    parser.add_argument("--output-json", default="")
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


def take_quantile_slice(values, q_lo, q_hi):
    if not values:
        return []
    n = len(values)
    lo = max(0, min(n - 1, int(math.floor((n - 1) * q_lo))))
    hi = max(lo + 1, min(n, int(math.ceil((n - 1) * q_hi)) + 1))
    return values[lo:hi]


def main():
    args = parse_args()

    with open(args.input, "r") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("Empty input CSV")

    by_gpu = {}
    for row in rows:
        gpu = row.get("gpu", "unknown").strip()
        by_gpu.setdefault(gpu, []).append(row)

    summary = {}

    for gpu, gpu_rows in by_gpu.items():
        pts = []
        bw_pts = []
        for row in gpu_rows:
            size_bytes = as_int(row.get("size_bytes", "0"))
            cycles = as_float(row.get("cycles_per_load", "0"))
            ns = as_float(row.get("ns_per_load", "0"))
            if size_bytes > 0 and cycles > 0:
                pts.append((size_bytes, cycles, ns))
            bw = as_float(row.get("stream_bandwidth_gbps", "0"))
            if size_bytes > 0 and bw > 0:
                bw_pts.append((size_bytes, bw))

        pts = sorted(pts, key=lambda x: x[0])
        if len(pts) < 4:
            continue

        cycles_all = [p[1] for p in pts]
        ns_all = [p[2] for p in pts]
        sizes_all = [p[0] for p in pts]

        l2_cycles = min(take_quantile_slice(cycles_all, 0.0, 0.10))
        l2_ns = min(take_quantile_slice(ns_all, 0.0, 0.10))

        # Use the last 3 data points (largest arrays) for DRAM latency.
        # This is more robust than a fixed quantile when most data points
        # fall within L2 (e.g. L40S with 96 MB L2).
        n_dram = min(3, len(pts))
        dram_cycles = median([p[1] for p in pts[-n_dram:]])
        dram_ns = median([p[2] for p in pts[-n_dram:]])

        knee_value = l2_cycles + float(args.threshold) * (dram_cycles - l2_cycles)
        l2_eff_bytes = sizes_all[-1]
        for size_bytes, cycles, _ in pts:
            if cycles >= knee_value:
                l2_eff_bytes = size_bytes
                break

        summary[gpu] = {
            "l2_eff_mb": round(l2_eff_bytes / (1024.0 * 1024.0), 4),
            "l2_latency_cycles": round(l2_cycles, 4),
            "dram_latency_cycles": round(dram_cycles, 4),
            "l2_latency_ns": round(l2_ns, 4),
            "dram_latency_ns": round(dram_ns, 4),
            "threshold": float(args.threshold),
        }

        if bw_pts:
            bw_pts = sorted(bw_pts, key=lambda x: x[0])
            bw_vals = [x[1] for x in bw_pts]
            summary[gpu]["stream_bandwidth_peak_gbps"] = round(max(bw_vals), 4)
            summary[gpu]["stream_bandwidth_l2_gbps"] = round(
                median(take_quantile_slice(bw_vals, 0.0, 0.30)),
                4,
            )
            summary[gpu]["stream_bandwidth_dram_gbps"] = round(
                median(take_quantile_slice(bw_vals, 0.70, 1.0)),
                4,
            )

    if not summary:
        raise RuntimeError("No usable rows parsed from CSV")

    print(json.dumps(summary, indent=2, sort_keys=True))

    print("\nSuggested gpu_specs.py updates:")
    for gpu in sorted(summary.keys()):
        s = summary[gpu]
        print(
            "{} -> l2_eff_mb={}, l2_latency_cycles={}, dram_latency_cycles={}".format(
                gpu,
                s["l2_eff_mb"],
                s["l2_latency_cycles"],
                s["dram_latency_cycles"],
            )
        )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print("Wrote {}".format(args.output_json))


if __name__ == "__main__":
    main()
