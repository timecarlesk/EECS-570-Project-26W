#!/usr/bin/env python3
"""Run prediction generation + measured-speedup conversion + MAPE evaluation."""

import argparse
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable or "python3"


def run(cmd):
    print("+ {}".format(" ".join(cmd)))
    subprocess.check_call(cmd, cwd=ROOT)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-list", default="all")
    parser.add_argument("--gemm-raw", default=os.path.join(ROOT, "outputs", "gemm_raw.csv"))
    parser.add_argument("--stencil-raw", default=os.path.join(ROOT, "outputs", "stencil_raw.csv"))
    parser.add_argument("--measured-out", default=os.path.join(ROOT, "outputs", "measured_speedup.csv"))
    parser.add_argument("--compare-l2-modes", action="store_true")
    parser.add_argument("--leave-one-out", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    run(
        [
            PY,
            "scripts/generate_predictions.py",
            "--workload",
            "gemm",
            "--gpus",
            args.gpu_list,
            "--output",
            "outputs/gemm_predictions.csv",
        ]
    )
    run(
        [
            PY,
            "scripts/generate_predictions.py",
            "--workload",
            "stencil",
            "--gpus",
            args.gpu_list,
            "--output",
            "outputs/stencil_predictions.csv",
        ]
    )

    raw_inputs = []
    if os.path.exists(args.gemm_raw):
        raw_inputs.append(args.gemm_raw)
    if os.path.exists(args.stencil_raw):
        raw_inputs.append(args.stencil_raw)
    raw_inputs = sorted(set(raw_inputs))

    if not raw_inputs:
        print("No raw benchmark CSV found. Skipping measured-speedup conversion.")
        return 0

    run(
        [
            PY,
            "scripts/build_measured_speedup_csv.py",
            "--raw",
            ",".join(raw_inputs),
            "--output",
            args.measured_out,
        ]
    )

    eval_cmd = [PY, "scripts/evaluate_mape.py", "--measured", args.measured_out]
    if args.compare_l2_modes:
        eval_cmd.append("--compare-l2-modes")
    if args.leave_one_out:
        eval_cmd.append("--leave-one-out")
    run(eval_cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
