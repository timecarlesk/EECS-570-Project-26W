#!/usr/bin/env python3
"""Collect Nsight Compute metrics for selected benchmark configurations."""

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_csv_ints(raw):
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def run(cmd, dry_run=False):
    print("+ {}".format(" ".join(cmd)))
    if not dry_run:
        subprocess.check_call(cmd, cwd=ROOT)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default=os.path.join(ROOT, "build"))
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "outputs", "nsight"))
    parser.add_argument("--gpu-name", default="")
    parser.add_argument("--workloads", default="gemm,stencil")
    parser.add_argument("--n-values", default="2048,4096")
    parser.add_argument("--lengths", default="1048576,4194304")
    parser.add_argument("--stages", default="1,2,3,4")
    parser.add_argument("--tile-sizes", default="16,32,64,128")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--metrics",
        default=(
            "lts__t_sector_hit_rate.pct,dram__bytes_read.sum,"
            "sm__warps_active.avg.pct_of_peak_sustained_active,"
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct,"
            "lts__throughput.avg.pct_of_peak_sustained_elapsed"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if shutil.which("ncu") is None and not args.dry_run:
        print("ncu not found in PATH.")
        return 1

    workloads = [w.strip().lower() for w in args.workloads.split(",") if w.strip()]
    stages = parse_csv_ints(args.stages)
    n_values = parse_csv_ints(args.n_values)
    lengths = parse_csv_ints(args.lengths)
    tile_sizes = parse_csv_ints(args.tile_sizes)

    os.makedirs(args.output_dir, exist_ok=True)

    gemm_exe = os.path.join(args.build_dir, "benchmark_gemm")
    stencil_exe = os.path.join(args.build_dir, "benchmark_stencil")

    gpu_args = []
    if args.gpu_name:
        gpu_args = ["--gpu-name", args.gpu_name]

    metrics = args.metrics

    if "gemm" in workloads:
        for n in n_values:
            for tile in tile_sizes:
                for stage in stages:
                    log_path = os.path.join(
                        args.output_dir,
                        "ncu_gemm_N{}_T{}_S{}.csv".format(n, tile, stage),
                    )
                    raw_out = os.path.join(
                        args.output_dir,
                        "ncu_gemm_driver_N{}_T{}_S{}.csv".format(n, tile, stage),
                    )

                    cmd = [
                        "ncu",
                        "--csv",
                        "--force-overwrite",
                        "--target-processes",
                        "all",
                        "--metrics",
                        metrics,
                        "--log-file",
                        log_path,
                        gemm_exe,
                        "--n-values",
                        str(n),
                        "--tile-sizes",
                        str(tile),
                        "--stages",
                        str(stage),
                        "--iters",
                        str(args.iters),
                        "--warmup",
                        str(args.warmup),
                        "--output",
                        raw_out,
                    ] + gpu_args
                    run(cmd, dry_run=args.dry_run)

    if "stencil" in workloads:
        for length in lengths:
            for stage in stages:
                log_path = os.path.join(args.output_dir, "ncu_stencil_L{}_S{}.csv".format(length, stage))
                raw_out = os.path.join(args.output_dir, "ncu_stencil_driver_L{}_S{}.csv".format(length, stage))

                cmd = [
                    "ncu",
                    "--csv",
                    "--force-overwrite",
                    "--target-processes",
                    "all",
                    "--metrics",
                    metrics,
                    "--log-file",
                    log_path,
                    stencil_exe,
                    "--lengths",
                    str(length),
                    "--stages",
                    str(stage),
                    "--iters",
                    str(args.iters),
                    "--warmup",
                    str(args.warmup),
                    "--output",
                    raw_out,
                ] + gpu_args
                run(cmd, dry_run=args.dry_run)

    print("Nsight profiling complete. Outputs in {}".format(args.output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
