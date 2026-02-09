#!/usr/bin/env python3
"""Demonstrate Triton autotune pre-filter behavior."""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from gpu_specs import DEFAULT_STAGE_CANDIDATES, list_gpu_names
from triton_prefilter import prune_num_stages
from workloads import default_problem_sizes


def parse_int_list(raw):
    if raw is None or raw.strip() == "":
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="gemm", choices=["gemm", "stencil"])
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--problem-sizes", default="")
    parser.add_argument(
        "--stages",
        default=",".join(str(s) for s in DEFAULT_STAGE_CANDIDATES),
    )
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--tile-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpus.strip().lower() == "all":
        gpus = list_gpu_names()
    else:
        gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]

    sizes = parse_int_list(args.problem_sizes)
    if not sizes:
        sizes = default_problem_sizes(args.workload)

    stages = parse_int_list(args.stages)
    if not stages:
        stages = list(DEFAULT_STAGE_CANDIDATES)

    for gpu in gpus:
        for size in sizes:
            result = prune_num_stages(
                args.workload,
                gpu,
                size,
                stages,
                epsilon=args.epsilon,
                tile_size=args.tile_size,
            )
            print(
                "{} {} size={} kept={} pruned={}".format(
                    args.workload,
                    gpu,
                    size,
                    result["kept"],
                    result["pruned"],
                )
            )


if __name__ == "__main__":
    main()
