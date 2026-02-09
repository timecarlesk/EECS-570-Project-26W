#!/usr/bin/env python3
"""Build CUDA benchmarks and run GEMM / Stencil / pointer-chase experiments."""

import argparse
import os
import shlex
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, cwd=None):
    print("+ {}".format(" ".join(cmd)))
    subprocess.check_call(cmd, cwd=cwd)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default=os.path.join(ROOT, "build"))
    parser.add_argument("--gpu-name", default="")

    parser.add_argument("--run-gemm", action="store_true")
    parser.add_argument("--run-stencil", action="store_true")
    parser.add_argument("--run-pointer", action="store_true")
    parser.add_argument("--skip-build", action="store_true")

    parser.add_argument("--gemm-output", default=os.path.join(ROOT, "outputs", "gemm_raw.csv"))
    parser.add_argument("--stencil-output", default=os.path.join(ROOT, "outputs", "stencil_raw.csv"))
    parser.add_argument("--pointer-output", default=os.path.join(ROOT, "outputs", "pointer_chase_raw.csv"))

    parser.add_argument("--gemm-n-values", default="")
    parser.add_argument("--gemm-tile-sizes", default="16,32,64,128")
    parser.add_argument("--gemm-stages", default="2,3,4")
    parser.add_argument("--run-v0", action="store_true")

    parser.add_argument("--stencil-lengths", default="")
    parser.add_argument("--stencil-stages", default="2,3,4")

    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)

    parser.add_argument("--gemm-extra-args", default="")
    parser.add_argument("--stencil-extra-args", default="")
    parser.add_argument("--pointer-extra-args", default="")

    return parser.parse_args()


def main():
    args = parse_args()

    if not (args.run_gemm or args.run_stencil or args.run_pointer):
        args.run_gemm = True
        args.run_stencil = True
        args.run_pointer = True

    if shutil.which("nvcc") is None and not args.skip_build:
        print("nvcc not found in PATH. CUDA build cannot proceed in this environment.")
        return 1
    if shutil.which("nvcc") is None and args.skip_build:
        print("nvcc not found in PATH; continuing with --skip-build using existing binaries.")

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)

    if not args.skip_build:
        os.makedirs(args.build_dir, exist_ok=True)
        run(["cmake", "-S", ROOT, "-B", args.build_dir])
        run(["cmake", "--build", args.build_dir, "-j"])

    def exe(name):
        return os.path.join(args.build_dir, name)

    gpu_args = []
    if args.gpu_name:
        gpu_args = ["--gpu-name", args.gpu_name]

    if args.run_gemm:
        cmd = [
            exe("benchmark_gemm"),
            "--output",
            args.gemm_output,
            "--tile-sizes",
            args.gemm_tile_sizes,
            "--stages",
            args.gemm_stages,
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ]
        if args.gemm_n_values.strip():
            cmd.extend(["--n-values", args.gemm_n_values.strip()])
        if args.run_v0:
            cmd.append("--run-v0")
        if args.gemm_extra_args.strip():
            cmd.extend(shlex.split(args.gemm_extra_args.strip()))
        cmd.extend(gpu_args)
        run(cmd)

    if args.run_stencil:
        cmd = [
            exe("benchmark_stencil"),
            "--output",
            args.stencil_output,
            "--stages",
            args.stencil_stages,
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ]
        if args.stencil_lengths.strip():
            cmd.extend(["--lengths", args.stencil_lengths.strip()])
        if args.stencil_extra_args.strip():
            cmd.extend(shlex.split(args.stencil_extra_args.strip()))
        cmd.extend(gpu_args)
        run(cmd)

    if args.run_pointer:
        cmd = [exe("microbench_pointer_chase"), "--output", args.pointer_output]
        if args.pointer_extra_args.strip():
            cmd.extend(shlex.split(args.pointer_extra_args.strip()))
        cmd.extend(gpu_args)
        run(cmd)

    print("Finished CUDA benchmarks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
