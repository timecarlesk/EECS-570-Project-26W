#!/usr/bin/env python3
"""Minimal Triton GEMM example using cp.async stage pre-filter.

This is a demonstration scaffold for Section 9 integration.
"""

from __future__ import division

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    import torch
    import triton
    import triton.language as tl
except Exception as e:
    raise RuntimeError("Triton demo requires torch + triton: {}".format(e))

from triton_prefilter import autotune_with_cp_async_prefilter


@autotune_with_cp_async_prefilter(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
    workload="gemm",
    gpu_name="A100_MIG_3g40gb",
    epsilon=0.03,
    problem_size_keys=("M", "N", "K"),
    tile_size_keys=("BLOCK_SIZE",),
)
@triton.jit
def gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a, b):
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


def _main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run Triton demo")

    M = 1024
    N = 1024
    K = 1024
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)

    c = matmul(a, b)
    ref = torch.matmul(a, b)

    max_err = (c - ref).abs().max().item()
    print("max_err={:.6f}".format(max_err))


if __name__ == "__main__":
    _main()
