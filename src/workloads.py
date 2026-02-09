"""Workload definitions for GEMM and 1D stencil."""

GEMM_N_SWEEP = [256, 512, 1024, 2048, 3072, 4096, 6144, 8192]
STENCIL_L_SWEEP = [2 ** 16, 2 ** 18, 2 ** 20, 2 ** 22, 2 ** 24, 2 ** 26]
TILE_SWEEP = [16, 32, 64, 128]

DEFAULT_GEMM_TILE = 64
DEFAULT_STENCIL_TILE = 1024
DEFAULT_STENCIL_HALO = 2
DEFAULT_STENCIL_FLOPS_PER_ELEM = 20


def ceil_div(x, y):
    return (x + y - 1) // y


def make_gemm_point(n, tile_size=DEFAULT_GEMM_TILE):
    tm = tile_size
    tn = tile_size
    tk = tile_size

    w_block_bytes = (tm * tk + tk * tn) * 4
    grid_blocks = ceil_div(n, tm) * ceil_div(n, tn)

    # Standard GEMM FLOP count per tile.
    flops_per_tile = 2.0 * tm * tn * tk
    bytes_per_tile_for_ai = float(w_block_bytes)
    ai = flops_per_tile / bytes_per_tile_for_ai

    # Pipeline overlap happens at finer granularity than full output tile.
    overlap_chunks = max(1, tile_size // 8)

    return {
        "workload": "gemm",
        "problem_size": int(n),
        "tile_size": int(tile_size),
        "w_block_bytes": float(w_block_bytes),
        "grid_blocks": int(grid_blocks),
        "flops_per_tile": float(flops_per_tile),
        "bytes_per_tile_for_ai": float(bytes_per_tile_for_ai),
        "arithmetic_intensity": float(ai),
        "overlap_chunks": int(overlap_chunks),
    }


def make_stencil_point(
    length,
    tile_elems=DEFAULT_STENCIL_TILE,
    halo=DEFAULT_STENCIL_HALO,
    flops_per_elem=DEFAULT_STENCIL_FLOPS_PER_ELEM,
):
    w_block_bytes = (tile_elems + 2 * halo) * 4
    grid_blocks = ceil_div(length, tile_elems)

    # The proposal uses unrolling/multi-element per thread to boost stage compute density.
    flops_per_tile = float(tile_elems * flops_per_elem)
    bytes_per_tile_for_ai = float(tile_elems * 4)
    ai = flops_per_tile / bytes_per_tile_for_ai

    # Unrolling / multiple elements per thread increases per-stage compute granularity.
    overlap_chunks = max(1, tile_elems // 256)

    return {
        "workload": "stencil",
        "problem_size": int(length),
        "tile_size": int(tile_elems),
        "halo": int(halo),
        "w_block_bytes": float(w_block_bytes),
        "grid_blocks": int(grid_blocks),
        "flops_per_tile": float(flops_per_tile),
        "bytes_per_tile_for_ai": float(bytes_per_tile_for_ai),
        "arithmetic_intensity": float(ai),
        "overlap_chunks": int(overlap_chunks),
    }


def make_workload_point(workload, problem_size, tile_size=None):
    workload = workload.lower().strip()
    if workload == "gemm":
        use_tile = DEFAULT_GEMM_TILE if tile_size is None else int(tile_size)
        return make_gemm_point(problem_size, use_tile)
    if workload == "stencil":
        use_tile = DEFAULT_STENCIL_TILE if tile_size is None else int(tile_size)
        return make_stencil_point(problem_size, tile_elems=use_tile)
    raise ValueError("Unsupported workload: {}".format(workload))


def default_problem_sizes(workload):
    workload = workload.lower().strip()
    if workload == "gemm":
        return list(GEMM_N_SWEEP)
    if workload == "stencil":
        return list(STENCIL_L_SWEEP)
    raise ValueError("Unsupported workload: {}".format(workload))
