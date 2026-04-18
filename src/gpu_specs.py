"""GPU parameter table derived from proposal_rq2.tex.

Values for latency/clock are placeholders until replaced by real
microbenchmark measurements.
"""

from collections import OrderedDict

# Default stage candidates: stage 1 is synchronous baseline, 2-4 are cp.async stages.
DEFAULT_STAGE_CANDIDATES = [1, 2, 3, 4]

GPU_SPECS = OrderedDict(
    [
        (
            "V100",
            {
                "supports_cp_async": False,
                "arch": "Volta",
                "sm_ver": 70,
                "n_sm": 80,
                "smem_per_sm_kb": 96,
                "l2_nominal_mb": 6.0,
                "l2_eff_mb": 6.5,
                "bw_gbps": 900.0,
                "fp32_tflops": 15.7,
                "ridge": 17.4,
                "sm_clock_ghz": 1.50,
                "l2_latency_cycles": 219.1,
                "dram_latency_cycles": 426.8,
                "max_blocks_per_sm": 32,
                "occupancy_alpha": 0.15,
            },
        ),
        (
            "A40",
            {
                "supports_cp_async": True,
                "arch": "Ampere",
                "sm_ver": 86,
                "n_sm": 84,
                "smem_per_sm_kb": 100,
                "l2_nominal_mb": 6.0,
                "l2_eff_mb": 6.0,
                "bw_gbps": 696.0,
                "fp32_tflops": 37.4,
                "ridge": 53.7,
                "sm_clock_ghz": 1.74,
                "l2_latency_cycles": 254.0,
                "dram_latency_cycles": 559.1,
                "max_blocks_per_sm": 32,
                "occupancy_alpha": 0.15,
            },
        ),
        (
            "A100_MIG_3g40gb",
            {
                "supports_cp_async": True,
                "arch": "Ampere",
                "sm_ver": 80,
                "n_sm": 42,
                "smem_per_sm_kb": 164,
                "l2_nominal_mb": 20.0,
                "l2_eff_mb": 21.0,
                "bw_gbps": 968.0,
                "fp32_tflops": 7.6,
                "ridge": 7.9,
                "sm_clock_ghz": 1.41,
                "l2_latency_cycles": 199.0,
                "dram_latency_cycles": 446.7,
                "max_blocks_per_sm": 32,
                "occupancy_alpha": 0.15,
            },
        ),
        (
            "L40S",
            {
                "supports_cp_async": True,
                "arch": "Ada",
                "sm_ver": 89,
                "n_sm": 142,
                "smem_per_sm_kb": 100,
                "l2_nominal_mb": 96.0,
                "l2_eff_mb": 96.0,
                "bw_gbps": 864.0,
                "fp32_tflops": 45.7,
                "ridge": 52.9,
                "sm_clock_ghz": 2.52,
                "l2_latency_cycles": 287.0,
                "dram_latency_cycles": 431.7,
                "max_blocks_per_sm": 32,
                "occupancy_alpha": 0.15,
            },
        ),
        (
            "H100_SXM5",
            {
                "supports_cp_async": True,
                "arch": "Hopper",
                "sm_ver": 90,
                "n_sm": 132,
                "smem_per_sm_kb": 228,
                "l2_nominal_mb": 50.0,
                "l2_eff_mb": 26.0,
                "bw_gbps": 3350.0,
                "fp32_tflops": 67.0,
                "ridge": 20.0,
                "sm_clock_ghz": 1.98,
                "l2_latency_cycles": 263.9,
                "dram_latency_cycles": 614.0,
                "max_blocks_per_sm": 32,
                "occupancy_alpha": 0.15,
            },
        ),
    ]
)


def list_gpu_names():
    return list(GPU_SPECS.keys())


def get_gpu_spec(gpu_name):
    if gpu_name not in GPU_SPECS:
        raise KeyError("Unknown GPU: {}".format(gpu_name))
    spec = dict(GPU_SPECS[gpu_name])
    spec["name"] = gpu_name
    return spec
