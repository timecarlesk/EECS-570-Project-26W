"""Generate roofline figure showing V0/V1/V3 attained GFLOPS per GPU.

Handout 9.7 deliverable: "Roofline analysis showing how async copy
changes arithmetic intensity."
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "figures", "roofline.png")

# Per-GPU specs (ridge point from paper Table 1)
GPUS = {
    "V100":            dict(peak_tf=15.7, bw_gbs=900,  color="#9467bd", marker="v"),
    "A40":             dict(peak_tf=37.4, bw_gbs=696,  color="#1f77b4", marker="o"),
    "A100_MIG_3g40gb": dict(peak_tf=7.6,  bw_gbs=968,  color="#ff7f0e", marker="s"),
    "L40S":            dict(peak_tf=45.7, bw_gbs=864,  color="#2ca02c", marker="^"),
    "H100_SXM5":       dict(peak_tf=67.0, bw_gbs=3350, color="#d62728", marker="D"),
}
LABELS = {"V100": "V100 (no cp.async)",
          "A40": "A40", "A100_MIG_3g40gb": "A100 MIG",
          "L40S": "L40S", "H100_SXM5": "H100"}

# Arithmetic intensity (FLOP / byte of DRAM traffic)
#   GEMM tiled V1/V3: AI = T/4 (one 4-byte load per K-stripe, tile reuse = T)
#   GEMM naive  V0:   AI ~ 0.25 (each A/B element loaded ~N times without reuse)
#   Stencil:          AI ~ 1.25 (5 FLOP / 4-byte load per interior point)
def ai_gemm(variant, tile):
    if variant == "V0":
        return 0.25
    return tile / 4.0

def ai_stencil(variant, _tile):
    # Per workloads.py: 28 FLOP/elem, 4 bytes/elem → 7 FLOP/byte (unrolled kernel).
    return 7.0

def best_gflops(df, variant):
    """Best (highest-gflops) entry per (gpu, problem_size, variant)."""
    sub = df[df.variant == variant]
    return sub.groupby(["gpu", "problem_size", "variant", "tile_size"])\
              ["gflops"].max().reset_index()

def draw_roofline(ax, peak_tf, bw_gbs, color, label):
    x = np.logspace(-2, 3, 200)
    peak_gf = peak_tf * 1e3  # TFLOPS -> GFLOPS
    mem_gf = bw_gbs * x      # GB/s * FLOP/B = GFLOPS
    roof = np.minimum(peak_gf, mem_gf)
    ax.plot(x, roof, linestyle="--", color=color, alpha=0.35, linewidth=1.0)
    ax.axhline(peak_gf, linestyle=":", color=color, alpha=0.25, linewidth=0.8)

def plot(ax, workload):
    frames = []
    for gpu in GPUS:
        if "MIG" in gpu:
            path = os.path.join(ROOT, "outputs", "a100_mig", f"{workload}_raw.csv")
        elif "H100" in gpu:
            path = os.path.join(ROOT, "outputs", "h100", f"{workload}_raw.csv")
        else:
            path = os.path.join(ROOT, "outputs", gpu.lower(),
                                f"{workload}_raw.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df.variant.isin(["V0", "V1", "V3"])]
        df = df[df.correct == 1]
        frames.append(df)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)

    for gpu, spec in GPUS.items():
        gdf = df[df.gpu == gpu]
        if gdf.empty:
            continue
        draw_roofline(ax, spec["peak_tf"], spec["bw_gbs"], spec["color"], LABELS[gpu])
        for variant, mk, alpha in [("V0", "x", 0.55), ("V1", spec["marker"], 0.7), ("V3", spec["marker"], 1.0)]:
            vdf = gdf[gdf.variant == variant]
            if vdf.empty:
                continue
            if workload == "gemm":
                # best tile per variant
                vdf = vdf.loc[vdf.groupby(["problem_size","tile_size"])["gflops"].idxmax()]
                ai = vdf["tile_size"].apply(lambda t: ai_gemm(variant, t))
            else:
                vdf = vdf.loc[vdf.groupby(["problem_size"])["gflops"].idxmax()]
                ai = pd.Series([ai_stencil(variant, None)] * len(vdf))
            face = "none" if variant == "V1" else spec["color"]
            ax.scatter(ai, vdf["gflops"], s=28, marker=mk,
                       facecolor=face, edgecolor=spec["color"],
                       alpha=alpha, linewidth=1.0)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Attained GFLOPS")
    ax.set_title({"gemm":"GEMM","stencil":"Stencil"}[workload])
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    ax.set_xlim(0.1, 200)

def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    plot(axes[0], "gemm")
    plot(axes[1], "stencil")

    # Unified legend
    handles = []
    for gpu, spec in GPUS.items():
        handles.append(plt.Line2D([],[], marker=spec["marker"], color=spec["color"],
                                  linestyle="--", label=LABELS[gpu], markersize=5))
    handles.append(plt.Line2D([],[], marker="x", color="gray", linestyle="None",
                              label="V0 naive", markersize=5))
    handles.append(plt.Line2D([],[], marker="o", color="gray", linestyle="None",
                              markerfacecolor="none", label="V1 sync tiled", markersize=5))
    handles.append(plt.Line2D([],[], marker="o", color="gray", linestyle="None",
                              label="V3 cp.async", markersize=5))
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
