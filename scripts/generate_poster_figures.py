#!/usr/bin/env python3
"""Generate the four figures used in the poster individually.

Figures produced in outputs/figures/:
  1. gemm_speedup_by_gpu.png     -- GEMM cp.async speedup per GPU (bar chart)
  2. pred_vs_meas_scatter.png    -- Predicted vs measured speedup scatter
  3. l2_characterization_v2.png  -- L2 effective capacity & latency per GPU
  4. gemm_cross_gpu_best.png     -- Best GEMM pipeline speedup vs problem size
"""

from __future__ import division

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from gpu_specs import GPU_SPECS, get_gpu_spec
from predictor import predict_one

# ── Poster GPU color palette (matches poster.tex definitions) ──────────────────
GPU_COLORS = {
    "V100":             "#A9A9A9",   # gpuV100 grey
    "A40":              "#4682B4",   # gpuA40  steel blue
    "A100_MIG_3g40gb":  "#3CB371",   # gpuA100 medium sea green
    "L40S":             "#9467BD",   # gpuL40S medium orchid
    "H100_SXM5":        "#FF8C00",   # gpuH100 dark orange
}

# Short display labels
GPU_LABELS = {
    "V100":             "V100",
    "A40":              "A40",
    "A100_MIG_3g40gb":  "A100\nMIG",
    "L40S":             "L40S",
    "H100_SXM5":        "H100",
}

# Ordered list (poster table order)
GPU_ORDER = ["V100", "A40", "A100_MIG_3g40gb", "L40S", "H100_SXM5"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_deps():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        return plt, np, pd
    except ImportError as e:
        sys.exit("Missing dependency: {}".format(e))


def save(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print("Wrote  {}".format(path))


def enrich(df):
    """Add pred_speedup, w_conc_over_l2, ape columns."""
    rows = []
    for _, r in df.iterrows():
        tile = int(r["tile_size"]) if str(r["tile_size"]).strip() else None
        p = predict_one(str(r["workload"]), str(r["gpu"]),
                        int(r["problem_size"]), int(r["stage"]),
                        tile_size=tile, l2_mode="effective")
        rows.append({
            "pred_speedup":    float(p.get("pred_speedup", 0.0)),
            "w_conc_over_l2":  float(p.get("w_conc_over_l2", 0.0)),
        })
    df = df.copy()
    df["pred_speedup"]   = [x["pred_speedup"]   for x in rows]
    df["w_conc_over_l2"] = [x["w_conc_over_l2"] for x in rows]
    df["ape"] = (df["pred_speedup"] - df["measured_speedup"]).abs() \
                / df["measured_speedup"].abs().replace(0, 1.0)
    return df


# ── Figure 1: GEMM speedup by GPU (grouped bar, stages 2-4) ───────────────────

def fig_gemm_speedup_by_gpu(plt, np, df, out_dir, dpi):
    gemm = df[(df["workload"] == "gemm") & (df["stage"] > 1)].copy()

    # Best speedup per (gpu, stage) across all problem sizes and tile sizes
    agg = gemm.groupby(["gpu", "stage"])["measured_speedup"].max().reset_index()

    stages   = [2, 3, 4]
    gpus     = [g for g in GPU_ORDER if g in agg["gpu"].unique()]
    n_gpus   = len(gpus)
    x        = np.arange(n_gpus)
    width    = 0.22
    offsets  = [-width, 0.0, width]
    hatches  = ["", "//", ".."]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    for s, offset, hatch in zip(stages, offsets, hatches):
        vals = []
        for gpu in gpus:
            row = agg[(agg["gpu"] == gpu) & (agg["stage"] == s)]
            vals.append(float(row["measured_speedup"].values[0]) if len(row) else 1.0)
        ax.bar(x + offset, vals, width=width, label="S={}".format(s),
               color=[GPU_COLORS.get(g, "#888888") for g in gpus],
               hatch=hatch, edgecolor="white", linewidth=0.5, alpha=0.90)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([GPU_LABELS.get(g, g) for g in gpus], fontsize=10)
    ax.set_ylabel("cp.async Speedup vs S=1", fontsize=11)
    ax.set_title("GEMM cp.async Speedup per GPU", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.set_ylim(0.85, None)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, os.path.join(out_dir, "gemm_speedup_by_gpu.png"), dpi)
    plt.close(fig)


# ── Figure 2: Predicted vs Measured scatter ────────────────────────────────────

def fig_pred_vs_meas(plt, np, df, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)

    markers = {"gemm": "o", "stencil": "s"}
    for workload, grp in df.groupby("workload"):
        ax.scatter(grp["pred_speedup"], grp["measured_speedup"],
                   alpha=0.65, s=22,
                   marker=markers.get(workload, "o"),
                   label=workload.upper(),
                   color="#4682B4" if workload == "gemm" else "#FF8C00")

    max_v = max(float(df["pred_speedup"].max()),
                float(df["measured_speedup"].max()), 1.05)
    ax.plot([0.8, max_v], [0.8, max_v], "k--", linewidth=0.9, alpha=0.6)
    ax.set_xlim(0.8, max_v)
    ax.set_ylim(0.8, max_v)
    ax.set_xlabel("Predicted Speedup", fontsize=11)
    ax.set_ylabel("Measured Speedup", fontsize=11)
    ax.set_title("Predicted vs Measured Speedup", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    mape = float(df["ape"].mean()) * 100.0
    ax.text(0.05, 0.93, "MAPE = {:.1f}%".format(mape),
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFCB05", alpha=0.7))
    ax.legend(fontsize=9, framealpha=0.8)

    save(fig, os.path.join(out_dir, "pred_vs_meas_scatter.png"), dpi)
    plt.close(fig)


# ── Figure 3: L2 characterization v2 ──────────────────────────────────────────

def fig_l2_characterization(plt, np, out_dir, dpi):
    gpus = [g for g in GPU_ORDER if get_gpu_spec(g)["supports_cp_async"]]

    l2_eff    = [get_gpu_spec(g)["l2_eff_mb"]         for g in gpus]
    l2_nom    = [get_gpu_spec(g)["l2_nominal_mb"]      for g in gpus]
    l2_lat    = [get_gpu_spec(g)["l2_latency_cycles"]  for g in gpus]
    dram_lat  = [get_gpu_spec(g)["dram_latency_cycles"] for g in gpus]
    labels    = [GPU_LABELS.get(g, g).replace("\n", " ") for g in gpus]
    colors    = [GPU_COLORS.get(g, "#888888") for g in gpus]

    x = np.arange(len(gpus))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=dpi)

    # ---- Left: L2 capacity (nominal vs effective) ----
    ax1.bar(x - width / 2, l2_nom, width=width, color=colors,
            alpha=0.45, edgecolor="white", label="Nominal")
    ax1.bar(x + width / 2, l2_eff, width=width, color=colors,
            alpha=0.95, edgecolor="white", label="Effective")

    for i, (nom, eff) in enumerate(zip(l2_nom, l2_eff)):
        if abs(eff - nom) > 1.0:
            ax1.annotate("", xy=(x[i] + width / 2, eff),
                         xytext=(x[i] - width / 2, nom),
                         arrowprops=dict(arrowstyle="->", color="red",
                                         lw=1.2, connectionstyle="arc3,rad=0.3"))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("L2 Capacity (MB)", fontsize=11)
    ax1.set_title("L2 Capacity: Nominal vs Effective", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, framealpha=0.8)
    ax1.grid(axis="y", alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ---- Right: Latency comparison ----
    ax2.bar(x - width / 2, l2_lat,   width=width, color=colors,
            alpha=0.95, edgecolor="white", label="L2 Latency")
    ax2.bar(x + width / 2, dram_lat, width=width, color=colors,
            alpha=0.45, edgecolor="white", label="DRAM Latency",
            hatch="//")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Latency (cycles)", fontsize=11)
    ax2.set_title("Memory Latency by GPU", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.8)
    ax2.grid(axis="y", alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("L2 Characterization (pointer-chase microbenchmark)",
                 fontsize=12, fontweight="bold", y=1.01)
    save(fig, os.path.join(out_dir, "l2_characterization_v2.png"), dpi)
    plt.close(fig)


# ── Figure 4: Cross-GPU best GEMM pipeline speedup ────────────────────────────

def fig_gemm_cross_gpu_best(plt, np, df, out_dir, dpi):
    gemm = df[df["workload"] == "gemm"].copy()

    # Best speedup across stages and tile sizes for each (gpu, problem_size)
    best = (gemm.sort_values("measured_speedup", ascending=False)
                .groupby(["gpu", "problem_size"], as_index=False)
                .head(1))

    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)

    gpus_present = [g for g in GPU_ORDER if g in best["gpu"].unique()]
    for gpu in gpus_present:
        sub = best[best["gpu"] == gpu].sort_values("problem_size")
        ax.plot(sub["problem_size"], sub["measured_speedup"],
                marker="o", linewidth=1.8, markersize=6,
                color=GPU_COLORS.get(gpu, "#888888"),
                label=GPU_LABELS.get(gpu, gpu).replace("\n", " "))

    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Problem Size N (GEMM)", fontsize=11)
    ax.set_ylabel("Best cp.async Speedup (vs S=1)", fontsize=11)
    ax.set_title("Cross-GPU Best GEMM Pipeline Speedup", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.8, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, os.path.join(out_dir, "gemm_cross_gpu_best.png"), dpi)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measured",
                   default=os.path.join(ROOT, "outputs", "all_measured_speedup.csv"),
                   help="measured speedup CSV (default: outputs/all_measured_speedup.csv)")
    p.add_argument("--output-dir",
                   default=os.path.join(ROOT, "outputs", "figures"),
                   help="directory to write figures into")
    p.add_argument("--dpi", type=int, default=200,
                   help="output DPI (default: 200)")
    p.add_argument("--figure", choices=["all", "1", "2", "3", "4"],
                   default="all",
                   help="which figure to generate: all | 1 | 2 | 3 | 4")
    return p.parse_args()


def main():
    args = parse_args()
    plt, np, pd = load_deps()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load & normalise
    df = pd.read_csv(args.measured)
    df["workload"]        = df["workload"].astype(str).str.strip().str.lower()
    df["gpu"]             = df["gpu"].astype(str).str.strip()
    df["problem_size"]    = df["problem_size"].astype(int)
    df["stage"]           = df["stage"].astype(int)
    df["measured_speedup"]= df["measured_speedup"].astype(float)
    if "tile_size" not in df.columns:
        df["tile_size"] = ""

    fig_num = args.figure
    need_enriched = fig_num in ("all", "2")

    if need_enriched:
        print("Computing predictions (this may take a moment)...")
        df_enriched = enrich(df)
    else:
        df_enriched = df  # not used

    if fig_num in ("all", "1"):
        print("-- Figure 1: GEMM speedup by GPU")
        fig_gemm_speedup_by_gpu(plt, np, df, args.output_dir, args.dpi)

    if fig_num in ("all", "2"):
        print("-- Figure 2: Predicted vs Measured scatter")
        fig_pred_vs_meas(plt, np, df_enriched, args.output_dir, args.dpi)

    if fig_num in ("all", "3"):
        print("-- Figure 3: L2 characterization")
        fig_l2_characterization(plt, np, args.output_dir, args.dpi)

    if fig_num in ("all", "4"):
        print("-- Figure 4: Cross-GPU best GEMM speedup")
        fig_gemm_cross_gpu_best(plt, np, df, args.output_dir, args.dpi)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
