#!/usr/bin/env python3
"""Regenerate outputs/figures/l2_characterization.png.

Left panel:  Pointer-chase latency curves (cycles vs working-set size in MB).
Right panel: Nominal vs effective L2 capacity bar chart.
"""

import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from gpu_specs import get_gpu_spec

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as e:
    sys.exit("Missing dependency: {}".format(e))

OUT = os.path.join(ROOT, "outputs", "figures", "l2_characterization.png")
DPI = 200

GPU_ORDER = ["V100", "A40", "A100_MIG_3g40gb", "L40S", "H100_SXM5"]
GPU_LABELS = {
    "V100":            "V100",
    "A40":             "A40",
    "A100_MIG_3g40gb": "A100\nMIG",
    "L40S":            "L40S",
    "H100_SXM5":       "H100",
}
GPU_COLORS = {
    "V100":            "#A9A9A9",
    "A40":             "#4682B4",
    "A100_MIG_3g40gb": "#3CB371",
    "L40S":            "#9467BD",
    "H100_SXM5":       "#FF8C00",
}

POINTER_CHASE_FILES = {
    "V100":            ("v100",    "pointer_chase_raw.csv"),
    "A40":             ("a40",     "pointer_chase_raw.csv"),
    "A100_MIG_3g40gb": ("a100_mig","pointer_chase_raw.csv"),
    "L40S":            ("l40s",    "pointer_chase_raw.csv"),
    "H100_SXM5":       ("h100",    "pointer_chase_raw.csv"),
}


def load_pc(gpu):
    subdir, fname = POINTER_CHASE_FILES[gpu]
    path = os.path.join(ROOT, "outputs", subdir, fname)
    try:
        df = pd.read_csv(path)
    except OSError:
        return None
    # deduplicate by size, keep median cycles, then sort ascending
    df = (df.groupby("size_bytes", as_index=False)["cycles_per_load"]
            .median()
            .sort_values("size_bytes")
            .reset_index(drop=True))
    df["size_mb"] = df["size_bytes"] / (1024 * 1024)
    return df


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=DPI)

# ── Left: latency curves ──────────────────────────────────────────────────────
for gpu in GPU_ORDER:
    df = load_pc(gpu)
    if df is None:
        continue
    ax1.plot(df["size_mb"], df["cycles_per_load"],
             color=GPU_COLORS[gpu], linewidth=2.0,
             marker="o", markersize=4, markeredgewidth=0.5, markeredgecolor="white",
             label=GPU_LABELS[gpu].replace("\n", " "))
    # vertical dotted line at effective L2 cliff
    eff_mb = get_gpu_spec(gpu)["l2_eff_mb"]
    ax1.axvline(eff_mb, color=GPU_COLORS[gpu], lw=1.0, ls=":", alpha=0.65)

ax1.set_xscale("log", base=2)
ax1.set_xlabel("Working Set Size (MB)", fontsize=11)
ax1.set_ylabel("Latency (cycles/load)", fontsize=11)
ax1.set_title("Pointer-Chase Latency vs Working Set Size\n(dotted line = effective L2 cliff)",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=9, framealpha=0.85)
ax1.grid(True, alpha=0.2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# region annotations — placed in empty corners away from the curves
ax1.text(0.02, 0.50, "L2 region\n(low latency)", transform=ax1.transAxes,
         fontsize=8.5, color="#444444", va="center", ha="left",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.9))
ax1.text(0.78, 0.30, "DRAM region\n(high latency)", transform=ax1.transAxes,
         fontsize=8.5, color="#444444", va="bottom", ha="right",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.9))

# ── Right: nominal vs effective L2 bar chart ──────────────────────────────────
gpus   = GPU_ORDER
labels = [GPU_LABELS[g].replace("\n", " ") for g in gpus]
nom    = [get_gpu_spec(g)["l2_nominal_mb"] for g in gpus]
eff    = [get_gpu_spec(g)["l2_eff_mb"]     for g in gpus]
colors = [GPU_COLORS[g] for g in gpus]
x      = np.arange(len(gpus))
w      = 0.35

ax2.bar(x - w/2, nom, w, color=colors, alpha=0.40, edgecolor="white", label="Nominal L2")
ax2.bar(x + w/2, eff, w, color=colors, alpha=0.95, edgecolor="white", label="Effective L2 (measured)")

# annotate value on top of each effective bar
for i, (n, e) in enumerate(zip(nom, eff)):
    ax2.text(x[i] - w/2, n + 1.5, str(int(n)),
             ha="center", va="bottom", fontsize=8.5, color="#444444")
    diff   = e - n
    sign   = "+" if diff > 0 else ""
    fc     = "#2ca02c" if diff > 0 else "#d62728"
    label  = "{}\n({}{}%)".format(int(e), sign, int(round(100*diff/n)))
    ax2.text(x[i] + w/2, e + 1.5, label,
             ha="center", va="bottom", fontsize=8, color=fc, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10)
ax2.set_ylabel("L2 Cache Size (MB)", fontsize=11)
ax2.set_title("Nominal vs Effective L2 Capacity\n(effective ≠ nominal on every GPU)",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=9, framealpha=0.85)
ax2.grid(axis="y", alpha=0.2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("L2 Characterization (pointer-chase microbenchmark)",
             fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT, dpi=DPI, bbox_inches="tight")
print("Wrote  {}".format(OUT))
fig.clf()
