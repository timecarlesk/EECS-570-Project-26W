#!/usr/bin/env python3
"""Generate the three tables from the poster as individual PNG files.

Tables produced in outputs/figures/:
  table_kernels.png       -- Kernels & Methodology
  table_gpu_specs.png     -- Hardware: 5 GPU Architectures
  table_model_accuracy.png -- Model Accuracy (MAPE)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs", "figures")

# ── UMich palette ──────────────────────────────────────────────────────────────
UM_BLUE   = "#00274C"
UM_MAIZE  = "#FFCB05"
MAIZE_25  = "#FFF5CC"
ROW_ALT   = "#F4F8FF"
WHITE     = "#FFFFFF"

GPU_COLORS = {
    "V100":            "#A9A9A9",
    "A40":             "#4682B4",
    "A100 MIG":        "#3CB371",
    "L40S":            "#9467BD",
    "H100":            "#FF8C00",
}


def load_mpl(dpi=200):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib import rcParams
        rcParams["font.family"] = "DejaVu Sans"
        return plt, mpatches
    except ImportError as e:
        sys.exit("Missing matplotlib: {}".format(e))


def save(fig, name, dpi=200):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print("Wrote  {}".format(path))


# ── Table 1: Kernels & Methodology ────────────────────────────────────────────

def table_kernels(plt, dpi):
    col_labels = ["Kernel", "Variants", "Description"]
    rows = [
        ["GEMM",   "V1, V3", "C = AB, compute-bound.\nN ∈ {256–8192}, tile ∈ {16–128}"],
        ["Stencil","V1, V3", "1D, memory-bound (AI ≈ 5).\nL ∈ {2¹⁶ – 2²⁶}"],
    ]
    col_widths = [0.12, 0.14, 0.74]

    fig_w, fig_h = 7.0, 1.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_axis_off()

    row_h   = 0.18   # height per data row (axes fraction)
    hdr_h   = 0.22
    pad     = 0.04
    top     = 0.88

    # header
    x = pad
    for label, w in zip(col_labels, col_widths):
        ax.add_patch(plt.Rectangle((x, top), w - 0.01, hdr_h,
                                   transform=ax.transAxes,
                                   fc=UM_BLUE, ec="white", lw=0.8, clip_on=False))
        ax.text(x + (w - 0.01) / 2, top + hdr_h / 2, label,
                transform=ax.transAxes, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        x += w

    # data rows
    for ri, row in enumerate(rows):
        bg = WHITE if ri % 2 == 0 else ROW_ALT
        y  = top - hdr_h - ri * row_h
        x  = pad
        for ci, (cell, w) in enumerate(zip(row, col_widths)):
            ax.add_patch(plt.Rectangle((x, y - row_h), w - 0.01, row_h,
                                       transform=ax.transAxes,
                                       fc=bg, ec="#CCCCCC", lw=0.5, clip_on=False))
            ax.text(x + (w - 0.01) / 2, y - row_h / 2, cell,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9.5, color="#222222", multialignment="center")
            x += w

    # footer note
    ax.text(pad, top - hdr_h - len(rows) * row_h - 0.08,
            "V3: cp.async pipelined, S ∈ {2, 3, 4}.  Median of 10 runs.  Same source on all GPUs.",
            transform=ax.transAxes, fontsize=8.5, color="#555555", style="italic")

    ax.set_title("Kernels & Methodology", fontsize=12, fontweight="bold",
                 color=UM_BLUE, pad=6)
    save(fig, "table_kernels.png", dpi)
    plt.close(fig)


# ── Table 2: GPU Specs ─────────────────────────────────────────────────────────

def table_gpu_specs(plt, dpi):
    gpu_names  = ["V100", "A40", "A100 MIG", "L40S", "H100"]
    col_labels = ["", "V100", "A40", "A100 MIG", "L40S", "H100"]

    rows = [
        ["Arch",        "Volta", "Ampere", "Ampere",  "Ada",    "Hopper"],
        ["SMs",         "80",    "84",     "42",       "142",    "132"],
        ["BW (GB/s)",   "900",   "696",    "968",      "864",    "3350"],
        ["SMEM/SM",     "96 K",  "100 K",  "164 K",    "100 K",  "228 K"],
        ["Ridge",       "17.4",  "53.7",   "7.9",      "52.9",   "20.0"],
        ["L2 nominal",  "6 MB",  "6 MB",   "20 MB",    "96 MB",  "50 MB"],
        ["L2 effective","8 MB",  "8 MB",   "24 MB",    "112 MB", "28 MB"],
    ]
    highlight_row = len(rows) - 1   # "L2 effective" row
    col_widths = [0.20, 0.16, 0.16, 0.16, 0.16, 0.16]

    fig_w, fig_h = 8.5, 3.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_axis_off()

    row_h = 0.115
    hdr_h = 0.135
    pad   = 0.02
    top   = 0.86

    # header
    x = pad
    for ci, (label, w) in enumerate(zip(col_labels, col_widths)):
        color = GPU_COLORS.get(label, UM_BLUE)
        ax.add_patch(plt.Rectangle((x, top), w - 0.01, hdr_h,
                                   transform=ax.transAxes,
                                   fc=color if ci > 0 else UM_BLUE,
                                   ec="white", lw=0.8, clip_on=False))
        ax.text(x + (w - 0.01) / 2, top + hdr_h / 2, label,
                transform=ax.transAxes, ha="center", va="center",
                color="white", fontsize=10.5, fontweight="bold")
        x += w

    # data rows
    for ri, row in enumerate(rows):
        is_hl = (ri == highlight_row)
        bg    = MAIZE_25 if is_hl else (WHITE if ri % 2 == 0 else ROW_ALT)
        y     = top - hdr_h - ri * row_h
        x     = pad
        for ci, (cell, w) in enumerate(zip(row, col_widths)):
            ax.add_patch(plt.Rectangle((x, y - row_h), w - 0.01, row_h,
                                       transform=ax.transAxes,
                                       fc=bg, ec="#CCCCCC", lw=0.5, clip_on=False))
            fw = "bold" if (is_hl and ci > 0) else "normal"
            ax.text(x + (w - 0.01) / 2, y - row_h / 2, cell,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9.5, color="#222222", fontweight=fw)
            x += w

    # key note
    note_y = top - hdr_h - len(rows) * row_h - 0.06
    ax.text(pad, note_y,
            "Key: L2 effective ≠ nominal on all GPUs. H100: 50 → 28 MB (slice locality). L40S: 96 → 112 MB.",
            transform=ax.transAxes, fontsize=8.5, color="#555555", style="italic")

    ax.set_title("Hardware: 5 GPU Architectures", fontsize=12, fontweight="bold",
                 color=UM_BLUE, pad=6)
    save(fig, "table_gpu_specs.png", dpi)
    plt.close(fig)


# ── Table 3: Model Accuracy (MAPE) ────────────────────────────────────────────

def table_model_accuracy(plt, dpi):
    col_labels = ["GPU", "GEMM MAPE", "Stencil MAPE"]
    rows = [
        ["A40",      "11.1%", "4.1%"],
        ["A100 MIG", "9.2%",  "6.7%"],
        ["L40S",     "5.7%",  "7.2%"],
        ["H100",     "9.9%",  "5.5%"],
        ["Overall",  "9.0%",  "5.9%"],
    ]
    highlight_row = len(rows) - 1  # Overall
    col_widths = [0.34, 0.33, 0.33]

    fig_w, fig_h = 5.0, 2.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_axis_off()

    row_h = 0.145
    hdr_h = 0.16
    pad   = 0.06
    top   = 0.84

    # header
    x = pad
    for label, w in zip(col_labels, col_widths):
        ax.add_patch(plt.Rectangle((x, top), w - 0.015, hdr_h,
                                   transform=ax.transAxes,
                                   fc=UM_BLUE, ec="white", lw=0.8, clip_on=False))
        ax.text(x + (w - 0.015) / 2, top + hdr_h / 2, label,
                transform=ax.transAxes, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        x += w

    gpu_label_colors = {
        "A40":      GPU_COLORS["A40"],
        "A100 MIG": GPU_COLORS["A100 MIG"],
        "L40S":     GPU_COLORS["L40S"],
        "H100":     GPU_COLORS["H100"],
        "Overall":  UM_BLUE,
    }

    # data rows
    for ri, row in enumerate(rows):
        is_hl = (ri == highlight_row)
        bg    = MAIZE_25 if is_hl else (WHITE if ri % 2 == 0 else ROW_ALT)
        y     = top - hdr_h - ri * row_h
        x     = pad
        for ci, (cell, w) in enumerate(zip(row, col_widths)):
            ax.add_patch(plt.Rectangle((x, y - row_h), w - 0.015, row_h,
                                       transform=ax.transAxes,
                                       fc=bg, ec="#CCCCCC", lw=0.5, clip_on=False))
            color = gpu_label_colors.get(cell, "#222222") if ci == 0 else "#222222"
            fw    = "bold" if (is_hl or ci == 0) else "normal"
            ax.text(x + (w - 0.015) / 2, y - row_h / 2, cell,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10.5, color=color, fontweight=fw)
            x += w

    # subtitle
    ax.text(pad, top - hdr_h - len(rows) * row_h - 0.07,
            "Leave-one-out: held-out GPU < 15% MAPE.  Zero fitted parameters.",
            transform=ax.transAxes, fontsize=8.5, color="#555555", style="italic")

    ax.set_title("Model Accuracy (MAPE)", fontsize=12, fontweight="bold",
                 color=UM_BLUE, pad=6)
    save(fig, "table_model_accuracy.png", dpi)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--table", choices=["all", "1", "2", "3"], default="all")
    args = p.parse_args()

    plt, _ = load_mpl(args.dpi)

    if args.table in ("all", "1"):
        print("-- Table 1: Kernels & Methodology")
        table_kernels(plt, args.dpi)

    if args.table in ("all", "2"):
        print("-- Table 2: GPU Specs")
        table_gpu_specs(plt, args.dpi)

    if args.table in ("all", "3"):
        print("-- Table 3: Model Accuracy")
        table_model_accuracy(plt, args.dpi)

    print("Done.")


if __name__ == "__main__":
    main()
