#!/usr/bin/env python3
"""Generate core figures required by the proposal.

Figures generated:
1) speedup_vs_size_<workload>.png
2) speedup_vs_wconc_over_l2_<workload>.png
3) best_stage_vs_size_<workload>.png
4) speedup_rank_vs_ridge_rank.png
5) predicted_vs_measured_scatter.png
6) pipeline_benefit_map_<gpu>.png
"""

from __future__ import division

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from gpu_specs import get_gpu_spec
from predictor import predict_one


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured", required=True, help="measured speedup CSV")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "outputs", "figures"))
    parser.add_argument("--l2-mode", choices=["effective", "nominal"], default="effective")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def require_plotting_deps():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except Exception as e:
        print("Missing plotting dependencies: {}".format(e))
        print("Install with: pip install -r requirements.txt")
        return None, None, None
    return plt, np, pd


def enrich_with_predictions(df, l2_mode):
    pred_speedup = []
    w_over_l2 = []
    ai_vals = []

    for _, row in df.iterrows():
        tile_size = int(row["tile_size"]) if str(row["tile_size"]).strip() else None
        pred = predict_one(
            str(row["workload"]),
            str(row["gpu"]),
            int(row["problem_size"]),
            int(row["stage"]),
            tile_size=tile_size,
            l2_mode=l2_mode,
        )
        pred_speedup.append(float(pred.get("pred_speedup", 0.0)))
        w_over_l2.append(float(pred.get("w_conc_over_l2", 0.0)))
        ai_vals.append(float(pred.get("arithmetic_intensity", 0.0)))

    df = df.copy()
    df["pred_speedup"] = pred_speedup
    df["w_conc_over_l2"] = w_over_l2
    df["arithmetic_intensity"] = ai_vals
    df["ape"] = (df["pred_speedup"] - df["measured_speedup"]).abs() / df["measured_speedup"].abs().replace(0, 1.0)
    return df


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    print("Wrote {}".format(path))


def plot_speedup_vs_size(plt, best_rows, output_dir, dpi):
    workloads = sorted(best_rows["workload"].unique())
    for workload in workloads:
        fig = plt.figure(figsize=(7, 4), dpi=dpi)
        sub = best_rows[best_rows["workload"] == workload]
        for gpu in sorted(sub["gpu"].unique()):
            g = sub[sub["gpu"] == gpu].sort_values("problem_size")
            plt.plot(g["problem_size"], g["measured_speedup"], marker="o", label=gpu)
        plt.title("Speedup vs Problem Size ({})".format(workload))
        plt.xlabel("problem_size")
        plt.ylabel("best measured speedup (V1/V3)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        save_fig(fig, os.path.join(output_dir, "speedup_vs_size_{}.png".format(workload)))
        plt.close(fig)


def plot_speedup_vs_wconc(plt, best_rows, output_dir, dpi):
    workloads = sorted(best_rows["workload"].unique())
    for workload in workloads:
        fig = plt.figure(figsize=(7, 4), dpi=dpi)
        sub = best_rows[best_rows["workload"] == workload]
        for gpu in sorted(sub["gpu"].unique()):
            g = sub[sub["gpu"] == gpu].sort_values("w_conc_over_l2")
            plt.plot(g["w_conc_over_l2"], g["measured_speedup"], marker="o", label=gpu)
        plt.title("Speedup vs W_conc/L2 ({})".format(workload))
        plt.xlabel("W_conc / C_L2")
        plt.ylabel("best measured speedup")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        save_fig(fig, os.path.join(output_dir, "speedup_vs_wconc_over_l2_{}.png".format(workload)))
        plt.close(fig)


def plot_best_stage(plt, best_rows, output_dir, dpi):
    workloads = sorted(best_rows["workload"].unique())
    for workload in workloads:
        fig = plt.figure(figsize=(7, 4), dpi=dpi)
        sub = best_rows[best_rows["workload"] == workload]
        for gpu in sorted(sub["gpu"].unique()):
            g = sub[sub["gpu"] == gpu].sort_values("problem_size")
            plt.plot(g["problem_size"], g["stage"], marker="o", label=gpu)
        plt.title("Best Stage S*(N) ({})".format(workload))
        plt.xlabel("problem_size")
        plt.ylabel("best stage")
        plt.yticks([1, 2, 3, 4])
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        save_fig(fig, os.path.join(output_dir, "best_stage_vs_size_{}.png".format(workload)))
        plt.close(fig)


def plot_rank_vs_ridge(plt, np, best_rows, output_dir, dpi):
    speed_mean = best_rows.groupby("gpu")["measured_speedup"].mean()
    gpus = sorted(speed_mean.index.tolist())

    speed_rank = speed_mean.rank(ascending=False, method="dense")

    ridge_vals = []
    for gpu in gpus:
        ridge_vals.append(float(get_gpu_spec(gpu)["ridge"]))

    ridge_series = speed_mean.copy()
    for gpu, ridge in zip(gpus, ridge_vals):
        ridge_series[gpu] = ridge
    ridge_rank = ridge_series.rank(ascending=True, method="dense")

    x = np.arange(len(gpus))
    width = 0.35

    fig = plt.figure(figsize=(8, 4), dpi=dpi)
    plt.bar(x - width / 2.0, [speed_rank[g] for g in gpus], width=width, label="speedup rank")
    plt.bar(x + width / 2.0, [ridge_rank[g] for g in gpus], width=width, label="ridge rank")
    plt.xticks(x, gpus, rotation=20, ha="right")
    plt.ylabel("rank (1=best)")
    plt.title("Speedup Rank vs Ridge-Point Rank")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    save_fig(fig, os.path.join(output_dir, "speedup_rank_vs_ridge_rank.png"))
    plt.close(fig)


def plot_pred_vs_measured(plt, np, all_rows, output_dir, dpi):
    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    workloads = sorted(all_rows["workload"].unique())
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for i, workload in enumerate(workloads):
        sub = all_rows[all_rows["workload"] == workload]
        plt.scatter(
            sub["pred_speedup"],
            sub["measured_speedup"],
            alpha=0.65,
            s=20,
            label=workload,
            color=colors[i % len(colors)],
        )

    max_v = max(float(all_rows["pred_speedup"].max()), float(all_rows["measured_speedup"].max()), 1.0)
    plt.plot([0, max_v], [0, max_v], "k--", linewidth=1)
    plt.xlim(0, max_v)
    plt.ylim(0, max_v)
    plt.xlabel("predicted speedup")
    plt.ylabel("measured speedup")
    plt.title("Predicted vs Measured Speedup")
    plt.grid(True, alpha=0.25)

    mape = float(all_rows["ape"].mean()) * 100.0
    plt.text(0.05 * max_v, 0.92 * max_v, "MAPE={:.2f}%".format(mape), fontsize=10)
    plt.legend(fontsize=8)

    save_fig(fig, os.path.join(output_dir, "predicted_vs_measured_scatter.png"))
    plt.close(fig)


def plot_pipeline_benefit_maps(plt, np, best_rows, output_dir, dpi):
    gemm_rows = best_rows[best_rows["workload"] == "gemm"].copy()
    if gemm_rows.empty:
        return

    gemm_rows["ai_bin"] = gemm_rows["arithmetic_intensity"].round(3)
    gemm_rows["w_bin"] = gemm_rows["w_conc_over_l2"].round(3)

    for gpu in sorted(gemm_rows["gpu"].unique()):
        g = gemm_rows[gemm_rows["gpu"] == gpu]
        pivot = g.pivot_table(index="w_bin", columns="ai_bin", values="measured_speedup", aggfunc="mean")
        if pivot.empty:
            continue

        x_labels = [float(x) for x in pivot.columns]
        y_labels = [float(y) for y in pivot.index]
        mat = pivot.values

        fig = plt.figure(figsize=(7, 5), dpi=dpi)
        im = plt.imshow(mat, origin="lower", aspect="auto", cmap="viridis")
        plt.colorbar(im, label="best measured speedup")
        plt.title("Pipeline Benefit Map ({})".format(gpu))
        plt.xlabel("arithmetic_intensity")
        plt.ylabel("W_conc / C_L2")

        plt.xticks(range(len(x_labels)), ["{:.2f}".format(v) for v in x_labels], rotation=35, ha="right")
        plt.yticks(range(len(y_labels)), ["{:.2f}".format(v) for v in y_labels])

        save_fig(fig, os.path.join(output_dir, "pipeline_benefit_map_{}.png".format(gpu)))
        plt.close(fig)


def main():
    args = parse_args()
    plt, np, pd = require_plotting_deps()
    if plt is None:
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.measured)
    required = {"workload", "gpu", "problem_size", "stage", "measured_speedup"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError("Missing columns in measured CSV: {}".format(missing))

    if "tile_size" not in df.columns:
        df["tile_size"] = ""

    df["workload"] = df["workload"].astype(str).str.strip().str.lower()
    df["gpu"] = df["gpu"].astype(str).str.strip()
    df["problem_size"] = df["problem_size"].astype(int)
    df["stage"] = df["stage"].astype(int)
    df["measured_speedup"] = df["measured_speedup"].astype(float)

    all_rows = enrich_with_predictions(df, l2_mode=args.l2_mode)

    best_rows = (
        all_rows.sort_values(["workload", "gpu", "problem_size", "measured_speedup"], ascending=[True, True, True, False])
        .groupby(["workload", "gpu", "problem_size"], as_index=False)
        .head(1)
        .copy()
    )

    plot_speedup_vs_size(plt, best_rows, args.output_dir, args.dpi)
    plot_speedup_vs_wconc(plt, best_rows, args.output_dir, args.dpi)
    plot_best_stage(plt, best_rows, args.output_dir, args.dpi)
    plot_rank_vs_ridge(plt, np, best_rows, args.output_dir, args.dpi)
    plot_pred_vs_measured(plt, np, all_rows, args.output_dir, args.dpi)
    plot_pipeline_benefit_maps(plt, np, best_rows, args.output_dir, args.dpi)

    print("All figures written to {}".format(args.output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
