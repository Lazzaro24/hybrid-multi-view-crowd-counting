"""
scripts/visualisation/plot_test_comparison.py

Per-scene test MAE of several models scored by evaluation/evaluate_paper_protocol.py, as grouped
horizontal bars (scenes sorted by the first model's MAE), with each model's overall MAE in the legend.

Usage (from src/):
    python scripts/visualisation/plot_test_comparison.py \
        --evals runs/baseline/run_1/eval_test_best.json,eval_test_<a>.json,eval_test_<b>.json \
        --labels "Baseline,Proposed (MSE sparse),Proposed (OT sparse)" --out figs/comparison_test_per_scene.png
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from visualize_paper_prediction import SURFACE, INK, INK_2, MUTED

SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]  # reference palette, categorical slots 1-8 in order


def main(args):
    evals = [json.load(open(p)) for p in args.evals.split(",")]
    labels = args.labels.split(",")
    assert len(labels) == len(evals) <= len(SERIES)
    scenes = sorted(evals[0]["per_scene_MAE"], key=lambda s: evals[0]["per_scene_MAE"][s])
    y = np.arange(len(scenes))
    h = 0.8 / len(evals)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(scenes) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (ev, label, color) in enumerate(zip(evals, labels, SERIES)):
        vals = [ev["per_scene_MAE"][s] for s in scenes]
        offset = (len(evals) - 1 - i) * h - 0.4 + h / 2  # first model on top within each group
        ax.barh(
            y + offset,
            vals,
            height=h,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            label=f"{label} (MAE {ev['MAE']:.1f})",
        )
        for yi, v in zip(y + offset, vals):
            ax.text(v + 4, yi, f"{v:.0f}", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, scenes, fontsize=9, color=INK_2)
    ax.set_xlabel("test MAE per scene (people, whole scene, 5 random 5-view draws per frame)", color=INK_2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title(args.title, loc="left", fontsize=11, color=INK)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-scene test MAE of several models")
    parser.add_argument(
        "--evals", type=str, required=True, help="Comma-separated eval JSONs (first one sets the order)"
    )
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument(
        "--title", type=str, default="Per-scene test MAE  ·  paper protocol, same frames and view draws"
    )
    parser.add_argument("--out", type=str, default=os.path.join("figs", "comparison_test_per_scene.png"))
    parser.add_argument("--dpi", type=int, default=150)
    main(parser.parse_args())
