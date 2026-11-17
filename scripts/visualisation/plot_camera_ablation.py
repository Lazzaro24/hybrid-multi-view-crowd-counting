"""
scripts/visualisation/plot_camera_ablation.py

Figures of the camera ablation (doc/camera_ablation.md §7) from camera_ablation/summary.json
(evaluation/summarize_camera_ablation.py --out camera_ablation/summary.json):
  camera_ablation_views  (a) Tier A test MAE vs views: fusion trained at N (solid) and the 5-view model with vnorm
                         (dashed); (b)-(d) Tier B, the final models with vnorm: MAE, bias, F1
  camera_ablation_raw    Tier B bias vs views as trained (raw): the view softmax / camera sum scaling

Colour follows the entity: the homography fusion and baseline_ot (baseline decoder on the homography fusion) share
slot 1, ca_sa and h1_casa slot 2, h1_hom slot 3, hybrid_hom slot 4 (reference palette, categorical slots in order).

Usage (from src/):
    python scripts/visualisation/plot_camera_ablation.py              # figs/camera_ablation_{views,raw}.{png,svg}
"""

import os
import json
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK_2, MUTED = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
)  # as visualize_paper_prediction.py (no torch import)
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # reference palette, categorical slots 1-4 in order
GRID, AXIS = "#e5e4df", "#c3c2b7"
VIEWS = [3, 5, 7, 10]
MODELS = [
    ("baseline_ot", "Baseline (OT)", SERIES[0]),
    ("h1_hom", "H1, homography_256", SERIES[2]),
    ("h1_casa", "H1, ca_sa", SERIES[1]),
    ("hybrid_hom", "Hybrid, homography_256", SERIES[3]),
]
FUSIONS = [("homography_256", "homography_256", SERIES[0]), ("ca_sa", "ca_sa", SERIES[1])]


def style(ax, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_xticks(VIEWS)
    ax.set_xlim(2.4, 11.9)  # room for the end labels right of 10 views
    ax.set_xlabel("views", color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)


def line(ax, ys, color, dashed=False, open_marker=False):
    """ys over VIEWS (None = missing; the line is broken there)."""
    y = np.array([np.nan if v is None else v for v in ys], dtype=float)
    ax.plot(
        VIEWS,
        y,
        color=color,
        linewidth=1.8,
        linestyle=(0, (4, 2.5)) if dashed else "-",
        marker="o",
        markersize=5.5,
        markerfacecolor=SURFACE if open_marker else color,
        markeredgecolor=color,
        markeredgewidth=1.5,
        zorder=3,
    )
    # a missing point between two present ones: connect them thinly so the trend stays readable
    ok = ~np.isnan(y)
    if (~ok).any() and ok.sum() >= 2:
        ax.plot(np.array(VIEWS)[ok], y[ok], color=color, linewidth=0.8, linestyle=":", zorder=2)


def end_labels(ax, items, min_gap_frac=0.07):
    """Direct labels right of the last point, nudged apart so they don't overlap. items: (y, text)."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(items, key=lambda t: t[0])
    placed = []
    for y, text in items:
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)
    shift = max(0.0, placed[-1] - (hi - gap / 2)) if placed else 0.0  # keep the top label inside the axes
    for (_, text), y in zip(items, placed):
        ax.text(10.35, y - shift, text, va="center", ha="left", fontsize=7.5, color=INK_2)


def tier_b(summary, mode, model, key):
    d = summary["tier_b"][mode].get(model, {})
    return [(d.get(str(n)) or {}).get(key) for n in VIEWS]


def views_figure(summary, out):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), facecolor=SURFACE)
    (a, b), (c, d) = axes

    # (a) Tier A
    ta = summary["tier_a"]
    labels = []
    for v, name, color in FUSIONS:
        trained = [ta[v][str(n)]["test_final"]["mae"] for n in VIEWS]
        transfer = [
            ta[v]["5"]["test_final"]["mae"] if n == 5 else ta[v][str(n)]["transfer"]["vnorm"]["mae"] for n in VIEWS
        ]
        line(a, trained, color)
        line(a, transfer, color, dashed=True, open_marker=True)
        labels += [(trained[-1], f"{name}, trained"), (transfer[-1], f"{name}, 5-view")]
    style(a, "test MAE", "(a) Tier A: fusion + baseline decoder, K-means cameras, one seed")
    end_labels(a, labels)
    a.legend(
        handles=[
            Line2D([], [], color=INK_2, linewidth=1.8, marker="o", markersize=5, label="trained at N views"),
            Line2D(
                [],
                [],
                color=INK_2,
                linewidth=1.8,
                linestyle=(0, (4, 2.5)),
                marker="o",
                markersize=5,
                markerfacecolor=SURFACE,
                label="trained at 5, tested at N (vnorm)",
            ),
        ]
        + [Line2D([], [], color=col, linewidth=1.8, label=name) for _, name, col in FUSIONS],
        loc="upper left",
        frameon=False,
        fontsize=7.5,
        ncol=2,
    )
    a.set_ylim(55, 115)

    # (b)-(d) Tier B, vnorm
    for ax, key, ylabel, title in (
        (b, "MAE", "test MAE", "(b) Tier B: final models, count error"),
        (c, "bias", "bias (predicted − GT)", "(c) Tier B: count bias"),
        (d, "F1", "F1 (%)", "(d) Tier B: localization F1"),
    ):
        items = []
        for m, name, color in MODELS:
            ys = tier_b(summary, "vnorm", m, key)
            line(ax, ys, color)
            last = next(y for y in reversed(ys) if y is not None)
            items.append((last, name))
        style(ax, ylabel, title)
        if key == "bias":
            ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
        end_labels(ax, items)
    fig.legend(
        handles=[
            Line2D([], [], color=col, linewidth=1.8, marker="o", markersize=5, label=name) for _, name, col in MODELS
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.01),
        title="(b)-(d): paper protocol, random draws of N of 51 cameras, view magnitude restored (vnorm)",
        title_fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2.0, w_pad=2.0)
    save(fig, out)


def raw_figure(summary, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor=SURFACE)
    items = []
    for m, name, color in MODELS:
        ys = tier_b(summary, "raw", m, "bias")
        line(ax, ys, color)
        items.append((next(y for y in reversed(ys) if y is not None), name))
    style(ax, "bias (predicted − GT)", "Tier B as trained (raw): count bias vs views")
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
    end_labels(ax, items)
    ax.legend(
        handles=[
            Line2D([], [], color=col, linewidth=1.8, marker="o", markersize=5, label=name) for _, name, col in MODELS
        ],
        loc="lower left",
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    save(fig, out)


def save(fig, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print(f"Saved {out}.{ext}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Camera ablation figures")
    ap.add_argument("--summary", default=os.path.join(src_dir, "camera_ablation", "summary.json"))
    ap.add_argument("--out_dir", default=os.path.join(src_dir, "figs"))
    args = ap.parse_args()
    with open(args.summary) as f:
        summary = json.load(f)
    plt.rcParams.update(
        {"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "svg.fonttype": "none"}
    )
    views_figure(summary, os.path.join(args.out_dir, "camera_ablation_views"))
    raw_figure(summary, os.path.join(args.out_dir, "camera_ablation_raw"))


if __name__ == "__main__":
    main()
