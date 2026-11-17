"""
plot_neck_upsides.py

Slide figure for the neck ablation's conclusion (doc/neck_ablation.md section 10): what the proposed attention fusion
(ca_sa, ablation_v2) does better and worse than the baseline's homography fusion (homography_256), from the existing
per-frame outputs (no model runs):

  (a) one scene over time: predicted vs true count per frame (test/scene_50 by default) -- the same MAE story can
      hide better tracking under a larger constant offset;
  (b) per held-out scene: within-scene error (MAE after removing each scene's mean signed error) -- the upside;
  (c) per held-out scene: share of predicted density more than 1 m from every person -- the downside.

Inputs: diagnostics/per_frame/ablation_v2__{homography_256,ca_sa}.csv (eval_per_frame.py, all frames) and
diagnostics/map_quality/ablation_v2__{homography_256,ca_sa}.csv (eval_map_quality.py, 50 frames per scene).

Usage (from src):
    uv run python scripts/visualisation/plot_neck_upsides.py
    uv run python scripts/visualisation/plot_neck_upsides.py --example_scene scene_50 --example_split test
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
MODELS = [
    ("homography_256", "Baseline fusion (homography)", "#2a78d6", "o"),
    ("ca_sa", "Proposed fusion (cross-attn → self-attn)", "#eb6834", "s"),
]


def load(kind, variant):
    """-> {(split, scene): {column: np.array sorted by snapshot}}"""
    rows = defaultdict(list)
    with open(os.path.join(src_dir, "diagnostics", kind, f"ablation_v2__{variant}.csv"), newline="") as f:
        for r in csv.DictReader(f):
            split = "val" if r["split"] == "validate" else r["split"]
            rows[(split, r["scene"])].append(r)
    out = {}
    for key, rs in rows.items():
        rs.sort(key=lambda r: int(r["snapshot"]))
        out[key] = {c: np.array([float(r[c]) for r in rs]) for c in rs[0] if c not in ("split", "scene")}
    return out


def within_scene_error(data):
    """MAE after removing each scene's mean signed error, per (split, scene)."""
    res = {}
    for key, d in data.items():
        e = d["pred"] - d["gt"]
        res[key] = float(np.abs(e - e.mean()).mean())
    return res


def split_mean(values, split):
    return float(np.mean([v for k, v in values.items() if k[0] == split]))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.6)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def paired_panel(ax, values, ylabel, title, better_low=True):
    """values: {variant: {(split, scene): value}} -> paired dots per scene, connected."""
    hom, prop = values["homography_256"], values["ca_sa"]
    keys = sorted(hom, key=lambda k: (k[0] != "val", hom[k]))
    x = np.arange(len(keys))
    for i, k in enumerate(keys):
        ax.plot([i, i], [hom[k], prop[k]], color=GRID, lw=1.6, zorder=1)
    for variant, label, color, marker in MODELS:
        v = values[variant]
        ax.scatter(
            x,
            [v[k] for k in keys],
            s=42,
            c=color,
            marker=marker,
            edgecolors=SURFACE,
            linewidths=1.2,
            zorder=3,
            label=label,
        )
    n_val = sum(k[0] == "val" for k in keys)
    ax.axvline(n_val - 0.5, color=MUTED, lw=0.6, ls=":")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.06 * (ymax - ymin))
    ax.text(n_val / 2 - 0.5, ymax, "val scenes", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(
        n_val + (len(keys) - n_val) / 2 - 0.5, ymax, "test scenes", ha="center", va="bottom", fontsize=8, color=MUTED
    )
    ax.set_xticks(x)
    ax.set_xticklabels([k[1].replace("scene_", "") for k in keys], fontsize=8)
    ax.set_xlabel("held-out scene", fontsize=9, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    wins = sum((prop[k] < hom[k]) == better_low for k in keys)
    tally = (
        f"proposed better in {wins} of {len(keys)} scenes"
        if wins >= len(keys) / 2
        else f"proposed worse in {len(keys) - wins} of {len(keys)} scenes"
    )
    ax.set_title(f"{title}\n{tally}", fontsize=10, color=INK, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example_scene", default="scene_50")
    ap.add_argument("--example_split", default="test")
    ap.add_argument("--out", default=os.path.join(src_dir, "figs", "neck_ablation", "fig_neck_upsides.png"))
    args = ap.parse_args()

    per_frame = {v: load("per_frame", v) for v, *_ in MODELS}
    map_q = {v: load("map_quality", v) for v, *_ in MODELS}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.15, 1, 1]})
    for ax in axes:
        style(ax)

    # (a) one scene: predicted vs true count per frame
    ax = axes[0]
    split = "val" if args.example_split in ("val", "validate") else args.example_split
    lo, hi = np.inf, -np.inf
    lines = []
    for variant, label, color, marker in MODELS:
        d = per_frame[variant][(split, args.example_scene)]
        e = d["pred"] - d["gt"]
        ax.scatter(d["gt"], d["pred"], s=14, c=color, marker=marker, alpha=0.55, linewidths=0, zorder=3)
        slope, icpt = np.polyfit(d["gt"], d["pred"], 1)
        xs = np.array([d["gt"].min(), d["gt"].max()])
        ax.plot(xs, slope * xs + icpt, color=color, lw=2.0, zorder=4)
        lines.append(
            f"{label.split(' (')[0]}: offset {e.mean():+.0f}, slope {slope:.2f}, "
            f"within-scene error {np.abs(e - e.mean()).mean():.0f}"
        )
        lo, hi = min(lo, d["gt"].min(), d["pred"].min()), max(hi, d["gt"].max(), d["pred"].max())
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, ls="--", zorder=2)
    ax.text(hi, hi, " perfect", fontsize=8, color=INK_2, va="bottom", ha="right")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=8,
        color=INK_2,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID),
    )
    ax.set_xlabel("true people per frame", fontsize=9, color=INK_2)
    ax.set_ylabel("predicted people per frame", fontsize=9, color=INK_2)
    ax.set_title(
        f"(a) example {split}/{args.example_scene}, every frame\n"
        f"proposed: larger constant offset, but its points follow the diagonal",
        fontsize=10,
        color=INK,
        loc="left",
    )

    # (b) upside: within-scene error per scene
    w = {v: within_scene_error(per_frame[v]) for v, *_ in MODELS}
    paired_panel(
        axes[1], w, "MAE after removing the scene offset (people/frame)", "(b) upside: follows the count within a scene"
    )

    # (c) downside: far mass per scene
    far = {v: {k: float(d["far1m"].mean() * 100) for k, d in map_q[v].items()} for v, *_ in MODELS}
    paired_panel(
        axes[2], far, "predicted density > 1 m from every person (%)", "(c) downside: more diffuse density map"
    )

    handles = [plt.Line2D([], [], color=c, marker=m, lw=1.6, ms=7, mec=SURFACE, label=lab) for _, lab, c, m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.text(
        0.01,
        0.005,
        "Same frozen backbone, head, loss and data; only the fusion differs. One seed, 30 epochs. "
        "(a, b) all frames of the 20 held-out scenes; (c) 50 frames per scene.",
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)

    for s in ("val", "test"):
        print(
            f"{s}: within-scene error homography {split_mean(w['homography_256'], s):.1f}, "
            f"ca_sa {split_mean(w['ca_sa'], s):.1f}; far>1m homography {split_mean(far['homography_256'], s):.1f}%, "
            f"ca_sa {split_mean(far['ca_sa'], s):.1f}%"
        )
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
