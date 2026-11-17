"""
plot_thesis_routing_masks.py

Thesis figure 4.4 (Thesis/figures/fig_4_4_routing_masks.svg): the pointwise density mask (top row) and the
window-density mask (bottom row) for the same frame of three scenes, GT heads as dots. Data only, no model.
Same masks as training: pointwise > 1.273 single-person peaks; window = mean density over 25 x 25 cells > 0.4 p/m^2 (the trained H2 head)
with the minimum region area of the H2 runs (50 cells = 2 m^2, doc/head_ablation.md). Layout and frame choice follow
visualize_oracle_mask_candidates.py (median-count frame per scene from the window-sweep JSON), styled for the thesis:
text width, 8 pt fonts, thesis palette.

Usage (from src):
    uv run python scripts/visualisation/plot_thesis_routing_masks.py
    uv run python scripts/visualisation/plot_thesis_routing_masks.py --scenes scene_18,scene_48,scene_24
"""

import os
import sys
import json
import argparse

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
for _d in ("", "training_scripts"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

# Thesis palette (Thesis/figures/tikz/*.tex): orange = proposed head, ink / muted for text
ORANGE, INK, MUTED, SURFACE = "#E4572E", "#222222", "#7A7A7A", "#FFFFFF"
POINTWISE_TAU = 1.273  # single-person peaks (utils/oracle_mask.py)
TEXT_WIDTH_IN = 15.7 / 2.54


def median_frame(sweep_rows, scene):
    frames = sorted(
        {(r["frame"], r["people"]) for r in sweep_rows if r["scene"] == scene and r["config"] == "current"},
        key=lambda x: x[1],
    )
    return frames[len(frames) // 2][0] if frames else 0


def main():
    import torch
    import torch.nn.functional as F
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import train_proposed_multiscene as T
    from utils.oracle_mask import compute_oracle_mask

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene_18,scene_48,scene_24")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--density", type=float, default=0.4, help="people/m^2 (0.4 = the trained H2 head)")
    ap.add_argument("--min_area", type=int, default=50, help="cells (50 = 2 m^2, as in the H2 runs)")
    ap.add_argument(
        "--sweep_json", default=os.path.join(src_dir, "diagnostics", "oracle_mask_extent", "oracle_mask_window.json")
    )
    ap.add_argument(
        "--out", default=os.path.join(src_dir, "figs", "thesis", "fig_4_4_routing_masks.svg")
    )
    args = ap.parse_args()
    scenes = [s.strip() for s in args.scenes.split(",")]

    with open(args.sweep_json) as f:
        sweep_rows = json.load(f)["frames"]
    kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE)
    peak, ksum = float(kernel.max()), float(kernel.sum())
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams = json.load(f)
    registry = {}
    for split in ("train", "validate"):
        for e in T.build_scene_registry(split, 5, cams, {}, validate_setup=False):
            registry[e["scene_name"]] = e

    panels = []
    for scene in scenes:
        idx = median_frame(sweep_rows, scene)
        s = registry[scene]["dataset"][idx]
        wld = s["wld_map_paras"].unsqueeze(0)
        h, w = int(wld[0, 3].item()), int(wld[0, 4].item())
        dots = T.create_ground_truth_heatmap(s["gt_world_xy"], wld, torch.zeros(2, dtype=torch.int32), bev_shape=(h, w))
        conv = F.conv2d(dots, kernel, padding=kernel.shape[-1] // 2)
        masks = [
            compute_oracle_mask(conv, POINTWISE_TAU * peak, smoothing_kernel=T.MASK_SMOOTHING_KERNEL),
            compute_oracle_mask(
                conv,
                args.density,
                smoothing_kernel=T.MASK_SMOOTHING_KERNEL,
                min_area=args.min_area,
                density_window=args.window,
                kernel_sum=ksum,
                px_per_m=5.0,
            ),
        ]
        n_people = float(dots.sum())
        stats = []
        for m in masks:
            m_np = (m[0, 0].numpy() > 0.5).astype(np.uint8)
            share = float((dots * m).sum()) / max(n_people, 1.0)
            stats.append((share, cv2.connectedComponents(m_np, connectivity=8)[0] - 1, m_np))
        ys, xs = np.nonzero(dots[0, 0].numpy())
        panels.append((scene, idx, h, w, int(n_people), ys, xs, stats))
        print(f"{scene} frame {idx}: {int(n_people)} people; pointwise {stats[0][1]} regions, window {stats[1][1]}")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
    plt.rcParams["svg.fonttype"] = "none"
    widths = [w / h for _, _, h, w, *_ in panels]
    panel_h = (TEXT_WIDTH_IN - 0.45) / sum(widths)
    fig, axes = plt.subplots(
        2,
        len(panels),
        figsize=(TEXT_WIDTH_IN, 2 * panel_h + 0.75),
        gridspec_kw={"width_ratios": widths, "wspace": 0.04, "hspace": 0.32},
        squeeze=False,
        facecolor=SURFACE,
    )
    rows = ("Pointwise density mask", "Window-density mask")
    for col, (scene, idx, h, w, n_people, ys, xs, stats) in enumerate(panels):
        for row in range(2):
            ax = axes[row, col]
            share, n_comp, m_np = stats[row]
            ax.set_facecolor(SURFACE)
            ax.imshow(
                m_np,
                cmap=ListedColormap([SURFACE, ORANGE]),
                alpha=0.3,
                interpolation="nearest",
                vmin=0,
                vmax=1,
                extent=(0, w, h, 0),
            )
            if m_np.any():
                ax.contour(np.arange(w) + 0.5, np.arange(h) + 0.5, m_np, levels=[0.5], colors=[ORANGE], linewidths=0.7)
            ax.scatter(xs + 0.5, ys + 0.5, s=0.6, c=INK, linewidths=0)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(MUTED)
                sp.set_linewidth(0.5)
            head = f"{scene.replace('_', ' ')}, {n_people} people\n" if row == 0 else ""
            ax.set_title(
                f"{head}{100 * share:.0f}% of people dense, {n_comp} regions", fontsize=7.5, color=INK, pad=3
            )
            if col == 0:
                ax.set_ylabel(rows[row], fontsize=8, color=INK)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.9, bottom=0.02)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
