"""
Visual sanity check for the oracle-mask redesign: per scene, the current pointwise mask (left) next to a
window-density candidate (right), GT people as dots, full scene BEV. Data only, no model.

Scenes are picked from analyse_oracle_mask_extent.py's window-sweep JSON (unless --scenes is given): the scene where
the candidate routes the most people dense, one around the median, one where the candidate mask is empty but the
current mask is not, and the scene where the current mask routes the most people dense. Per scene the frame with
the median people count among the analysed frames is shown.

Usage:
    uv run scripts/visualisation/visualize_oracle_mask_candidates.py --window 25 --density 0.5
"""

import os
import sys
import json
import argparse

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

# Reference palette (same as scripts/visualisation/visualize_paper_prediction.py): categorical slots 1 and 2
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
CURRENT_COLOR, CANDIDATE_COLOR = "#2a78d6", "#eb6834"
CURRENT_TAU = 1.273


def pick_scenes(sweep_json, candidate):
    with open(sweep_json) as f:
        rows = json.load(f)["frames"]

    def scene_means(cfg):
        acc = {}
        for r in rows:
            if r["config"] == cfg:
                acc.setdefault(r["scene"], []).append(r["people_dense"])
        return {s: float(np.mean(v)) for s, v in acc.items()}

    cand, cur = scene_means(candidate), scene_means("current")
    if not cand:
        raise SystemExit(f"config {candidate!r} not in {sweep_json}; run the window sweep with this window/density")
    order = sorted(cand, key=cand.get)
    nonzero = [s for s in order if cand[s] > 0]
    picks = [order[-1], nonzero[len(nonzero) // 2] if nonzero else order[len(order) // 2]]
    empty = [s for s in order if cand[s] == 0 and cur.get(s, 0) > 0]
    if empty:
        picks.append(max(empty, key=cur.get))
    picks.append(max(cur, key=cur.get))
    return list(dict.fromkeys(picks)), rows


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
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--density", type=float, default=0.5, help="persons/m^2")
    ap.add_argument(
        "--sweep_json", default=os.path.join(T.src_dir, "diagnostics", "oracle_mask_extent", "oracle_mask_window.json")
    )
    ap.add_argument("--scenes", default=None, help="comma-separated; default: picked from --sweep_json")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--layout",
        choices=("portrait", "landscape"),
        default="portrait",
        help="portrait: one row per scene (current | candidate); landscape: one row per mask, one "
        "column per scene (for slides)",
    )
    args = ap.parse_args()

    candidate = f"W{args.window}_D{args.density:g}"
    picked, rows = pick_scenes(args.sweep_json, candidate)
    scenes = [s.strip() for s in args.scenes.split(",")] if args.scenes else picked
    suffix = "_landscape" if args.layout == "landscape" else ""
    out = args.out or os.path.join(T.src_dir, "diagnostics", "oracle_mask_extent", f"masks_{candidate}{suffix}.png")

    kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE)
    peak, ksum = float(kernel.max()), float(kernel.sum())
    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams = json.load(f)
    registry = {}
    for split in ("train", "validate"):
        for e in T.build_scene_registry(split, 5, cams, {}, validate_setup=False):
            registry[e["scene_name"]] = e

    panels = []
    for scene in scenes:
        frames = sorted(
            {(r["frame"], r["people"]) for r in rows if r["scene"] == scene and r["config"] == "current"},
            key=lambda x: x[1],
        )
        idx = frames[len(frames) // 2][0] if frames else 0
        s = registry[scene]["dataset"][idx]
        wld = s["wld_map_paras"].unsqueeze(0)
        h, w = int(wld[0, 3].item()), int(wld[0, 4].item())
        dots = T.create_ground_truth_heatmap(s["gt_world_xy"], wld, torch.zeros(2, dtype=torch.int32), bev_shape=(h, w))
        conv = F.conv2d(dots, kernel, padding=kernel.shape[-1] // 2)
        masks = {
            "current": compute_oracle_mask(conv, CURRENT_TAU * peak, smoothing_kernel=T.MASK_SMOOTHING_KERNEL),
            "candidate": compute_oracle_mask(
                conv,
                args.density,
                smoothing_kernel=T.MASK_SMOOTHING_KERNEL,
                density_window=args.window,
                kernel_sum=ksum,
                px_per_m=5.0,
            ),
        }
        n_people = float(dots.sum())
        stats = {}
        for name, m in masks.items():
            m_np = (m[0, 0].numpy() > 0.5).astype(np.uint8)
            stats[name] = (
                float((dots * m).sum()) / max(n_people, 1.0),
                cv2.connectedComponents(m_np, connectivity=8)[0] - 1,
                m_np,
            )
        ys, xs = np.nonzero(dots[0, 0].numpy())
        panels.append((scene, idx, h, w, int(n_people), ys, xs, stats))

    titles = {
        "current": (f"Current: pointwise > {CURRENT_TAU} single-person peaks", CURRENT_COLOR),
        "candidate": (
            f"Candidate: mean density over {args.window} px ({args.window / 5:g} m) > {args.density:g} persons/m²",
            CANDIDATE_COLOR,
        ),
    }
    if args.layout == "landscape":
        widths = [w / h for _, _, h, w, *_ in panels]
        fig, axes = plt.subplots(
            2,
            len(panels),
            figsize=(4.2 * sum(widths) + 0.6, 9.0),
            gridspec_kw={"width_ratios": widths},
            squeeze=False,
            facecolor=SURFACE,
        )
    else:
        ratios = [max(h / w, 0.35) for _, _, h, w, *_ in panels]
        fig, axes = plt.subplots(
            len(panels),
            2,
            figsize=(13, 6.5 * sum(ratios) + 0.8),
            gridspec_kw={"height_ratios": ratios},
            squeeze=False,
            facecolor=SURFACE,
        )
    for row, (scene, idx, h, w, n_people, ys, xs, stats) in enumerate(panels):
        for col, name in enumerate(("current", "candidate")):
            ax = axes[col, row] if args.layout == "landscape" else axes[row, col]
            title, color = titles[name]
            share, n_comp, m_np = stats[name]
            ax.set_facecolor(SURFACE)
            ax.imshow(
                m_np,
                cmap=ListedColormap([SURFACE, color]),
                alpha=0.35,
                interpolation="nearest",
                vmin=0,
                vmax=1,
                extent=(0, w, h, 0),
            )
            if m_np.any():
                ax.contour(np.arange(w) + 0.5, np.arange(h) + 0.5, m_np, levels=[0.5], colors=[color], linewidths=1.0)
            ax.scatter(xs + 0.5, ys + 0.5, s=2, c=INK, linewidths=0)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(MUTED)
                sp.set_linewidth(0.5)
            if args.layout == "landscape":
                if row == 0:
                    ax.set_ylabel(title.replace(": ", ":\n", 1), fontsize=10, color=color)
                ax.set_title(
                    f"{scene} · {n_people} people\n{100 * share:.0f}% dense, {n_comp} regions", fontsize=10, color=INK
                )
            else:
                head = f"{title}\n" if row == 0 else ""
                ax.set_title(
                    f"{head}{scene} · frame {idx} · {n_people} people — {100 * share:.0f}% dense, {n_comp} regions",
                    fontsize=9,
                    color=INK if row == 0 else INK_2,
                    loc="left",
                )
    fig.text(
        0.01,
        0.002,
        "Dots: GT people. Shaded: dense zone (routed to the dense expert). BEV 5 px = 1 m; "
        "dense expert receptive field 25 px.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.01, 1, 1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"[saved] {out}  scenes: {scenes}")


if __name__ == "__main__":
    main()
