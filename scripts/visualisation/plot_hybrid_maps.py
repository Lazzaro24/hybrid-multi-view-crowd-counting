"""
plot_hybrid_maps.py

Qualitative figure of the dense-first hybrid head (doc/head_ablation.md §8) against the baseline decoder H0 on the same
fusion: per scene, the five camera views of one map-quality frame and three BEV maps on ONE colour scale (persons per
cell): GT density (dots blurred with the training kernel), H0, and the hybrid with its dense-zone outline (density rule,
--thr). The frame is not hand-picked: the scene's map-quality frame whose hybrid count error is closest to the scene's
median (from the sweep CSV, threshold --thr).

Writes diagnostics/figures/hybrid_maps/<scene>.{pdf,png}.

Usage (from src):
    uv run python scripts/visualisation/plot_hybrid_maps.py --scenes scene_11,scene_78,scene_13
"""

import os
import sys
import csv
import json
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

import neck_ablation as NA
import evaluate_paper_protocol as EP
from utils.exclusions import load_excluded_scenes
from losses.gaussian_mse import get_map_kernel

OUT = os.path.join(src_dir, "diagnostics", "figures", "hybrid_maps")
# reference palette (dataviz skill): sequential blue ramp on the light surface, text tokens, categorical slot 2 for the
# dense-zone outline (a line, contrasting with the blue ramp)
SURFACE, TEXT, TEXT2, ZONE = "#fcfcfb", "#0b0b0b", "#52514e", "#eb6834"
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", RAMP)


def pick_frame(sweep_csv, scene, thr):
    rows = [
        r
        for r in csv.DictReader(open(sweep_csv))
        if r["split"] == "test" and r["scene"] == scene and float(r["thr"]) == thr
    ]
    err = np.array([float(r["pred"]) - float(r["gt"]) for r in rows])
    i = int(np.argmin(np.abs(np.abs(err) - np.median(np.abs(err)))))
    return int(rows[i]["snapshot"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene_11,scene_78,scene_13")
    ap.add_argument("--h0", default="runs/neck_ablation/ablation_v2/homography_256/last.pth")
    ap.add_argument("--hybrid", default="multiscene_final_stage_2_abl_homography_256_dense_S4D2_s42_ema.pth")
    ap.add_argument("--sweep_csv", default="diagnostics/density_sweep/hom_S4D2_s42.csv")
    ap.add_argument("--thr", type=float, default=0.15)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda")
    h0, _ = EP.load_model("ablation", os.path.join(src_dir, args.h0), device)
    hyb, _ = EP.load_model("proposed", os.path.join(src_dir, args.hybrid), device)
    hyb.T.apply_sparse_render(hyb, {"mode": "peaks", "kernel": 3, "threshold": 0.064})
    hyb.head.render_kernel = hyb.head.render_kernel.to(device)
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    test = dict(NA.load_split("test", cams_cfg, load_excluded_scenes(), "master_refs.pt", 4))
    kernel = get_map_kernel(sigma=NA.MAP_SIGMA, kernel_size=NA.MAP_KERNEL_SIZE).to(device)
    kernel = (kernel / kernel.sum()).float().reshape(1, 1, *kernel.shape[-2:])
    plt.rcParams.update(
        {
            "font.size": 8,
            "text.color": TEXT,
            "axes.labelcolor": TEXT2,
            "xtick.color": TEXT2,
            "ytick.color": TEXT2,
            "axes.edgecolor": TEXT2,
            "figure.facecolor": SURFACE,
        }
    )

    for scene in args.scenes.split(","):
        ds = test[scene]
        snap = pick_frame(os.path.join(src_dir, args.sweep_csv), scene, args.thr)
        idx = [str(s) for s in ds.snapshots].index(str(snap))
        sample = default_collate([ds[idx]])
        shape = (int(sample["wld_map_paras"][0, 3]), int(sample["wld_map_paras"][0, 4]))
        with torch.no_grad():
            p_h0 = (EP.predict_ablation(h0, sample, device) / EP.SCALE).float()
            zone = EP.density_zone(hyb, sample, device, args.thr, shape)
            p_hy = (EP.predict_proposed(hyb, sample, device, zone) / EP.SCALE).float()
        gts = EP.gt_pixels(sample["gt_world_xy"], sample["wld_map_paras"], shape)
        dots = torch.zeros((1, 1) + shape, device=device)
        if len(gts):
            dots[0, 0, gts[:, 0].astype(int), gts[:, 1].astype(int)] = 1.0
        gt = F.conv2d(dots, kernel, padding=kernel.shape[-1] // 2)[0, 0][: shape[0], : shape[1]]
        maps = [m.cpu().numpy() for m in (gt, p_h0, p_hy)]
        vmax = float(np.percentile(maps[0][maps[0] > 1e-4], 99.5)) if (maps[0] > 1e-4).any() else 1.0
        n_gt = len(gts)
        counts = [n_gt, float(maps[1].sum()), float(maps[2].sum())]
        titles = [
            f"Ground truth: {n_gt} people",
            f"Baseline decoder H0: {counts[1]:.0f} ({counts[1] - n_gt:+.0f})",
            f"Hybrid head: {counts[2]:.0f} ({counts[2] - n_gt:+.0f})",
        ]

        cams = cams_cfg[scene][str(NA.NUM_CAMS)]
        aspect = shape[1] / shape[0]
        fig = plt.figure(figsize=(7.2, 1.0 + 2.4 / max(aspect, 0.6) + 0.4), constrained_layout=True)
        sub = fig.subfigures(2, 1, height_ratios=[1.0, 2.6 / max(aspect, 0.6)])
        top = sub[0].subplots(1, len(cams))
        for ax, c in zip(np.atleast_1d(top), cams):
            img = cv2.imread(os.path.join(ds.scene_dir, str(snap), "jpgs", f"{c}.jpg"))
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"camera {c}", color=TEXT2, fontsize=7)
            ax.set_axis_off()
        bottom = sub[1].subplots(1, 3)
        for ax, m, t in zip(bottom, maps, titles):
            im = ax.imshow(m, cmap=CMAP, vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(t, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        bottom[2].contour(zone.numpy().astype(float), levels=[0.5], colors=ZONE, linewidths=0.8)
        bottom[2].plot([], [], color=ZONE, lw=0.8, label=f"dense zone (own density ≥ {args.thr:g} p/m²)")
        bottom[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7, frameon=False, labelcolor=TEXT)
        cb = sub[1].colorbar(im, ax=list(bottom), shrink=0.8, pad=0.01)
        cb.set_label("persons per cell (0.2 m)", color=TEXT2)
        cb.outline.set_edgecolor(TEXT2)
        sub[0].suptitle(f"{scene} (test), snapshot {snap}", color=TEXT, fontsize=9)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"{scene}.{ext}"), dpi=300, facecolor=SURFACE)
        plt.close(fig)
        print(
            f"[saved] {scene} snapshot {snap}: GT {n_gt}, H0 {counts[1]:.0f}, hybrid {counts[2]:.0f}, "
            f"dense zone {100 * float(zone.float().mean()):.0f}%"
        )


if __name__ == "__main__":
    main()
