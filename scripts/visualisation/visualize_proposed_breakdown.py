"""
scripts/visualisation/visualize_proposed_breakdown.py

Where does the proposed model's paper-protocol error come from? For each scene of a split, one
representative sample (frame whose error is closest to the scene MAE from the eval JSON, scanning
every --frame_stride-th frame) is plotted with the baseline on the SAME views:

  row 1  ground truth | baseline prediction | proposed prediction (learned routing)
  row 2  router mask (+ oracle dense-zone outline) | dense expert x mask | sparse expert x (1 - mask)
  row 3  signed error baseline | signed error proposed | proposed detections (evaluator post-processing)

plus a summary chart over all scenes: the proposed model's predicted mass split by expert and by
distance to the nearest person (near <= 2 m < far), against the GT and baseline counts, and a JSON with
the numbers. Uses evaluate_paper_protocol.py's model loading, units, GT and detection code.

Usage (from src/):
    python scripts/visualisation/visualize_proposed_breakdown.py \
        --weights multiscene_best_stage_3_s8_dout0.7_cw0.01_f10_a0.7.pth \
        --eval_json eval_test_multiscene_best_stage_3_s8_dout0.7_cw0.01_f10_a0.7.json
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
from evaluation.evaluate_paper_protocol import (
    load_model,
    predict_baseline,
    gt_pixels,
    detect,
    oracle_zone as _oracle_zone,
    SCALE,
)
from losses.gaussian_mse import get_map_kernel
import training_scripts.train_proposed_multiscene as T


def oracle_zone(gts, shape):
    return _oracle_zone(T, gts, shape).numpy()


from visualize_paper_prediction import (
    SURFACE,
    INK,
    INK_2,
    MUTED,
    DENSITY_CMAP,
    IMG_MEAN,
    IMG_STD,
    match_assign,
    load_sample,
    batched,
    gt_density,
    style_map_axis,
    plot_detections,
)

# Reference palette: categorical slots 1-4 in fixed order, blue<->red diverging with a gray midpoint
DENSE_NEAR, DENSE_FAR, SPARSE_NEAR, SPARSE_FAR = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
ERROR_CMAP = LinearSegmentedColormap.from_list("err", ["#0d366b", "#3987e5", "#f0efec", "#e66767", "#a32a2a"])
MASK_CMAP = LinearSegmentedColormap.from_list("mask", [SURFACE, "#9ec5f4", "#256abf", "#0d366b"])
FAR_PX = 10  # 2 m at 5 px/m


@torch.no_grad()
def predict_proposed_parts(model, sample, device):
    """Same tiling as evaluate_paper_protocol.predict_proposed, keeping mask / dense / sparse separately.
    Maps in people per pixel (mask in [0, 1])."""
    images, refs = sample["images"].to(device), sample["ref_points_full"].to(device)
    wld = sample["wld_map_paras"]
    h_full, w_full = int(wld[0, 3].item()), int(wld[0, 4].item())
    th, tw = model.bev_shape

    def starts(full, tile):
        return sorted(set(list(range(0, full - tile + 1, tile)) + [full - tile]))

    parts = {k: torch.zeros((h_full, w_full), device=device) for k in ("mask", "dense", "sparse")}
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        B, N = images.shape[:2]
        feats = model.backbone(images.flatten(0, 1))
        feats = feats.view(B, N, *feats.shape[1:])
        for h in starts(h_full, th):
            for w in starts(w_full, tw):
                bev = model.neck(feats, refs[:, :, h : h + th, w : w + tw])
                parts["mask"][h : h + th, w : w + tw] = model.head.gating_net(bev)[0, 0].float()
                parts["dense"][h : h + th, w : w + tw] = F.relu(model.head.dense_expert(bev))[0, 0].float()
                parts["sparse"][h : h + th, w : w + tw] = F.relu(model.head.sparse_expert(bev))[0, 0].float()
    to_people = model.to_scale / SCALE
    m = parts["mask"]
    out = {"mask": m, "dense_c": m * parts["dense"] * to_people, "sparse_c": (1 - m) * parts["sparse"] * to_people}
    out["pred"] = out["dense_c"] + out["sparse_c"]
    return {k: v.cpu().numpy() for k, v in out.items()}


def scene_figure(args, scene, snapshot, raw, gt, base, parts, gts, dets, tp_det, tp_gt, stats, out_path):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
            "text.color": INK,
            "axes.labelcolor": INK_2,
        }
    )
    fig = plt.figure(figsize=(15, 16.5), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 15, height_ratios=[0.42, 1, 1, 1], hspace=0.2, wspace=0.35)
    images = raw["images"].permute(0, 2, 3, 1).numpy() * IMG_STD + IMG_MEAN
    for v in range(len(images)):
        ax = fig.add_subplot(gs[0, 3 * v : 3 * v + 3])
        ax.imshow(np.clip(images[v], 0, 1))
        style_map_axis(ax, f"View {v + 1}")

    n = len(gts)
    vmax = max(gt.max(), base.max(), np.percentile(parts["pred"], 99.9))
    err_lim = max(np.abs(base - gt).max(), np.percentile(np.abs(parts["pred"] - gt), 99.9), 1e-3)
    zone = oracle_zone(gts, gt.shape)

    def add(row, col, data, title, cmap, norm=None, vmin=None, vmax_=None):
        ax = fig.add_subplot(gs[row, 5 * col : 5 * col + 5])
        im = ax.imshow(data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax_, interpolation="nearest")
        style_map_axis(ax, title)
        return ax, im

    _, im_d = add(1, 0, gt, f"Ground truth  ·  {n} people", DENSITY_CMAP, vmin=0, vmax_=vmax)
    add(1, 1, base, f"Baseline  ·  {base.sum():.0f} (error {base.sum() - n:+.0f})", DENSITY_CMAP, vmin=0, vmax_=vmax)
    add(
        1,
        2,
        parts["pred"],
        f"Proposed  ·  {parts['pred'].sum():.0f} (error {parts['pred'].sum() - n:+.0f})",
        DENSITY_CMAP,
        vmin=0,
        vmax_=vmax,
    )
    ax, im_m = add(
        2,
        0,
        parts["mask"],
        f"Router mask  ·  {100 * (parts['mask'] > 0.5).mean():.0f}% > 0.5 "
        f"(oracle zone {100 * zone.mean():.0f}%, outlined)",
        MASK_CMAP,
        vmin=0,
        vmax_=1,
    )
    if zone.any():
        ax.contour(zone.astype(float), levels=[0.5], colors=[INK], linewidths=0.6)
    add(
        2,
        1,
        parts["dense_c"],
        f"Dense expert × mask  ·  {stats['dense_near'] + stats['dense_far']:.0f} "
        f"({stats['dense_far']:.0f} > 2 m from anyone)",
        DENSITY_CMAP,
        vmin=0,
        vmax_=vmax,
    )
    add(
        2,
        2,
        parts["sparse_c"],
        f"Sparse expert × (1 − mask)  ·  {stats['sparse_near'] + stats['sparse_far']:.0f} "
        f"({stats['sparse_far']:.0f} > 2 m)",
        DENSITY_CMAP,
        vmin=0,
        vmax_=vmax,
    )
    norm = TwoSlopeNorm(vcenter=0, vmin=-err_lim, vmax=err_lim)
    add(3, 0, base - gt, "Baseline − GT (red = over)", ERROR_CMAP, norm=norm)
    _, im_e = add(3, 1, parts["pred"] - gt, "Proposed − GT (red = over)", ERROR_CMAP, norm=norm)
    ax = fig.add_subplot(gs[3, 10:15])
    ax.imshow(np.zeros(gt.shape), cmap=DENSITY_CMAP, vmin=0, vmax=1)
    plot_detections(ax, dets, gts, tp_det, tp_gt, size=2.5)
    style_map_axis(
        ax, f"Proposed detections  ·  {len(tp_det)} matched, {len(dets) - len(tp_det)} false, {n - len(tp_gt)} missed"
    )

    for im, x, label in (
        (im_d, 0.13, "density (people per 0.2 m cell)"),
        (im_m, 0.40, "router mask (dense weight)"),
        (im_e, 0.67, "signed error (people per cell)"),
    ):
        cb = fig.colorbar(im, cax=fig.add_axes([x, 0.07, 0.22, 0.008]), orientation="horizontal")
        cb.set_label(label, color=INK_2, fontsize=9)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle(
        f"Proposed vs. baseline, same views  ·  {args.split}/{scene}, snapshot {snapshot}  ·  "
        f"scene MAE proposed {stats['scene_mae_proposed']:.0f}, baseline {stats['scene_mae_baseline']:.0f}",
        x=0.125,
        ha="left",
        y=0.915,
        fontsize=12,
        color=INK,
    )
    fig.savefig(out_path, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def summary_figure(rows, out_path, dpi):
    rows = sorted(rows, key=lambda r: r["pred"] - r["gt"])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    left = np.zeros(len(rows))
    for key, color, label in (
        ("dense_near", DENSE_NEAR, "dense expert, ≤ 2 m from a person"),
        ("sparse_near", SPARSE_NEAR, "sparse expert, ≤ 2 m"),
        ("dense_far", DENSE_FAR, "dense expert, > 2 m (empty ground)"),
        ("sparse_far", SPARSE_FAR, "sparse expert, > 2 m"),
    ):
        vals = np.array([r[key] for r in rows])
        ax.barh(y, vals, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        left += vals
    ax.scatter([r["gt"] for r in rows], y, marker="|", s=420, color=INK, linewidths=2, zorder=3, label="ground truth")
    ax.scatter(
        [r["baseline"] for r in rows],
        y,
        marker="D",
        s=36,
        facecolor=SURFACE,
        edgecolor=INK_2,
        linewidths=1.4,
        zorder=3,
        label="baseline prediction",
    )
    for yi, r in zip(y, rows):
        ax.text(left[yi] + 8, yi, f"{r['pred'] - r['gt']:+.0f}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y, [f"{r['scene']} (snap {r['snapshot']})" for r in rows], fontsize=9, color=INK_2)
    ax.set_xlabel("people (predicted mass of the proposed model, stacked)", color=INK_2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9)
    ax.set_title(
        "Where the proposed model puts its mass  ·  one representative sample per scene, label = proposed error",
        loc="left",
        fontsize=11,
        color=INK,
    )
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model("proposed", args.weights, device)
    baseline, _ = load_model("baseline", args.baseline_weights, device)
    with open(args.eval_json) as f:
        ev = json.load(f)
    with open(args.baseline_eval_json) as f:
        ev_base = json.load(f)
    split_dir = os.path.join(src_dir, "datasets", "SynMVCrowd", "data", args.split)
    scenes = args.scenes.split(",") if args.scenes else sorted(ev["per_scene_MAE"])
    out_dir = args.out_dir or os.path.join(
        src_dir, "figs", "proposed_breakdown_" + os.path.splitext(os.path.basename(args.weights))[0]
    )
    os.makedirs(out_dir, exist_ok=True)
    kernel = get_map_kernel(sigma=5, kernel_size=20)

    rows = []
    for scene in scenes:
        ds = SynMVCrowdDataset(
            scene_dir=os.path.join(split_dir, scene), camera_ids=None, num_sampled_cams=5, load_refs=True
        )
        target = ev["per_scene_MAE"][scene]
        errs = {}
        for idx in tqdm(range(0, len(ds), args.frame_stride), desc=f"Scanning {scene}", mininterval=10):
            s = batched(load_sample(ds, idx, args.seed))
            p = predict_proposed_parts(model, s, device)["pred"]
            errs[idx] = abs(p.sum() - len(gt_pixels(s["gt_world_xy"], s["wld_map_paras"], p.shape)))
        idx = min(errs, key=lambda i: abs(errs[i] - target))

        raw = load_sample(ds, idx, args.seed)
        s = batched(raw)
        parts = predict_proposed_parts(model, s, device)
        base_t = predict_baseline(baseline, s, device)
        base = base_t.cpu().numpy() / SCALE
        wld = s["wld_map_paras"]
        gts = gt_pixels(s["gt_world_xy"], wld, base.shape)
        gt = gt_density(gts, base.shape, kernel)
        pred_t = torch.from_numpy(parts["pred"] * SCALE).to(device)
        dets = detect(pred_t, args.cls_thres, args.nms_radius_px)
        tp_det, tp_gt = match_assign(dets, gts, args.match_radius_m * wld[0, 0].item())

        occ = np.ones(gt.shape, bool)
        if len(gts):
            occ[gts[:, 0].astype(int), gts[:, 1].astype(int)] = False
        far = distance_transform_edt(occ) > FAR_PX
        stats = {
            "scene": scene,
            "snapshot": ds.snapshots[idx],
            "frame_index": idx,
            "gt": len(gts),
            "pred": float(parts["pred"].sum()),
            "baseline": float(base.sum()),
            "dense_near": float(parts["dense_c"][~far].sum()),
            "dense_far": float(parts["dense_c"][far].sum()),
            "sparse_near": float(parts["sparse_c"][~far].sum()),
            "sparse_far": float(parts["sparse_c"][far].sum()),
            "baseline_far": float(base[far].sum()),
            "area_far": float(far.mean()),
            "mask_cov": float((parts["mask"] > 0.5).mean()),
            "oracle_cov": float(oracle_zone(gts, gt.shape).mean()),
            "tp": int(len(tp_det)),
            "fp": int(len(dets) - len(tp_det)),
            "fn": int(len(gts) - len(tp_gt)),
            "scene_mae_proposed": target,
            "scene_mae_baseline": ev_base["per_scene_MAE"][scene],
        }
        rows.append(stats)
        out = os.path.join(out_dir, f"{scene}_snap{stats['snapshot']}.png")
        scene_figure(args, scene, stats["snapshot"], raw, gt, base, parts, gts, dets, tp_det, tp_gt, stats, out)
        print(
            f"{scene}: GT {stats['gt']} | proposed {stats['pred']:.0f} (dense {stats['dense_near']:.0f}+{stats['dense_far']:.0f}far, "
            f"sparse {stats['sparse_near']:.0f}+{stats['sparse_far']:.0f}far) | baseline {stats['baseline']:.0f} "
            f"({stats['baseline_far']:.0f} far) | mask {100 * stats['mask_cov']:.0f}% vs oracle {100 * stats['oracle_cov']:.0f}% -> {out}"
        )

    summary_figure(rows, os.path.join(out_dir, "summary_mass_breakdown.png"), args.dpi)
    with open(os.path.join(out_dir, "breakdown.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"Saved {len(rows)} scene figures, summary_mass_breakdown.png and breakdown.json to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proposed-model error breakdown per scene")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--eval_json", type=str, required=True)
    parser.add_argument("--baseline_weights", type=str, default=os.path.join("runs", "baseline", "run_1", "best.pth"))
    parser.add_argument(
        "--baseline_eval_json", type=str, default=os.path.join("runs", "baseline", "run_1", "eval_test_best.json")
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--scenes", type=str, default=None, help="Comma-separated; default: every scene in the eval JSON"
    )
    parser.add_argument("--frame_stride", type=int, default=10, help="Scan every n-th frame to pick the sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cls_thres", type=float, default=0.2)
    parser.add_argument("--nms_radius_px", type=float, default=2.5)
    parser.add_argument("--match_radius_m", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--out_dir", type=str, default=None)
    main(parser.parse_args())
