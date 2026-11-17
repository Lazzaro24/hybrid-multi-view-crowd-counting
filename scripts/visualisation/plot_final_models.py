"""
scripts/visualisation/plot_final_models.py

Qualitative comparison of the final models on the same sample (doc/final_evaluations.md E8): the 5 input views, then
one column per model (GT, Baseline (OT), Baseline (MSE), h1_hom, h1_casa, hybrid_hom) with
  row 1  the whole-scene density map (one colour scale for all, clipped at the GT maximum),
  row 2  a zoom into the densest 80 x 80 cells (16 x 16 m) of the GT,
  row 3  the detections in the zoom (paper-protocol post-processing: matched / false / missed),
  row 4  the routing: oracle dense zone (GT column), the router's soft mask (h1), the density rule's zone (hybrid).
Every model sees the same views; predictions use exactly the paths of evaluate_paper_protocol.py.

Frames are picked by a rule, not by eye (--pick):
  sparse     the test frame closest to the median GT count of the least crowded fifth of frames (E1 CSV);
  dense      the test frame closest to the median dense-zone share of the most crowded fifth (E1 CSV);
  occluded   the test frame of the occlusion run (E6) with the largest share of people blocked by others in >= 1 view;
             drawn with that run's K-means cameras, so the occlusion labels apply;
  scene/frame  a given test frame, e.g. scene_7/148.
sparse / dense use a seeded random draw of 5 of the scene's cameras (the paper protocol's camera distribution; not
the same draw as any E1 sample).

Usage (from src):
    uv run python scripts/visualisation/plot_final_models.py --pick dense
Writes figs/final_models_<pick>_<scene>_snap<id>.png.
"""

import os
import sys
import csv
import json
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, "evaluation"))
sys.path.append(os.path.join(src_dir, "scripts", "visualisation"))

import evaluate_paper_protocol as EP
from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
from losses.gaussian_mse import get_map_kernel
from visualize_paper_prediction import (
    DENSITY_CMAP,
    FN_COLOR,
    FP_COLOR,
    IMG_MEAN,
    IMG_STD,
    INK,
    INK_2,
    MUTED,
    SURFACE,
    TP_COLOR,
    densest_window,
    gt_density,
    match_assign,
    plot_detections,
    style_map_axis,
)

# name, label, model type, routing, local weights
MODELS = [
    ("baseline_ot", "Baseline (OT)", "baseline", "learned", "runs/baseline/ot/best.pth"),
    ("baseline_mse", "Baseline (MSE)", "baseline", "learned", "runs/baseline/run_1/best.pth"),
    (
        "h1_hom",
        "h1_hom",
        "proposed",
        "learned",
        "runs/full_length_dense_mask_homography/4_router_anchor/"
        "multiscene_final_stage_4_bud200_homography_256_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth",
    ),
    (
        "h1_casa",
        "h1_casa",
        "proposed",
        "learned",
        "runs/full_length_dense_mask_final/4_router_anchor/"
        "multiscene_final_stage_4_bud200_ca_sa_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth",
    ),
    (
        "hybrid_hom",
        "hybrid_hom (density rule)",
        "proposed",
        "dzone0.1p",
        "runs/full_length_hybrid_results/multiscene_final_stage_2_bud200_homography_256_S4D2_s42.pth",
    ),
]
ZONE_CMAP = "Greys"


def frame_table(csv_path):
    """Per test frame (scene, frame index): mean GT count and mean dense-zone share over its draws (E1 CSV)."""
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            a = acc[(r["scene"], int(r["frame"]))]
            a[0] += float(r["gt"])
            a[1] += float(r["oracle_dense_frac"])
            a[2] += 1
    return {k: (v[0] / v[2], v[1] / v[2]) for k, v in acc.items()}


def pick_frame(args):
    """(scene, frame index, camera ids or None) for --pick."""
    if args.pick in ("sparse", "dense"):
        table = frame_table(args.samples_csv)
        keys = list(table)
        col = 0 if args.pick == "sparse" else 1
        vals = np.array([table[k][col] for k in keys])
        sel = vals <= np.percentile(vals, 20) if args.pick == "sparse" else vals >= np.percentile(vals, 80)
        target = np.median(vals[sel])
        cand = [k for k, s in zip(keys, sel) if s]
        best = min(cand, key=lambda k: (abs(table[k][col] - target), k))
        print(
            f"[pick] {args.pick}: {best[0]} frame {best[1]} (GT {table[best][0]:.0f}, dense share {table[best][1]:.3f})"
        )
        return best[0], best[1], None
    if args.pick == "occluded":
        share = defaultdict(lambda: [0, 0])
        with open(args.occlusion_csv) as f:
            for r in csv.DictReader(f):
                if r["scene"] not in args.test_scenes.split(","):
                    continue
                s = share[(r["scene"], int(r["snapshot"]))]
                s[0] += int(r["n_block_people"]) >= 1
                s[1] += 1
        (scene, snap), (k, n) = max(share.items(), key=lambda kv: (kv[1][0] / kv[1][1], kv[0]))
        with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
            cams = json.load(f)[scene]["5"]
        ds = SynMVCrowdDataset(scene_dir=scene_dir(scene), camera_ids=cams, num_sampled_cams=5, load_refs=False)
        print(f"[pick] occluded: {scene} snapshot {snap} ({k}/{n} people blocked by others in >= 1 view), cams {cams}")
        return scene, [int(s) for s in ds.snapshots].index(snap), cams
    scene, frame = args.pick.split("/")
    return scene, int(frame), None


def scene_dir(scene):
    return os.path.join(src_dir, "datasets", "SynMVCrowd", "data", "test", scene)


@torch.no_grad()
def predict(spec, sample, device):
    name, _, model_type, routing, weights = spec
    model, _ = EP.load_model(model_type, os.path.join(src_dir, weights), device)
    base, peaks = EP.parse_routing(routing)
    if peaks:
        model.T.apply_sparse_render(model, {"mode": "peaks", "kernel": 3, "threshold": 0.064})
        model.head.render_kernel = model.head.render_kernel.to(device)
    shape = (int(sample["wld_map_paras"][0, 3].item()), int(sample["wld_map_paras"][0, 4].item()))
    zone = None
    if model_type == "baseline":
        pred = EP.predict_baseline(model, sample, device)
    elif base == "learned":
        pred, zone = EP.predict_proposed(model, sample, device, None, return_mask=True)
    else:
        zone = EP.density_zone(model, sample, device, float(base[len("dzone") :]), shape)
        pred = EP.predict_proposed(model, sample, device, zone)
    T = getattr(model, "T", None)
    del model
    torch.cuda.empty_cache()
    return pred, (zone.float().cpu().numpy() if zone is not None else None), T


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene, frame, cams = pick_frame(args)
    ds = SynMVCrowdDataset(scene_dir=scene_dir(scene), camera_ids=cams, num_sampled_cams=5, load_refs=True)
    random.seed(args.seed * 100_003 + frame)  # reproducible draw (identity for the K-means set)
    raw = ds[frame]
    sample = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in raw.items()}
    wld = sample["wld_map_paras"]
    shape = (int(wld[0, 3].item()), int(wld[0, 4].item()))
    gts = EP.gt_pixels(sample["gt_world_xy"], wld, shape)
    radius = args.match_radius_m * wld[0, 0].item()

    results, T = [], None
    for spec in MODELS:
        pred_t, zone, t = predict(spec, sample, device)
        T = T or t
        dets = EP.detect(pred_t, args.cls_thres, args.nms_radius_px)
        tp_det, tp_gt = match_assign(dets, gts, radius)
        results.append(
            {"spec": spec, "pred": pred_t.cpu().numpy() / EP.SCALE, "zone": zone, "dets": dets, "tp": (tp_det, tp_gt)}
        )
        r = results[-1]
        print(
            f"  {spec[0]:<12} count {r['pred'].sum():7.1f} (GT {len(gts)}) TP {len(tp_det)} FP {len(dets) - len(tp_det)} "
            f"FN {len(gts) - len(tp_gt)}"
        )
    oracle = EP.oracle_zone(T, gts, shape).float().numpy()
    gt = gt_density(gts, shape, get_map_kernel(sigma=5, kernel_size=20))
    vmax = gt.max()
    zoom = args.zoom
    r0, c0 = densest_window(gt, zoom)

    plt.rcParams.update(
        {"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "text.color": INK}
    )
    ncol = 1 + len(MODELS)
    fig = plt.figure(figsize=(2.6 * ncol, 13.5), facecolor=SURFACE)
    gs = fig.add_gridspec(5, 5 * ncol, height_ratios=[0.55, 1, 1, 1, 1], hspace=0.22, wspace=0.6)
    images = raw["images"].permute(0, 2, 3, 1).numpy() * IMG_STD + IMG_MEAN
    w = 5 * ncol // len(images)
    for v in range(len(images)):
        ax = fig.add_subplot(gs[0, w * v : w * v + w])
        ax.imshow(np.clip(images[v], 0, 1))
        style_map_axis(ax, f"View {v + 1}")

    def crop(m):
        return m[r0 : r0 + zoom, c0 : c0 + zoom]

    columns = [("Ground truth", gt, None, oracle, f"{len(gts)} people")] + [
        (
            r["spec"][1],
            r["pred"],
            r,
            r["zone"],
            f"{r['pred'].sum():.0f} ({r['pred'].sum() - len(gts):+.0f}) · F1 "
            f"{100 * 2 * len(r['tp'][0]) / max(len(r['dets']) + len(gts), 1):.0f}",
        )
        for r in results
    ]
    for c, (label, dmap, r, zone, sub) in enumerate(columns):
        sl = slice(5 * c, 5 * c + 5)
        ax = fig.add_subplot(gs[1, sl])
        im = ax.imshow(dmap, cmap=DENSITY_CMAP, vmin=0, vmax=vmax, interpolation="nearest")
        ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), zoom, zoom, fill=False, ec=INK, lw=0.7, ls="--"))
        style_map_axis(ax, f"{label}\n{sub}")
        ax = fig.add_subplot(gs[2, sl])
        ax.imshow(crop(dmap), cmap=DENSITY_CMAP, vmin=0, vmax=vmax, interpolation="nearest")
        style_map_axis(ax, "zoom" if c == 0 else "")
        ax = fig.add_subplot(gs[3, sl])
        ax.imshow(np.zeros((zoom, zoom)), cmap=DENSITY_CMAP, vmin=0, vmax=1)
        if r is not None:
            plot_detections(ax, r["dets"], gts, *r["tp"], window=(r0, c0, zoom), size=9)
        else:
            g = gts[(gts[:, 0] >= r0) & (gts[:, 0] < r0 + zoom) & (gts[:, 1] >= c0) & (gts[:, 1] < c0 + zoom)]
            ax.scatter(g[:, 1] - c0, g[:, 0] - r0, s=9, c=INK_2, marker=".", linewidths=0)
        style_map_axis(ax, "detections (zoom)" if c == 0 else "")
        ax = fig.add_subplot(gs[4, sl])
        if zone is not None:
            ax.imshow(zone, cmap=ZONE_CMAP, vmin=0, vmax=1, interpolation="nearest")
            if c > 0:
                ax.contour(oracle, levels=[0.5], colors=[FP_COLOR], linewidths=0.5)
            style_map_axis(
                ax, "oracle dense zone" if c == 0 else ("router mask" if r["spec"][3] == "learned" else "rule zone")
            )
        else:
            ax.axis("off")
    cax = fig.add_axes([0.13, 0.06, 0.3, 0.008])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("density (people per 0.2 m cell), clipped at the GT maximum", color=INK_2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    legend = [
        Line2D([], [], ls="", marker="o", color=TP_COLOR, label="matched (TP)"),
        Line2D([], [], ls="", marker="x", color=FP_COLOR, label="false (FP)"),
        Line2D([], [], ls="", marker="^", color=FN_COLOR, label="missed (FN)"),
        Line2D([], [], color=FP_COLOR, lw=0.8, label="oracle zone outline"),
    ]
    fig.legend(handles=legend, loc="lower right", bbox_to_anchor=(0.9, 0.045), ncol=4, frameon=False, fontsize=8)
    snapshot = ds.snapshots[frame]
    fig.suptitle(
        f"test/{scene}, snapshot {snapshot} ({args.pick}; {'K-means' if cams else 'random'} 5 views) · zoom "
        f"{zoom * 0.2:.0f} × {zoom * 0.2:.0f} m · count (error) · F1 at threshold {args.cls_thres}",
        x=0.125,
        ha="left",
        y=0.915,
        fontsize=10,
        color=INK,
    )
    tag = args.pick.replace("/", "_")
    out = args.out or os.path.join(src_dir, "figs", f"final_models_{tag}_{scene}_snap{snapshot}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Qualitative comparison of the final models on one test frame.")
    ap.add_argument("--pick", default="dense", help="sparse | dense | occluded | <scene>/<frame index>")
    ap.add_argument("--samples_csv", default=os.path.join(src_dir, "runs", "final_evals", "baseline_ot.samples.csv"))
    ap.add_argument(
        "--occlusion_csv", default=os.path.join(src_dir, "diagnostics", "occlusion", "heldout14_final", "persons.csv")
    )
    ap.add_argument(
        "--test_scenes",
        default="scene_5,scene_7,scene_11,scene_12,scene_13,scene_16,scene_29,scene_50,scene_78,scene_80",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zoom", type=int, default=80)
    ap.add_argument("--cls_thres", type=float, default=0.2)
    ap.add_argument("--nms_radius_px", type=float, default=2.5)
    ap.add_argument("--match_radius_m", type=float, default=0.5)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
