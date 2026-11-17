"""
scripts/visualisation/visualize_expert_leakage.py

Expert leakage under each model's OWN oracle mask, for several Stage 2 checkpoints side by side. For each scene
(full scene BEV, one frame, the training/validation camera set) and each checkpoint:

  col 1  GT density, oracle dense-zone outline
  col 2  dense expert output, UNMASKED (what it predicts everywhere, not just inside its zone)
  col 3  sparse expert output, UNMASKED

with the mass split printed in the panel titles and written to a JSON:
  dense_in   dense mass inside the zone (what oracle routing keeps)      vs GT inside
  dense_out  dense mass OUTSIDE the zone = dense leakage (what a router error lets through)
  sparse_in  sparse mass INSIDE the zone = sparse leakage into the dense zone
  sparse_out sparse mass outside the zone (kept)                         vs GT outside
  oracle     dense_in + sparse_out (oracle-routed count)                 vs GT total

Each checkpoint's mask comes from its stored 'oracle_mask_config' (checkpoints from before that field existed were
trained on the default pointwise mask). Colour scale per column is shared across checkpoints within a scene and
gamma-compressed (0.5) so low-level leakage stays visible.

Usage (from src/):
    python scripts/visualisation/visualize_expert_leakage.py --scenes scene_67,scene_18 \
        --weights multiscene_best_stage_2_s8_dout0.7_cw0.01_f10_a0.7.pth,multiscene_best_stage_2_s8_dout0.7_cw0.01_f10_W25D0.4_ma50.pth,multiscene_best_stage_2_s8_dout0.7_cw0.01_f10_W25D0.5_ma50.pth \
        --labels "Current mask,W25 / 0.4 p/m2,W25 / 0.5 p/m2"
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
from matplotlib.colors import PowerNorm

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
from evaluation.evaluate_paper_protocol import load_model, gt_pixels, oracle_zone, SCALE
from losses.gaussian_mse import get_map_kernel
import training_scripts.train_proposed_multiscene as T
from visualize_paper_prediction import SURFACE, INK, INK_2, MUTED, DENSITY_CMAP, batched, gt_density

ZONE_COLOR = "#eb6834"  # categorical slot 2: zone outline on the blue density ramp
DEFAULT_MASK = {
    "density_peak_multiplier": 2.0,
    "smoothing_kernel": 3,
    "min_extent": 1,
    "min_area": 0,
    "density_window": 1,
    "min_density_per_m2": None,
    "occupied_only": False,
}


def apply_mask_config(cfg):
    """Point T's oracle-mask globals at a checkpoint's stored config (None = pre-config checkpoint = default)."""
    cfg = dict(DEFAULT_MASK, **(cfg or {}))
    T.DENSITY_PEAK_MULTIPLIER = cfg["density_peak_multiplier"]
    T.MASK_SMOOTHING_KERNEL = cfg["smoothing_kernel"]
    T.ORACLE_MIN_EXTENT, T.ORACLE_MIN_AREA = cfg["min_extent"], cfg["min_area"]
    T.ORACLE_DENSITY_WINDOW, T.ORACLE_MIN_DENSITY = cfg["density_window"], cfg["min_density_per_m2"]
    T.ORACLE_OCCUPIED_ONLY = cfg["occupied_only"]
    return cfg


@torch.no_grad()
def predict_experts(model, sample, device):
    """Full-scene, unmasked dense and sparse expert outputs in people per pixel (same tiling as
    evaluate_paper_protocol.predict_proposed)."""
    images, refs = sample["images"].to(device), sample["ref_points_full"].to(device)
    wld = sample["wld_map_paras"]
    h_full, w_full = int(wld[0, 3].item()), int(wld[0, 4].item())
    th, tw = model.bev_shape

    def starts(full, tile):
        return sorted(set(list(range(0, full - tile + 1, tile)) + [full - tile]))

    parts = {k: torch.zeros((h_full, w_full), device=device) for k in ("dense", "sparse")}
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        B, N = images.shape[:2]
        feats = model.backbone(images.flatten(0, 1))
        feats = feats.view(B, N, *feats.shape[1:])
        for h in starts(h_full, th):
            for w in starts(w_full, tw):
                bev = model.neck(feats, refs[:, :, h : h + th, w : w + tw])
                parts["dense"][h : h + th, w : w + tw] = F.relu(model.head.dense_expert(bev))[0, 0].float()
                parts["sparse"][h : h + th, w : w + tw] = F.relu(model.head.sparse_expert(bev))[0, 0].float()
    to_people = model.to_scale / SCALE
    return {k: (v * to_people).cpu().numpy() for k, v in parts.items()}


def leakage_stats(gt_dens, dense, sparse, zone):
    z = zone.astype(np.float64)
    s = {
        "gt_total": gt_dens.sum(),
        "gt_in": (gt_dens * z).sum(),
        "gt_out": (gt_dens * (1 - z)).sum(),
        "dense_in": (dense * z).sum(),
        "dense_out": (dense * (1 - z)).sum(),
        "sparse_in": (sparse * z).sum(),
        "sparse_out": (sparse * (1 - z)).sum(),
        "zone_share": z.mean(),
    }
    s["oracle"] = s["dense_in"] + s["sparse_out"]
    return {k: float(v) for k, v in s.items()}


def scene_figure(scene, frame, rows, out_path, dpi):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"]})
    h, w = rows[0]["gt"].shape
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 5 * len(rows) * h / w + 1.2), squeeze=False, facecolor=SURFACE)
    vmax = {k: max(np.percentile(r[k], 99.8) for r in rows) + 1e-6 for k in ("gt", "dense", "sparse")}
    vmax["dense"] = vmax["sparse"] = max(vmax["dense"], vmax["sparse"])  # experts on one scale
    for i, r in enumerate(rows):
        s = r["stats"]
        titles = [
            f"GT · {s['gt_total']:.0f} people · zone {100 * s['zone_share']:.1f}% of scene, "
            f"{s['gt_in']:.0f} people in it",
            f"Dense expert · in zone {s['dense_in']:.0f} (GT {s['gt_in']:.0f}) · leaked outside {s['dense_out']:.0f}",
            f"Sparse expert · outside {s['sparse_out']:.0f} (GT {s['gt_out']:.0f}) · "
            f"leaked into zone {s['sparse_in']:.0f}",
        ]
        for j, key in enumerate(("gt", "dense", "sparse")):
            ax = axes[i, j]
            ax.imshow(r[key], cmap=DENSITY_CMAP, norm=PowerNorm(0.5, vmin=0, vmax=vmax[key]), interpolation="nearest")
            if r["zone"].any():
                ax.contour(r["zone"].astype(float), levels=[0.5], colors=[ZONE_COLOR], linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(MUTED)
                sp.set_linewidth(0.5)
            ax.set_title(titles[j], fontsize=8.5, color=INK_2, loc="left")
            if j == 0:
                ax.set_ylabel(
                    f"{r['label']}\noracle {s['oracle']:.0f} vs GT {s['gt_total']:.0f}", fontsize=10, color=INK
                )
    fig.suptitle(
        f"{scene} · frame {frame} — expert outputs unmasked; orange = each model's oracle dense zone",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.003,
        "People per pixel, gamma 0.5 colour scale shared per column (both experts on one scale) "
        "so low-level leakage is visible. Masses in people.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.98))
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = [w.strip() for w in args.weights.split(",")]
    labels = [s.strip() for s in args.labels.split(",")] if args.labels else weights
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams = json.load(f)
    kernel = get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE)
    models = []
    for wpath, label in zip(weights, labels):
        cfg = torch.load(wpath, map_location="cpu", weights_only=False).get("oracle_mask_config")
        model, _ = load_model("proposed", wpath, device)
        models.append((label, model, cfg))
    os.makedirs(args.out_dir, exist_ok=True)
    report = {}
    for scene in [s.strip() for s in args.scenes.split(",")]:
        ds = SynMVCrowdDataset(
            scene_dir=os.path.join(src_dir, "datasets", "SynMVCrowd", "data", args.split, scene),
            camera_ids=cams[scene][str(5)],
        )
        frame = len(ds) // 2 if args.frame is None else args.frame
        sample = batched(ds[frame])
        wld = sample["wld_map_paras"]
        shape = (int(wld[0, 3].item()), int(wld[0, 4].item()))
        gts = gt_pixels(sample["gt_world_xy"], wld, shape)
        gt = gt_density(gts, shape, kernel)
        rows = []
        for label, model, cfg in models:
            used = apply_mask_config(cfg)
            zone = oracle_zone(T, gts, shape).numpy()
            pred = predict_experts(model, sample, device)
            stats = leakage_stats(gt, pred["dense"], pred["sparse"], zone)
            rows.append({"label": label, "gt": gt, "zone": zone, "stats": stats, **pred})
            report.setdefault(scene, {"frame": frame})[label] = {"mask_config": used, **stats}
            print(
                f"{scene} | {label:>16} | zone {100 * stats['zone_share']:5.2f}% | GT in/out "
                f"{stats['gt_in']:6.1f}/{stats['gt_out']:6.1f} | dense in {stats['dense_in']:6.1f} "
                f"LEAK out {stats['dense_out']:6.1f} | sparse out {stats['sparse_out']:6.1f} LEAK in "
                f"{stats['sparse_in']:6.1f} | oracle {stats['oracle']:6.1f} vs GT {stats['gt_total']:6.1f}"
            )
        scene_figure(scene, frame, rows, os.path.join(args.out_dir, f"leakage_{scene}.png"), args.dpi)
    with open(os.path.join(args.out_dir, "leakage.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="comma-separated Stage 2+ checkpoints")
    ap.add_argument("--labels", default=None, help="comma-separated row labels, same order as --weights")
    ap.add_argument("--scenes", default="scene_67,scene_18")
    ap.add_argument("--split", default="validate")
    ap.add_argument("--frame", type=int, default=None, help="default: middle frame of each scene")
    ap.add_argument("--out_dir", default=os.path.join(src_dir, "figs", "expert_leakage"))
    ap.add_argument("--dpi", type=int, default=130)
    main(ap.parse_args())
