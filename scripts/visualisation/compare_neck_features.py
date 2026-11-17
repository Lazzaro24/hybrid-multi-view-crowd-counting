"""
compare_neck_features.py

Side-by-side picture of the neck output (proj_bev, 64 ch) of several checkpoints on the SAME patches -- works for the
legacy neck and the BEVFormer encoder neck. Uses the training-time validation cache (same val_seed, cameras and crops)
and precise-BN like training validation. Per checkpoint (one row) and sample (one figure):

  GT density | PCA of the 64 channels as RGB | channel mean | channel most correlated with GT | head output

The titles carry Cohen's d of the plotted map at GT points vs. background (same test as visualise_neck_features.py).
PCA colours are fitted per checkpoint, so compare structure, not colours, between rows.

Usage (from src):
    uv run python scripts/visualisation/compare_neck_features.py \
        --checkpoints multiscene_best_stage_1_s8_dout0.7_cw0.01_f10_a0.7.pth,multiscene_best_stage_1_s8_bev3.pth \
        --scenes scene_49,scene_38,scene_98
"""

import os
import re
import sys
import json
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(src_dir, "training_scripts"))
sys.path.insert(0, current_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import train_proposed_multiscene as T
from visualise_neck_features import cohens_d, sample_point_vs_background


def gt_pixels(sample, p):
    pts = T.extract_target_coordinates(sample["gt_world_xy"], sample["wld_map_paras"], sample["hw_random"][:, :, p])[
        0, 0
    ]
    rc = (
        np.stack(
            [
                (pts[:, 1] * T.BEV_SHAPE[0]).long().clamp(0, T.BEV_SHAPE[0] - 1).numpy(),
                (pts[:, 0] * T.BEV_SHAPE[1]).long().clamp(0, T.BEV_SHAPE[1] - 1).numpy(),
            ],
            1,
        )
        if len(pts)
        else np.zeros((0, 2), int)
    )
    return rc


def pca_rgb(feat):  # feat: [C, H, W] -> [H, W, 3] in [0, 1]
    C, H, W = feat.shape
    X = feat.reshape(C, -1).T
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(
        X[np.random.default_rng(0).choice(len(X), min(len(X), 20000), replace=False)], full_matrices=False
    )
    Y = X @ Vt[:3].T
    lo, hi = np.percentile(Y, 1, axis=0), np.percentile(Y, 99, axis=0)
    return np.clip((Y - lo) / (hi - lo + 1e-9), 0, 1).reshape(H, W, 3)


def d_title(name, arr, gt_rc):
    pv, bv = sample_point_vs_background(arr, gt_rc)
    return f"{name}\nd={cohens_d(pv, bv):+.2f}" if pv.size else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True, help="comma-separated; 'path@label' sets the row label")
    ap.add_argument(
        "--scenes", default="scene_49,scene_38,scene_98", help="val scenes to show (first cached frame each)"
    )
    ap.add_argument("--patch", type=int, default=0, help="which of the 3 crops of the frame")
    ap.add_argument("--val_seed", type=int, default=T.VAL_SEED_DEFAULT)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out_dir", default=os.path.join(src_dir, "diagnostics", "neck_compare"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams = json.load(f)
    want = [s.strip() for s in args.scenes.split(",") if s.strip()]
    val_reg = [
        e for e in T.build_scene_registry("validate", args.k, cams, {}, validate_setup=False) if e["scene_name"] in want
    ]
    samples, _ = T.build_cached_val_samples(val_reg, 10, val_seed=args.val_seed)  # same draws as training validation
    samples = [
        next(s for s in samples if s["scene_name"] == sc) for sc in want if any(s["scene_name"] == sc for s in samples)
    ]
    train_reg = T.build_scene_registry("train", args.k, cams, {}, validate_setup=False)
    bn_samples, _ = T.build_cached_val_samples(train_reg, 1, val_seed=args.val_seed + 7)

    kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(device)

    rows = []  # per checkpoint: label, per-sample (feat, head)
    for item in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
        path, _, label = item.partition("@")
        label = label or os.path.splitext(os.path.basename(path))[0].replace("multiscene_best_", "")
        m = re.search(r"stage_(\d)", os.path.basename(path))
        stage = 1 if (m and m.group(1) == "1") else 2
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = T.build_model_for_checkpoint(ck, num_cam=args.k).to(device)
        msd = model.state_dict()
        model.load_state_dict(
            {k: v for k, v in ck["model_state_dict"].items() if k not in msd or msd[k].shape == v.shape}, strict=False
        )
        bns = [x for x in model.neck.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
        for x in bns:
            x.reset_running_stats()
            x.momentum = None
        model.eval()
        for x in bns:
            x.train()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            for s in bn_samples:
                model(s["images"].to(device), s["ref_points_full"].to(device), s["hw_random"].to(device), stage=stage)
        model.eval()
        outs = []
        for s in samples:
            feats = []
            h = model.neck.register_forward_hook(lambda mod, i, o: feats.append(o.detach().float().cpu()))
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out, _ = model(
                    s["images"].to(device), s["ref_points_full"].to(device), s["hw_random"].to(device), stage=stage
                )
            h.remove()
            outs.append(
                (
                    feats[args.patch][0].numpy(),
                    out[0, args.patch, 0].float().cpu().numpy(),
                    "Stage 1 head" if stage == 1 else "dense expert",
                )
            )
        rows.append((label, getattr(model.neck, "describe", lambda: "legacy neck")(), outs))
        del model
        torch.cuda.empty_cache()
        print(f"[done] {label}")

    for si, s in enumerate(samples):
        hw = s["hw_random"].to(device)[:, :, args.patch]
        dots = T.create_ground_truth_heatmap(
            s["gt_world_xy"], s["wld_map_paras"].to(device), hw, bev_shape=T.BEV_SHAPE
        ).to(device)
        gt = F.conv2d(dots, kernel, padding=kernel.shape[-1] // 2)[0, 0].cpu().numpy()
        gt_rc = gt_pixels(s, args.patch)
        fig, axes = plt.subplots(len(rows), 5, figsize=(22, 4.4 * len(rows)), squeeze=False)
        for ri, (label, desc, outs) in enumerate(rows):
            feat, head, head_name = outs[si]
            flat = feat.reshape(feat.shape[0], -1)
            cors = np.array([np.corrcoef(c, gt.ravel())[0, 1] if c.std() > 0 else 0 for c in flat])
            best = int(np.nanargmax(np.abs(cors)))
            panels = [
                (gt, f"GT density ({len(gt_rc)} people)", "hot"),
                (pca_rgb(feat), "neck output: PCA top-3 as RGB", None),
                (feat.mean(0), d_title("neck output: channel mean", feat.mean(0), gt_rc), "viridis"),
                (feat[best], d_title(f"channel {best} (corr with GT {cors[best]:+.2f})", feat[best], gt_rc), "viridis"),
                (head, d_title(head_name, head, gt_rc), "hot"),
            ]
            for ci, (img, title, cmap) in enumerate(panels):
                ax = axes[ri, ci]
                ax.imshow(img, cmap=cmap) if cmap else ax.imshow(img)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
            axes[ri, 0].text(
                -0.08,
                0.5,
                label,
                transform=axes[ri, 0].transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=12,
                weight="bold",
            )
        fig.suptitle(f"{s['scene_name']} | snapshot {s['snapshot_id'][0]} | patch {args.patch}", fontsize=14)
        plt.tight_layout()
        out = os.path.join(args.out_dir, f"compare_{s['scene_name']}_p{args.patch}.png")
        plt.savefig(out, dpi=110)
        plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
