"""
visualise_neck_features.py

Visual checks of the legacy neck's output plus three longitudinal diagnostics:

  1. FIXED EVALUATION PATCH: bypasses the DataLoader to fetch the exact same
     (scene, snapshot, camera set, crop offset) every time it's run, so
     checkpoint-to-checkpoint comparisons aren't contaminated by the dataset's
     per-call random crop/camera sampling.

  2. POINT-VS-BACKGROUND CONTRAST: computes Cohen's d between feature values
     sampled at real GT point pixels vs. randomly sampled background pixels
     (excluded from a small radius around every point). This is the direct
     test of "does the Neck's output actually encode person-location signal",
     which spatial-smoothness visuals alone cannot answer -- a smooth field
     can still be completely uncorrelated with where people are.

  3. ATTENTION-INTERNALS HEALTH CHECKS: forward hooks on the offset/weight
     Linear layers inside both DeformableSelfAttention and
     DeformableCrossAttention, reporting offset magnitude (mean abs, std) and
     attention weight entropy (mean, as a fraction of max possible entropy
     for that K). Collapsing offsets (~0 everywhere) or entropy stuck at
     max (uniform weighting, never sharpens) are early warning signs of a
     dead/uninformative attention branch, independent of anything visible in
     the final feature map.

Every run appends one row to a CSV log (default: neck_diagnostics_log.csv),
so the numbers can be compared across checkpoints.

Usage:
    python scripts/visualisation/visualise_neck_features.py --checkpoint debug_best_stage_1.pth --label epoch60
"""

import os
import sys
import csv
import random
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(src_dir)

from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
from models.backbone.feature_extractor import DilatedResNet18
from models.neck.proposed.spatial_cross_attention import SpatialCrossAttentionNeck
from models.head.proposed.routing import HybridRoutingHead

# ==========================================
# CONFIG -- must match train.py exactly
# ==========================================
NUM_CAMS_TO_SAMPLE = 5
BEV_SHAPE = (200, 200)
NUM_Z_ANCHORS = 4  # also = num_points for both attention modules in this architecture

# ---- FIXED EVALUATION SAMPLE (edit once, then keep constant across runs) ----
SCENE_NAME = "scene_14"  # substring match against scene folder names, like train.py's scene_2 lookup
SNAPSHOT_INDEX = 0  # index into the scene's (already ::10 subsampled) snapshot list
CROP_SEED = 1234  # fixes hw_random's crop offset via random.seed() -- keep constant across runs
PATCH_IDX = 0  # which of the patch_num random crops (all seeded identically) to inspect

CHANNELS_TO_INSPECT = [0, 16, 32, 48]
ROW_TO_PLOT = 100

BACKGROUND_EXCLUSION_RADIUS = 3  # pixels around each GT point excluded from "background" sampling
N_BACKGROUND_SAMPLES = 3000
CONTRAST_SEED = 42


class DiagnosticModel(nn.Module):
    def __init__(self, num_cam=NUM_CAMS_TO_SAMPLE):
        super().__init__()
        self.backbone = DilatedResNet18(pretrained=False)  # overwritten by checkpoint below
        self.neck = SpatialCrossAttentionNeck(
            in_channels=512, out_channels=64, full_bev_shape=BEV_SHAPE, num_cameras=num_cam, num_z_anchors=NUM_Z_ANCHORS
        )
        self.head = HybridRoutingHead(in_channels=64, hidden_channels=64)

    def forward(self, images, patch_ref_points):
        B, N, C, H, W = images.shape
        images_flat = images.view(B * N, C, H, W)
        features_flat = self.backbone(images_flat)
        _, C_f, H_f, W_f = features_flat.shape
        features = features_flat.view(B, N, C_f, H_f, W_f)

        proj_bev = self.neck(features, patch_ref_points)

        dense_pred = F.softplus(self.head.dense_expert(proj_bev))
        sparse_pred = F.softplus(self.head.sparse_expert(proj_bev))

        return proj_bev, dense_pred, sparse_pred


# ==========================================
# 1. FIXED EVALUATION PATCH
# ==========================================


def get_fixed_sample(device):
    """Bypasses the DataLoader entirely to fetch a deterministic sample:
    fixed scene, fixed snapshot, deterministic (sorted) camera set via
    is_caching=True, and a fixed crop offset via a seeded random module
    right before the dataset call."""
    import json

    config_path = os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras.json")
    with open(config_path, "r") as f:
        optimal_cameras_config = json.load(f)

    train_dir = os.path.join(src_dir, "datasets", "SynMVCrowd", "data", "train")
    scene_folders = [
        os.path.join(train_dir, d) for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    ]

    scene_folder = None
    for folder in scene_folders:
        if SCENE_NAME.lower() in os.path.basename(folder).lower():
            scene_folder = folder
            break
    if scene_folder is None:
        raise ValueError(f"Could not find a scene folder matching '{SCENE_NAME}' under {train_dir}")

    scene_name = os.path.basename(scene_folder)
    cam_ids = optimal_cameras_config.get(scene_name, {}).get(str(NUM_CAMS_TO_SAMPLE))

    dataset = SynMVCrowdDataset(scene_dir=scene_folder, camera_ids=cam_ids)
    dataset.snapshots = dataset.snapshots[::10]
    dataset.is_caching = True  # forces deterministic sorted(valid_cams) camera selection

    if SNAPSHOT_INDEX >= len(dataset.snapshots):
        raise ValueError(f"SNAPSHOT_INDEX={SNAPSHOT_INDEX} out of range (scene has {len(dataset.snapshots)} snapshots)")
    dataset.snapshots = [dataset.snapshots[SNAPSHOT_INDEX]]  # restrict to just the target snapshot

    random.seed(CROP_SEED)  # fixes hw_random's random.sample() calls inside __getitem__
    sample = dataset[0]

    print(
        f"[fixed sample] scene={scene_name} snapshot_id={sample['snapshot_id']} "
        f"cameras={cam_ids} crop_seed={CROP_SEED} patch_idx={PATCH_IDX}"
    )

    # Manually add the batch dimension the DataLoader would normally add via default_collate.
    images = sample["images"].unsqueeze(0).to(device)
    wld_map_paras = sample["wld_map_paras"].unsqueeze(0).to(device)
    hw_random = sample["hw_random"].unsqueeze(0).to(device)
    ref_points_full = sample["ref_points_full"].unsqueeze(0).to(device)
    gt_world_xy = sample["gt_world_xy"]  # left 2D -- helper functions handle this shape directly

    return images, wld_map_paras, hw_random, ref_points_full, gt_world_xy


def create_ground_truth_heatmap(gt_world_xy, wld_map_paras, hw_offset, bev_shape=BEV_SHAPE):
    heatmap = np.zeros(bev_shape, dtype=np.float32)
    if gt_world_xy.dim() == 3:
        gt_world_xy = gt_world_xy.squeeze(0)
    if wld_map_paras.dim() == 2:
        wld_map_paras = wld_map_paras.squeeze(0)
    if hw_offset.dim() == 2:
        hw_offset = hw_offset.squeeze(0)
    if gt_world_xy.shape[0] == 0:
        return torch.zeros((1, 1, bev_shape[0], bev_shape[1]))
    r, a, b = wld_map_paras[0].item(), wld_map_paras[1].item(), wld_map_paras[2].item()
    w_min, h_min = wld_map_paras[6].item(), wld_map_paras[7].item()
    h_start, w_start = hw_offset[0].item(), hw_offset[1].item()
    for i in range(gt_world_xy.shape[0]):
        col = int(round((gt_world_xy[i, 0].item() - w_min + a) * r)) - w_start
        row = int(round((gt_world_xy[i, 1].item() - h_min + b) * r)) - h_start
        if 0 <= row < bev_shape[0] and 0 <= col < bev_shape[1]:
            heatmap[row, col] += 1.0
    return torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)


def get_gt_pixel_coords(gt_world_xy, wld_map_paras, hw_offset, bev_shape=BEV_SHAPE):
    """Same pixel math as create_ground_truth_heatmap, but returns an (N, 2) array
    of [row, col] integer coordinates instead of accumulating into a heatmap."""
    if gt_world_xy.dim() == 3:
        gt_world_xy = gt_world_xy.squeeze(0)
    if wld_map_paras.dim() == 2:
        wld_map_paras = wld_map_paras.squeeze(0)
    if hw_offset.dim() == 2:
        hw_offset = hw_offset.squeeze(0)
    if gt_world_xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    r, a, b = wld_map_paras[0].item(), wld_map_paras[1].item(), wld_map_paras[2].item()
    w_min, h_min = wld_map_paras[6].item(), wld_map_paras[7].item()
    h_start, w_start = hw_offset[0].item(), hw_offset[1].item()
    coords = []
    for i in range(gt_world_xy.shape[0]):
        col = int(round((gt_world_xy[i, 0].item() - w_min + a) * r)) - w_start
        row = int(round((gt_world_xy[i, 1].item() - h_min + b) * r)) - h_start
        if 0 <= row < bev_shape[0] and 0 <= col < bev_shape[1]:
            coords.append([row, col])
    return np.array(coords, dtype=np.int64)


# ==========================================
# 2. POINT-VS-BACKGROUND CONTRAST TEST
# ==========================================


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    pooled_std = np.sqrt(pooled_var + 1e-12)
    return (a.mean() - b.mean()) / pooled_std


def sample_point_vs_background(
    feature_2d,
    gt_pixel_coords,
    exclusion_radius=BACKGROUND_EXCLUSION_RADIUS,
    n_background=N_BACKGROUND_SAMPLES,
    seed=CONTRAST_SEED,
):
    H, W = feature_2d.shape
    if gt_pixel_coords.shape[0] == 0:
        return np.array([]), np.array([])

    point_vals = feature_2d[gt_pixel_coords[:, 0], gt_pixel_coords[:, 1]]

    mask = np.ones((H, W), dtype=bool)
    for row, col in gt_pixel_coords:
        r0, r1 = max(0, row - exclusion_radius), min(H, row + exclusion_radius + 1)
        c0, c1 = max(0, col - exclusion_radius), min(W, col + exclusion_radius + 1)
        mask[r0:r1, c0:c1] = False

    bg_candidates = np.argwhere(mask)
    rng = np.random.default_rng(seed)
    n = min(n_background, len(bg_candidates))
    chosen_idx = rng.choice(len(bg_candidates), size=n, replace=False)
    chosen = bg_candidates[chosen_idx]
    bg_vals = feature_2d[chosen[:, 0], chosen[:, 1]]

    return point_vals, bg_vals


# ==========================================
# 3. ATTENTION-INTERNALS HEALTH CHECKS
# ==========================================


class AttentionProbe:
    """Forward hooks on the raw offset/weight Linear layers inside both
    attention modules. Captures pre-transform outputs; compute_stats()
    re-applies the exact same scale/softmax the real modules use."""

    def __init__(self, model):
        self.raw = {}
        self.hooks = []
        self._register(model.neck.self_attention.offset_predictor, "self_offset_raw")
        self._register(model.neck.self_attention.weight_predictor, "self_weight_raw")
        self._register(model.neck.cross_attention.offset_predictor, "cross_offset_raw")
        self._register(model.neck.cross_attention.weight_predictor, "cross_weight_raw")

    def _register(self, module, key):
        def hook(mod, inp, out):
            self.raw[key] = out.detach()

        self.hooks.append(module.register_forward_hook(hook))

    def remove(self):
        for h in self.hooks:
            h.remove()

    def compute_stats(self, B, L, num_cameras, num_points_self=NUM_Z_ANCHORS, num_points_cross=NUM_Z_ANCHORS):
        stats = {}

        # --- Self-attention ---
        self_offsets = self.raw["self_offset_raw"].view(B, L, num_points_self, 2) * 0.1
        stats["self_offset_mean_abs"] = self_offsets.abs().mean().item()
        stats["self_offset_std"] = self_offsets.std().item()

        self_weight_logits = self.raw["self_weight_raw"].view(B, L, num_points_self)
        self_weights = F.softmax(self_weight_logits, dim=-1)
        self_entropy = -(self_weights * self_weights.clamp_min(1e-12).log()).sum(dim=-1)
        max_entropy_self = float(np.log(num_points_self))
        stats["self_weight_entropy_mean"] = self_entropy.mean().item()
        stats["self_weight_entropy_frac_of_max"] = self_entropy.mean().item() / max_entropy_self

        # --- Cross-attention ---
        cross_offsets = self.raw["cross_offset_raw"].view(B, L, num_cameras, num_points_cross, 2) * 0.1
        stats["cross_offset_mean_abs"] = cross_offsets.abs().mean().item()
        stats["cross_offset_std"] = cross_offsets.std().item()

        cross_weight_logits = self.raw["cross_weight_raw"].view(B, L, num_cameras, num_points_cross)
        cross_weights = F.softmax(cross_weight_logits, dim=-1)
        cross_entropy = -(cross_weights * cross_weights.clamp_min(1e-12).log()).sum(dim=-1)
        max_entropy_cross = float(np.log(num_points_cross))
        stats["cross_weight_entropy_mean"] = cross_entropy.mean().item()
        stats["cross_weight_entropy_frac_of_max"] = cross_entropy.mean().item() / max_entropy_cross

        return stats


# ==========================================
# MAIN
# ==========================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--label", type=str, default=None, help='Free-text label for this run (e.g. "epoch60") -- shown in the CSV log'
    )
    parser.add_argument("--output_dir", type=str, default="diagnostics_neck_v3")
    parser.add_argument("--log_csv", type=str, default="neck_diagnostics_log.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = DiagnosticModel(num_cam=NUM_CAMS_TO_SAMPLE).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"[info] missing keys on load (should be empty): {missing}")
    print(f"[info] unexpected keys on load (expected: throwaway_head.*): {unexpected}")
    model.eval()

    images, wld_map_paras, hw_random, ref_points_full, gt_world_xy = get_fixed_sample(device)

    hw_offset = hw_random[:, :, PATCH_IDX]
    h_start, w_start = hw_offset[0, 0].item(), hw_offset[0, 1].item()
    patch_ref_points = ref_points_full[:, :, h_start : h_start + BEV_SHAPE[0], w_start : w_start + BEV_SHAPE[1], :, :]

    probe = AttentionProbe(model)
    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            proj_bev, dense_pred, sparse_pred = model(images, patch_ref_points)
    proj_bev = proj_bev.float()
    dense_pred = dense_pred.float()
    sparse_pred = sparse_pred.float()

    B, L = images.shape[0], BEV_SHAPE[0] * BEV_SHAPE[1]
    attn_stats = probe.compute_stats(B=B, L=L, num_cameras=images.shape[1])
    probe.remove()

    # ---- Ground truth for this exact patch ----
    target_patch = create_ground_truth_heatmap(gt_world_xy, wld_map_paras, hw_offset, bev_shape=BEV_SHAPE).to(device)
    gt_count = target_patch.sum().item()
    gt_pixel_coords = get_gt_pixel_coords(gt_world_xy, wld_map_paras, hw_offset, bev_shape=BEV_SHAPE)

    proj_bev_np = proj_bev.cpu().numpy()[0]  # [64, 200, 200]
    channel_mean = proj_bev_np.mean(axis=0)
    dense_np = dense_pred.cpu().numpy()[0, 0]
    sparse_np = sparse_pred.cpu().numpy()[0, 0]

    # ---- Point-vs-background contrast ----
    contrast_stats = {}
    for name, arr in [("proj_bev_channel_mean", channel_mean), ("dense_pred", dense_np), ("sparse_pred", sparse_np)]:
        pt_vals, bg_vals = sample_point_vs_background(arr, gt_pixel_coords)
        d = cohens_d(pt_vals, bg_vals) if pt_vals.size > 0 else float("nan")
        contrast_stats[f"{name}_cohens_d"] = d
        contrast_stats[f"{name}_point_mean"] = float(pt_vals.mean()) if pt_vals.size > 0 else float("nan")
        contrast_stats[f"{name}_bg_mean"] = float(bg_vals.mean()) if bg_vals.size > 0 else float("nan")

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print(
        f"FIXED-PATCH DIAGNOSTIC SUMMARY | GT count in patch: {gt_count:.0f} | GT points used for contrast: {len(gt_pixel_coords)}"
    )
    print("=" * 70)
    print("\nPoint-vs-background contrast (Cohen's d -- 0 = no separation, |d|>0.5 = meaningful, |d|>0.8 = large):")
    for name in ["proj_bev_channel_mean", "dense_pred", "sparse_pred"]:
        d = contrast_stats[f"{name}_cohens_d"]
        pm = contrast_stats[f"{name}_point_mean"]
        bm = contrast_stats[f"{name}_bg_mean"]
        print(f"  {name:24s} d={d:+.3f}  (point_mean={pm:.4f}, bg_mean={bm:.4f})")

    print("\nAttention internals:")
    print(
        f"  Self-attention  | offset mean|abs|={attn_stats['self_offset_mean_abs']:.4f} std={attn_stats['self_offset_std']:.4f} | "
        f"weight entropy={attn_stats['self_weight_entropy_mean']:.4f} "
        f"({attn_stats['self_weight_entropy_frac_of_max'] * 100:.1f}% of max -- 100% = uniform/uninformative)"
    )
    print(
        f"  Cross-attention | offset mean|abs|={attn_stats['cross_offset_mean_abs']:.4f} std={attn_stats['cross_offset_std']:.4f} | "
        f"weight entropy={attn_stats['cross_weight_entropy_mean']:.4f} "
        f"({attn_stats['cross_weight_entropy_frac_of_max'] * 100:.1f}% of max -- 100% = uniform/uninformative)"
    )

    # ---- Append to CSV log ----
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": args.label or "",
        "checkpoint": args.checkpoint,
        "gt_count": gt_count,
        **contrast_stats,
        **attn_stats,
    }
    write_header = not os.path.exists(args.log_csv)
    with open(args.log_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\n[info] appended row to {args.log_csv}")

    # ---- Visualizations (same fixed patch every run) ----
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    axes = axes.flatten()

    im = axes[0].imshow(target_patch.cpu().numpy()[0, 0], cmap="hot")
    axes[0].set_title(f"Raw GT points (count={gt_count:.0f})")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    for idx, c in enumerate(CHANNELS_TO_INSPECT):
        im = axes[1 + idx].imshow(proj_bev_np[c], cmap="viridis")
        axes[1 + idx].set_title(f"proj_bev channel {c}")
        plt.colorbar(im, ax=axes[1 + idx], fraction=0.046)

    im = axes[5].imshow(channel_mean, cmap="viridis")
    axes[5].set_title("proj_bev channel-mean (all 64 ch)")
    plt.colorbar(im, ax=axes[5], fraction=0.046)

    im = axes[6].imshow(dense_np, cmap="hot")
    axes[6].set_title(f"Dense Expert pred\nCohen's d={contrast_stats['dense_pred_cohens_d']:+.3f}")
    plt.colorbar(im, ax=axes[6], fraction=0.046)

    im = axes[7].imshow(sparse_np, cmap="hot")
    axes[7].set_title(f"Sparse Expert pred\nCohen's d={contrast_stats['sparse_pred_cohens_d']:+.3f}")
    plt.colorbar(im, ax=axes[7], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    label_suffix = f"_{args.label}" if args.label else ""
    out_path = os.path.join(args.output_dir, f"fixed_patch{label_suffix}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[info] saved visualization -> {out_path}")

    # Staircase line plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    row_idx = ROW_TO_PLOT
    axes[0].plot(proj_bev_np[CHANNELS_TO_INSPECT[0]][row_idx, :], color="teal")
    axes[0].set_title(f"proj_bev channel {CHANNELS_TO_INSPECT[0]}, row {row_idx}")
    axes[1].plot(dense_np[row_idx, :], color="firebrick")
    axes[1].set_title(f"Dense Expert pred, row {row_idx}")
    axes[2].plot(sparse_np[row_idx, :], color="darkorange")
    axes[2].set_title(f"Sparse Expert pred, row {row_idx}")
    axes[2].set_xlabel("column index")
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, f"fixed_patch_staircase{label_suffix}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[info] saved staircase plot -> {out_path}")


if __name__ == "__main__":
    main()
