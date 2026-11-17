"""
train_proposed_sparse_only.py

Sanity check for the sparse-head Generalized Loss (doc/sparse_ot_diagnose.md): a model with NO dense expert and
NO router -- backbone -> neck -> a single sparse head (same architecture as HybridRoutingHead.sparse_expert) --
trained on top of the Stage 1 weights with the neck/layer4 unfrozen and only the sparse OT loss in the objective.

Data pipeline, ground-truth construction, validation cache and NMS/F1 matching are reused from
train_proposed_multiscene.py (imported as T) so numbers are directly comparable with the multiscene runs.
The sparse head's parameters are named head.sparse_expert.*, so checkpoints load into ProposedModel
(e.g. visualize_experts.py --allow_missing, diagnose_sparse_expert.py) with the dense expert simply missing.

Usage (from src/):
    uv run training_scripts/train_proposed_sparse_only.py --resume multiscene_best_stage_1.pth --skip_camera_validation
"""

import os
import sys
import gc
import json
import time
import random
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import train_proposed_multiscene as T  # noqa: E402  (also puts src/ on sys.path)


class SparseHead(nn.Module):
    """Wrapper so parameter names are head.sparse_expert.* -- identical to ProposedModel's."""

    def __init__(self, in_channels=64, hidden_channels=64):
        super().__init__()
        self.sparse_expert = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return torch.relu(self.sparse_expert(x))


class SparseOnlyModel(nn.Module):
    def __init__(self, num_cam):
        super().__init__()
        self.backbone = T.DilatedResNet18(pretrained=True)
        self.neck = T.SpatialCrossAttentionNeck(
            in_channels=512, out_channels=64, full_bev_shape=T.BEV_SHAPE, num_cameras=num_cam, num_z_anchors=4
        )
        self.head = SparseHead(in_channels=64, hidden_channels=64)

    def set_trainability(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.backbone.feature_extractor[-1].parameters():
            p.requires_grad = True
        for p in self.neck.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def train_mode(self):
        self.train()
        # frozen early backbone layers keep their BatchNorm statistics fixed
        self.backbone.feature_extractor[:-1].eval()

    def forward(self, images, ref_points_full, hw_random):
        B, N, C, H, W = images.shape
        features_flat = self.backbone(images.view(B * N, C, H, W))
        _, C_f, H_f, W_f = features_flat.shape
        features = features_flat.view(B, N, C_f, H_f, W_f)

        out = []
        for p in range(hw_random.shape[-1]):
            h_start, w_start = hw_random[0, 0, p].item(), hw_random[0, 1, p].item()
            patch_refs = ref_points_full[
                :, :, h_start : h_start + T.BEV_SHAPE[0], w_start : w_start + T.BEV_SHAPE[1], :, :
            ]
            out.append(self.head(self.neck(features, patch_refs)))
        return torch.stack(out, dim=1)  # [B, P, 1, H, W]


def load_stage1(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    keep = {k: v for k, v in sd.items() if k.startswith(("backbone.", "neck."))}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    missing_trunk = [k for k in missing if k.startswith(("backbone.", "neck."))]
    print(
        f"[load] {os.path.basename(path)}: loaded {len(keep)} backbone/neck tensors "
        f"(dropped {len(sd) - len(keep)} head/throwaway tensors); head.sparse_expert starts fresh."
    )
    if missing_trunk:
        raise RuntimeError(f"{len(missing_trunk)} backbone/neck keys missing in {path} (e.g. {missing_trunk[:3]})")
    if unexpected:
        print(f"[load] unexpected keys: {unexpected[:5]}")


def patch_targets(gt_world_xy, wld_paras, hw_offset, cam_locations, map_kernel, single_person_peak, device):
    target_dots = T.extract_target_coordinates(gt_world_xy, wld_paras, hw_offset, bev_shape=T.BEV_SHAPE).to(device)
    patch_cam_locs = T.project_cameras_to_patch(cam_locations, wld_paras, hw_offset, bev_shape=T.BEV_SHAPE).to(device)
    target_patch = T.create_ground_truth_heatmap(gt_world_xy, wld_paras, hw_offset, bev_shape=T.BEV_SHAPE).to(device)
    map_gt_conv = F.conv2d(target_patch, map_kernel, padding=map_kernel.shape[-1] // 2)

    # oracle dense mask, identical to the multiscene script -- only used to restrict supervision/metrics to the
    # sparse region when --region sparse, and always for the sparse-region F1 (comparable to the earlier runs)
    gt_mask = T.oracle_mask_from_conv(map_gt_conv)
    return target_dots, patch_cam_locs, map_gt_conv, gt_mask


def points_in_region(target_dots, region):
    """GT points (pixel x, y) that fall inside region ([1,1,H,W], 1 = keep)."""
    pts = target_dots[0, 0]
    if pts.shape[0] == 0:
        return np.zeros((0, 2))
    h, w = region.shape[-2], region.shape[-1]
    cols = (pts[:, 0] * w).long().clamp(0, w - 1)
    rows = (pts[:, 1] * h).long().clamp(0, h - 1)
    keep = region[0, 0, rows, cols] > 0.5
    xy = pts[keep] * torch.tensor([w, h], device=pts.device, dtype=pts.dtype)
    return xy.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="multiscene_best_stage_1.pth", help="Stage 1 checkpoint (backbone + neck)")
    parser.add_argument("--epochs", type=int, default=13)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--k", type=int, default=T.NUM_CAMS_TO_SAMPLE)
    parser.add_argument("--frames_per_scene", type=int, default=T.FRAMES_PER_SCENE_PER_EPOCH)
    parser.add_argument("--val_frames_per_scene", type=int, default=T.VAL_FRAMES_PER_SCENE)
    parser.add_argument("--val_seed", type=int, default=T.VAL_SEED_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_camera_validation", action="store_true")
    parser.add_argument(
        "--precise_bn_frames",
        type=int,
        default=0,
        help=">0: re-estimate trainable BatchNorm stats over this many frames per train scene before each validation",
    )
    # loss (defaults = the finalized recipe from doc/sparse_ot_diagnose.md)
    parser.add_argument(
        "--region",
        choices=["full", "sparse"],
        default="full",
        help="'full' = supervise every person in the patch (no dense expert exists to take the crowded "
        "blobs); 'sparse' = only outside the oracle dense mask, exactly like the multiscene genloss runs",
    )
    parser.add_argument("--genloss_weight", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--bg_weight", type=float, default=0.05)
    parser.add_argument("--count_weight", type=float, default=1.0)
    # evaluation / selection
    parser.add_argument("--nms_mult", type=float, default=1.0)
    parser.add_argument("--nms_kernel", type=int, default=7)
    parser.add_argument("--match_px", type=float, default=5.0)
    parser.add_argument("--select", choices=["sparse_f1", "full_f1", "count_mae"], default="sparse_f1")
    parser.add_argument("--tag", default="")
    parser.add_argument("--log_file", default=None)
    args = parser.parse_args()

    log_path = args.log_file or os.path.join(
        current_dir, f"sparseonly_training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    log_handle = open(log_path, "a", buffering=1)
    sys.stdout = T.Tee(sys.__stdout__, log_handle)
    print(f"Logging all output to: {log_path}")
    print(f"[config] {vars(args)}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume = args.resume if os.path.isabs(args.resume) else os.path.join(T.src_dir, args.resume)
    assert os.path.isfile(resume), f"--resume checkpoint does not exist: {resume!r}"

    # ---- data (same registries / exclusions / camera selection as the multiscene script)
    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        optimal_cameras_config = json.load(f)
    z_sign_path = os.path.join(T.src_dir, "utils", "z_sign_corrections_full.json")
    z_sign_corrections = json.load(open(z_sign_path)) if os.path.exists(z_sign_path) else {}
    T.validate_excluded_frame_keys()
    validate = not args.skip_camera_validation
    train_registry = T.build_scene_registry("train", args.k, optimal_cameras_config, z_sign_corrections, validate)
    val_registry = T.build_scene_registry("validate", args.k, optimal_cameras_config, z_sign_corrections, validate)
    print(f"Registries built: {len(train_registry)} train scenes, {len(val_registry)} val scenes.")

    # ---- model
    model = SparseOnlyModel(num_cam=args.k).to(device)
    load_stage1(model, resume, device)
    model.set_trainability()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[model] backbone -> neck -> sparse head only (no dense expert, no router); "
        f"{n_train:,} trainable parameters (layer4 + neck + sparse head)"
    )

    ot_criterion = T.GeneralizedLoss(loss_type="md", reduction="mean", scale=T.OT_SCALE, device=device, top_k=args.topk)
    map_kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(device)
    mass_per_person = map_kernel.sum().item()
    single_person_peak = T.GAUSSIAN_SCALE / (2 * torch.pi * (T.GAUSSIAN_SIGMA**2))
    nms_threshold = single_person_peak * T.PEAK_THRESHOLD_FRACTION * args.nms_mult

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=T.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    cached_val, _ = T.build_cached_val_samples(
        val_registry, args.val_frames_per_scene, val_seed=(args.val_seed if args.val_seed >= 0 else None)
    )
    print(f"[val] {len(cached_val)} cached validation samples.")

    tag = args.tag
    best_metric, epochs_without_improvement = float("inf"), 0
    train_loaders = None

    for epoch in range(args.epochs):
        t_epoch = time.perf_counter()
        # ---------------- train
        model.train_mode()
        if train_loaders is not None:
            train_loaders = None
            gc.collect()
        train_loaders = [
            (e["scene_name"], T.sample_train_subset_loader(e, args.frames_per_scene)) for e in train_registry
        ]
        random.shuffle(train_loaders)

        sums = {"loss": 0.0, "ot": 0.0, "bg": 0.0, "count": 0.0}
        n_batches = 0
        for scene_name, loader in train_loaders:
            pbar = tqdm(loader, desc=f"SparseOnly | Epoch {epoch + 1}/{args.epochs} | {scene_name}")
            for sample in pbar:
                images = sample["images"].to(device)
                ref_points_full = sample["ref_points_full"].to(device)
                hw_random = sample["hw_random"].to(device)
                wld_paras = sample["wld_map_paras"].to(device)
                gt_world_xy = sample["gt_world_xy"]
                cam_locations = T.extract_camera_locations(sample["camera_paras"].to(device)).squeeze(0)
                patch_num = hw_random.shape[-1]

                with torch.amp.autocast("cuda"):
                    preds = model(images, ref_points_full, hw_random)
                preds = preds.float()

                optimizer.zero_grad()
                total = 0.0
                parts = {"ot": 0.0, "bg": 0.0, "count": 0.0}
                for p in range(patch_num):
                    hw_offset = hw_random[:, :, p]
                    with torch.amp.autocast("cuda", enabled=False):
                        target_dots, cam_locs, map_gt_conv, gt_mask = patch_targets(
                            gt_world_xy, wld_paras, hw_offset, cam_locations, map_kernel, single_person_peak, device
                        )
                        region = (1.0 - gt_mask) if args.region == "sparse" else torch.ones_like(gt_mask)
                        pred = preds[:, p] * region
                        target_density = map_gt_conv * region
                        dots = (
                            T.filter_points_outside_mask(target_dots, gt_mask)
                            if args.region == "sparse"
                            else target_dots
                        )

                        ot_loss = ot_criterion(pred, dots.to(device), cam_locs)
                        empty = region * (target_density < 1e-6).float()
                        bg_loss = F.mse_loss(pred * empty, torch.zeros_like(pred))
                        count_loss = torch.abs(pred.sum() - target_density.sum()) / T.GAUSSIAN_SCALE

                        patch_loss = (
                            args.genloss_weight * ot_loss + args.bg_weight * bg_loss + args.count_weight * count_loss
                        )
                    total = total + patch_loss
                    parts["ot"] += float(ot_loss)
                    parts["bg"] += float(bg_loss)
                    parts["count"] += float(count_loss)

                loss = total / patch_num
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=T.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()

                sums["loss"] += loss.item()
                for key in parts:
                    sums[key] += parts[key] / patch_num
                n_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.3f}"})
            torch.cuda.empty_cache()
        scheduler.step()
        train_s = time.perf_counter() - t_epoch

        # ---------------- optional precise BN (neck BN layers are training here)
        if args.precise_bn_frames > 0:
            bn_mods = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training]
            if bn_mods:
                old = [m.momentum for m in bn_mods]
                for m in bn_mods:
                    m.reset_running_stats()
                    m.momentum = None
                with torch.no_grad():
                    for _, loader in train_loaders:
                        it = iter(loader)
                        for _ in range(args.precise_bn_frames):
                            try:
                                s = next(it)
                            except StopIteration:
                                break
                            with torch.amp.autocast("cuda"):
                                model(
                                    s["images"].to(device), s["ref_points_full"].to(device), s["hw_random"].to(device)
                                )
                for m, mom in zip(bn_mods, old):
                    m.momentum = mom

        # ---------------- validation
        model.eval()
        abs_err, f1 = [], {"full": [0, 0, 0], "sparse": [0, 0, 0]}  # tp, pred, gt
        s_p = s_t = s_tp = s_pp = 0.0
        dead = region_px = 0
        with torch.no_grad():
            for sample in tqdm(cached_val, desc=f"SparseOnly | Epoch {epoch + 1}/{args.epochs} [Val]"):
                images = sample["images"].to(device)
                ref_points_full = sample["ref_points_full"].to(device)
                hw_random = sample["hw_random"].to(device)
                wld_paras = sample["wld_map_paras"].to(device)
                gt_world_xy = sample["gt_world_xy"]
                cam_locations = T.extract_camera_locations(sample["camera_paras"].to(device)).squeeze(0)
                with torch.amp.autocast("cuda"):
                    preds = model(images, ref_points_full, hw_random)
                preds = preds.float()

                for p in range(hw_random.shape[-1]):
                    hw_offset = hw_random[:, :, p]
                    target_dots, _, map_gt_conv, gt_mask = patch_targets(
                        gt_world_xy, wld_paras, hw_offset, cam_locations, map_kernel, single_person_peak, device
                    )
                    pred = preds[:, p]
                    abs_err.append(abs(pred.sum().item() / mass_per_person - target_dots.shape[2]))

                    s_p += pred.sum().item()
                    s_t += map_gt_conv.sum().item()
                    s_tp += (pred * map_gt_conv).sum().item()
                    s_pp += (pred * pred).sum().item()
                    dead += int((pred == 0).sum().item())
                    region_px += pred.numel()

                    for name, region in (("full", torch.ones_like(gt_mask)), ("sparse", 1.0 - gt_mask)):
                        peaks = T.extract_peaks_via_nms(
                            pred * region, threshold=nms_threshold, kernel_size=args.nms_kernel
                        )
                        pr_xy = (
                            peaks.cpu().numpy() * np.array([T.BEV_SHAPE[1], T.BEV_SHAPE[0]])
                            if peaks.shape[0]
                            else np.zeros((0, 2))
                        )
                        gt_xy = points_in_region(target_dots, region)
                        f1[name][0] += T.count_matches_np(pr_xy, gt_xy, args.match_px)
                        f1[name][1] += len(pr_xy)
                        f1[name][2] += len(gt_xy)

        def prf(tp, npred, ngt):
            prec, rec = tp / max(npred, 1), tp / max(ngt, 1)
            return 2 * prec * rec / max(prec + rec, 1e-12), prec, rec, npred / max(ngt, 1)

        full_f1, full_p, full_r, full_ratio = prf(*f1["full"])
        sp_f1, sp_p, sp_r, sp_ratio = prf(*f1["sparse"])
        count_mae = float(np.mean(abs_err))
        n = max(n_batches, 1)
        print(
            f">>> SparseOnly Epoch {epoch + 1} | loss {sums['loss'] / n:.3f} (ot {sums['ot'] / n:.3f} | "
            f"bg {sums['bg'] / n:.4f} | count {sums['count'] / n:.3f}) | train {train_s / 60:.1f} min <<<"
        )
        print(
            f"    [val] count MAE {count_mae:.1f} persons/patch | "
            f"full F1 {full_f1:.3f} (P {full_p:.2f}, R {full_r:.2f}, pred/GT {full_ratio:.2f}) | "
            f"sparse-region F1 {sp_f1:.3f} (P {sp_p:.2f}, R {sp_r:.2f}, pred/GT {sp_ratio:.2f}) | "
            f"NMS mult {args.nms_mult:g}, kernel {args.nms_kernel}, radius {args.match_px:g} px"
        )
        print(
            f"    [val-calib] sum(P)/sum(T) {s_p / max(s_t, 1e-12):.2f} | amp slope {s_tp / max(s_pp, 1e-12):.2f} | "
            f"dead px {100 * dead / max(region_px, 1):.0f}%"
        )

        metric = {"sparse_f1": 1.0 - sp_f1, "full_f1": 1.0 - full_f1, "count_mae": count_mae}[args.select]
        state = {"model_state_dict": model.state_dict(), "epoch": epoch + 1, "args": vars(args)}
        if metric < best_metric:
            best_metric, epochs_without_improvement = metric, 0
            torch.save(state, os.path.join(T.src_dir, f"sparseonly_best{tag}.pth"))
            print(f"*** New best ({args.select}: {metric:.4f}). Saved checkpoint. ***")
        else:
            epochs_without_improvement += 1
            print(f"--- No improvement for {epochs_without_improvement}/{args.patience} epochs ---")
        if (epoch + 1) % 5 == 0:
            torch.save(state, os.path.join(T.src_dir, f"sparseonly_checkpoint{tag}_epoch_{epoch + 1}.pth"))
        if epochs_without_improvement >= args.patience:
            print("!!! Early stopping !!!")
            break

    final_path = os.path.join(T.src_dir, f"sparseonly_final{tag}.pth")
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, final_path)
    print(f"[checkpoint] Final weights saved to {final_path}")


if __name__ == "__main__":
    main()
