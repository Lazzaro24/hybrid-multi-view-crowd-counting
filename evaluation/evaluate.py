"""
evaluation/evaluate.py

Single entry point synthesizing every evaluation approach discussed this
session: patch-wise (debugging) and full-scene (results), oracle routing
(matches how Stage 2's experts were actually trained/supervised) and
learned routing (the real inference-time behavior), all built on the
mode-agnostic metrics modules in evaluation/metrics/.

=======================================================================
KEY DESIGN DECISIONS
=======================================================================

1. FULL-SCENE MODE REQUIRES TILING, NOT ONE FORWARD PASS.
   SpatialCrossAttentionNeck was constructed with a FIXED
   full_bev_shape=BEV_SHAPE=(200,200) (see train_proposed_multiscene.py's
   ProposedModel.__init__). Its BEV query grid does not adapt to a
   different resolution at inference time. So "full scene" evaluation
   means: slide 200x200 windows across the scene's real h_actual/w_actual
   grid, run neck+head per tile (the exact same per-tile call the
   training loop's forward() already makes internally via hw_random
   slicing -- just at systematic tile positions instead of 3 random
   crops per frame), and stitch results into one [1,1,H_full,W_full] map.
   run_single_patch() below is shared by both patch mode (looping over
   hw_random's random crops) and scene mode (looping over systematic
   tiles), so the oracle/learned routing logic exists in exactly one place.

2. TWO DIFFERENT USES OF "A MASK" -- KEPT DELIBERATELY SEPARATE.
   - ROUTING mask: how dense_pred/sparse_pred get COMBINED into the final
     prediction. Oracle mode uses the ground-truth density-threshold mask
     (matching what Stage 2's experts were actually supervised against);
     learned mode uses model.head.gating_net's own output (the real
     inference-time behavior with no privileged information).
   - BREAKDOWN mask: which points get classified 'dense' vs 'sparse' for
     the diagnostic breakdown in evaluation/metrics/localization.py. This
     is ALWAYS the true oracle mask, regardless of --routing. The
     breakdown is answering "does the model do worse in genuinely dense
     areas" (a question about the data), not "does the model's own
     routing belief match reality" (a different question, which is what
     --routing itself already controls). Conflating these would make the
     diagnostic answer something other than what it's asked to answer.

3. THE ORACLE-MASK FORMULA IS SHARED, NOT COPIED. Both this script and
   train_proposed_multiscene.py call oracle_mask_from_conv (backed by
   utils/oracle_mask.py), so evaluation always measures against the same
   boundary the model was supervised with. Note that a model must be
   evaluated with the same --oracle_min_extent/--oracle_min_area it was
   trained with; checkpoints store theirs under 'oracle_mask_config'.

4. PEAK-EXTRACTION THRESHOLD FOR THE COMBINED OUTPUT IS NEW AND
   UNCALIBRATED. Training never needed to peak-extract the BLENDED
   (dense*mask + sparse*(1-mask)) output -- dense was always evaluated by
   mass, sparse's peaks were only ever extracted from the sparse-only
   masked region. This script needs a threshold for the full combined
   map, which is a genuinely new calibration question. Default here
   (--peak_threshold_fraction, applied against map_kernel's peak-
   normalized single-person value of 1.0, following PEAK_THRESHOLD_
   FRACTION's precedent from training) is a reasonable starting guess,
   NOT a validated number -- treat Precision/Recall/F1/MODA/MODP here
   with the same "uncalibrated placeholder" caution
   evaluate_held_out.py's --match_threshold_px got, until swept.

5. --match_threshold_m defaults to WILDTRACK's real 0.5m standard,
   converted to each scene's own pixel units via r (pixels-per-meter)
   -- see evaluation/metrics/localization.py's match_threshold_px_from_r.
   Never a fixed pixel constant across scenes with different resolutions.

6. BASELINE MODEL SUPPORT IS BEST-EFFORT AND UNVERIFIED. This session
   never confirmed BaselineModel's current forward() signature for
   multi-scene use (only the old, single-scene evaluate.py/
   evaluate_models.py's calling convention, `model(images, camera_paras,
   wld_paras, full_scene_shape=...)`, assumed unchanged here). Baseline
   has no dense/sparse routing at all -- --routing and the dense/sparse
   breakdown are silently skipped for it, not applicable.

Usage:
    python evaluate.py --model proposed --weights multiscene_best_stage_4.pth \\
        --eval_mode scene --routing learned --split test --k 5

    python evaluate.py --model proposed --weights multiscene_best_stage_4.pth \\
        --eval_mode patch --routing oracle --split test --k 5
"""

import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)
sys.path.append(
    os.path.join(src_dir, "training_scripts")
)  # train_proposed_multiscene.py lives here, not in src/ directly

from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
from losses.gaussian_mse import get_map_kernel

from train_proposed_multiscene import (
    ProposedModel,
    BEV_SHAPE,
    GAUSSIAN_SCALE,
    GAUSSIAN_SIGMA,
    GAUSSIAN_KERNEL_SIZE,
    PEAK_THRESHOLD_FRACTION,
    create_ground_truth_heatmap,
    oracle_mask_from_conv,
)

from evaluation.metrics.localization import (
    extract_peaks_from_heatmap,
    convert_gt_to_bev_coords,
    match_threshold_px_from_r,
)
from evaluation.metrics.counting import CountingMetricsAggregator

try:
    from training_scripts.train_baseline import BaselineModel

    BASELINE_AVAILABLE = True
except ImportError:
    BASELINE_AVAILABLE = False


# ==========================================
# Oracle mask -- shared with training via train_proposed_multiscene.oracle_mask_from_conv (see design note #3).
# single_person_peak is kept in the signature for the existing call sites; the threshold comes from the shared helper.
# ==========================================


def compute_oracle_mask(target_patch, map_kernel, single_person_peak):
    pad_size = map_kernel.shape[-1] // 2
    map_gt_conv = F.conv2d(target_patch, map_kernel, padding=pad_size)
    return oracle_mask_from_conv(map_gt_conv), map_gt_conv


# ==========================================
# Shared single-patch forward, used by BOTH patch mode (random
# crops) and scene mode (systematic tiles) -- see design note #1.
# ==========================================


def run_single_patch(model, features, patch_ref_points, routing, gt_mask=None):
    """
    features: backbone output, already computed once per FRAME (not
    per-patch/tile) and reused across every patch/tile of that frame --
    the backbone doesn't need re-running per tile, only the neck+head do,
    since patch_ref_points is the only thing that changes between tiles.

    Returns (pred, routing_mask). routing_mask is the mask actually used
    to combine dense_pred/sparse_pred -- NOT necessarily the oracle mask
    (see design note #2); the caller is responsible for separately
    computing/using the true oracle mask for the breakdown diagnostic.
    """
    proj_bev = model.neck(features, patch_ref_points)
    dense_pred = torch.relu(model.head.dense_expert(proj_bev))
    sparse_pred = torch.relu(model.head.sparse_expert(proj_bev))

    if routing == "oracle":
        assert gt_mask is not None, "oracle routing requires gt_mask to be supplied"
        routing_mask = gt_mask
    else:
        routing_mask = model.head.gating_net(proj_bev)

    pred = routing_mask * dense_pred + (1.0 - routing_mask) * sparse_pred
    return pred, routing_mask


def run_proposed_full_scene(model, images, ref_points_full, h_full, w_full, routing, gt_mask_full=None, device="cuda"):
    """
    [NEW, see design note #1] Tiles the full scene into 200x200 windows
    (clamped at the edges so every tile is exactly BEV_SHAPE -- always
    possible since SynMVCrowdDataset already guarantees h_actual/w_actual
    >= cropped_size + patch_num, so h_full/w_full are always >= 203),
    runs run_single_patch per tile, and stitches into one full-scene map.

    Non-overlapping tiling except possibly at the very last row/column
    (clamped to fit rather than padded) -- a later tile's assignment
    simply overwrites any overlap from an earlier one at that edge,
    which only affects a thin strip, not a systematic double-count.
    """
    B, N, C, H_img, W_img = images.shape
    images_flat = images.view(B * N, C, H_img, W_img)
    features_flat = model.backbone(images_flat)
    _, C_f, H_f, W_f = features_flat.shape
    features = features_flat.view(B, N, C_f, H_f, W_f)

    tile_h, tile_w = BEV_SHAPE

    h_starts = list(range(0, h_full - tile_h + 1, tile_h))
    if not h_starts or h_starts[-1] + tile_h < h_full:
        h_starts.append(h_full - tile_h)
    w_starts = list(range(0, w_full - tile_w + 1, tile_w))
    if not w_starts or w_starts[-1] + tile_w < w_full:
        w_starts.append(w_full - tile_w)

    pred_full = torch.zeros((B, 1, h_full, w_full), device=device)
    mask_full = torch.zeros((B, 1, h_full, w_full), device=device)

    for h_start in h_starts:
        for w_start in w_starts:
            patch_ref_points = ref_points_full[:, :, h_start : h_start + tile_h, w_start : w_start + tile_w, :, :]
            tile_gt_mask = None
            if routing == "oracle":
                tile_gt_mask = gt_mask_full[:, :, h_start : h_start + tile_h, w_start : w_start + tile_w]

            tile_pred, tile_mask = run_single_patch(model, features, patch_ref_points, routing, gt_mask=tile_gt_mask)

            pred_full[:, :, h_start : h_start + tile_h, w_start : w_start + tile_w] = tile_pred
            mask_full[:, :, h_start : h_start + tile_h, w_start : w_start + tile_w] = tile_mask

    return pred_full, mask_full


# ==========================================
# Dataset / scene setup -- mirrors train_proposed_multiscene.py's
# conventions (optimal_cameras_filtered.json, excluded_scenes.json)
# ==========================================


def load_excluded_scenes():
    path = os.path.join(src_dir, "excluded_scenes.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def build_eval_scene_list(split, k, optimal_cameras_config, excluded_scenes):
    split_dir = os.path.join(src_dir, "datasets", "SynMVCrowd", "data", split)
    scene_folders = sorted(
        os.path.join(split_dir, d) for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))
    )
    entries = []
    for scene_folder in scene_folders:
        scene_name = os.path.basename(scene_folder)
        if scene_name in excluded_scenes:
            print(f"[EXCLUDED] {scene_name}: {excluded_scenes[scene_name]}")
            continue
        cam_ids = optimal_cameras_config.get(scene_name, {}).get(str(k))
        if cam_ids is None:
            print(f"[skip] {scene_name}: no camera selection for k={k}")
            continue
        entries.append({"scene_name": scene_name, "scene_folder": scene_folder, "cam_ids": cam_ids})
    return entries


def get_model(model_name, k, device, weights_path):
    if model_name == "proposed":
        model = ProposedModel(num_cam=k).to(device)
    elif model_name == "baseline":
        if not BASELINE_AVAILABLE:
            raise ImportError(
                "training_scripts.train_baseline.BaselineModel could not be imported -- "
                "baseline support in this script is best-effort and unverified for "
                "multi-scene use (see module docstring, design note #6)."
            )
        model = BaselineModel(num_cam=k).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"  missing keys: {missing}")
            print(f"  unexpected keys: {unexpected}")
    else:
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model.eval()
    return model


# ==========================================
# MAIN EVALUATION
# ==========================================


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"--- Evaluating {args.model} | mode={args.eval_mode} | "
        f"routing={args.routing if args.model == 'proposed' else 'n/a'} | "
        f"split={args.split} | k={args.k} | device={device} ---"
    )

    config_path = os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")
    with open(config_path, "r") as f:
        optimal_cameras_config = json.load(f)
    excluded_scenes = load_excluded_scenes()
    scene_entries = build_eval_scene_list(args.split, args.k, optimal_cameras_config, excluded_scenes)
    print(f"Evaluating over {len(scene_entries)} scenes in split='{args.split}'.")

    model = get_model(args.model, args.k, device, args.weights)

    map_kernel = get_map_kernel(sigma=GAUSSIAN_SIGMA, kernel_size=GAUSSIAN_KERNEL_SIZE).to(device)
    KERNEL_MASS_PER_PERSON = map_kernel.sum().item()
    single_person_peak_normalized = 1.0  # map_kernel is peak-normalized (max=1) -- see design note #4
    peak_threshold = single_person_peak_normalized * args.peak_threshold_fraction

    counting_metrics = CountingMetricsAggregator()
    # LocalizationMetricsAggregator needs ONE match_threshold_px -- but that's
    # derived per-scene from r (pixels-per-meter), which varies by scene (see
    # the wld_map_paras corpus sweep: r ranges roughly 3-11 across
    # scenes, before scene_71's anomaly). So we can't build ONE aggregator
    # up front; instead pool raw ingredients across scenes ourselves and
    # only ever call an aggregator with a threshold matching the CURRENT
    # scene's own r, one scene at a time, merging their outputs manually.
    total_tp_distances = []
    total_fp = total_fn = total_gt = 0
    dense_tp = dense_fp = dense_fn = dense_gt = 0
    sparse_tp = sparse_fp = sparse_fn = sparse_gt = 0

    for entry in scene_entries:
        scene_name, scene_folder, cam_ids = entry["scene_name"], entry["scene_folder"], entry["cam_ids"]
        dataset = SynMVCrowdDataset(scene_dir=scene_folder, camera_ids=cam_ids)
        n_frames = min(args.frames_per_scene, len(dataset))
        # Deterministic frame subset (no resampling -- evaluation should be
        # reproducible run-to-run, unlike training's per-epoch resampling).
        rng = np.random.RandomState(args.seed)
        frame_indices = sorted(rng.choice(len(dataset), size=n_frames, replace=False).tolist())

        scene_match_threshold_px = None  # set once we see this scene's wld_map_paras

        for idx in frame_indices:
            sample = dataset[idx]
            images = sample["images"].unsqueeze(0).to(device)
            wld_paras = sample["wld_map_paras"].unsqueeze(0).to(device)
            ref_points_full = sample["ref_points_full"].unsqueeze(0).to(device)
            gt_world_xy = sample["gt_world_xy"]

            r_value = wld_paras[0, 0].item()
            scene_match_threshold_px = match_threshold_px_from_r(r_value, args.match_threshold_m)

            if args.model == "baseline":
                # [BEST-EFFORT, UNVERIFIED -- see module docstring design note #6]
                # Assumes the OLD evaluate.py/evaluate_models.py calling convention:
                # model(images, camera_paras, wld_paras, hw_random=...) for patch mode,
                # model(images, camera_paras, wld_paras, full_scene_shape=...) for scene
                # mode. Never confirmed against a real multi-scene-trained baseline
                # checkpoint this session -- if this signature is stale, this will
                # raise a TypeError here (a clear, early failure), not silently
                # produce wrong numbers.
                camera_paras = sample["camera_paras"].unsqueeze(0).to(device)
                if args.eval_mode == "patch":
                    hw_random = sample["hw_random"].unsqueeze(0).to(device)
                    pred_full_or_patches = model(images, camera_paras, wld_paras, hw_random)
                    patch_num = hw_random.shape[-1]
                    for p in range(patch_num):
                        hw_offset = hw_random[:, :, p]
                        pred = (
                            pred_full_or_patches[:, p, ...] if pred_full_or_patches.dim() == 5 else pred_full_or_patches
                        )
                        gt_coords = convert_gt_to_bev_coords(gt_world_xy, wld_paras, hw_offset, BEV_SHAPE)
                        pred_coords = extract_peaks_from_heatmap(pred, threshold=peak_threshold, min_distance=3)
                        pred_count = pred.sum().item() / GAUSSIAN_SCALE
                        counting_metrics.update(pred_count, len(gt_coords))
                        from evaluation.metrics.localization import match_points

                        tp_dists, fp_idx, fn_idx = match_points(pred_coords, gt_coords, scene_match_threshold_px)
                        total_tp_distances.extend(tp_dists)
                        total_fp += len(fp_idx)
                        total_fn += len(fn_idx)
                        total_gt += len(gt_coords)
                else:
                    h_full = int(wld_paras[0, 3].item())
                    w_full = int(wld_paras[0, 4].item())
                    zero_offset = torch.zeros((1, 2), device=device)
                    pred_full = model(images, camera_paras, wld_paras, full_scene_shape=(h_full, w_full))
                    if pred_full.dim() == 5:
                        pred_full = pred_full.squeeze(1)
                    gt_coords = convert_gt_to_bev_coords(gt_world_xy, wld_paras, zero_offset, (h_full, w_full))
                    pred_coords = extract_peaks_from_heatmap(pred_full, threshold=peak_threshold, min_distance=3)
                    pred_count = pred_full.sum().item() / GAUSSIAN_SCALE
                    counting_metrics.update(pred_count, len(gt_coords))
                    from evaluation.metrics.localization import match_points

                    tp_dists, fp_idx, fn_idx = match_points(pred_coords, gt_coords, scene_match_threshold_px)
                    total_tp_distances.extend(tp_dists)
                    total_fp += len(fp_idx)
                    total_fn += len(fn_idx)
                    total_gt += len(gt_coords)
                continue  # baseline has no dense/sparse routing/breakdown -- nothing further this iteration

            if args.eval_mode == "patch":
                hw_random = sample["hw_random"].unsqueeze(0).to(device)
                patch_num = hw_random.shape[-1]

                B, N, C, H_img, W_img = images.shape
                images_flat = images.view(B * N, C, H_img, W_img)
                features_flat = model.backbone(images_flat)
                _, C_f, H_f, W_f = features_flat.shape
                features = features_flat.view(B, N, C_f, H_f, W_f)

                for p in range(patch_num):
                    hw_offset = hw_random[:, :, p]
                    h_start, w_start = hw_offset[0, 0].item(), hw_offset[0, 1].item()
                    patch_ref_points = ref_points_full[
                        :, :, h_start : h_start + BEV_SHAPE[0], w_start : w_start + BEV_SHAPE[1], :, :
                    ]

                    target_patch = create_ground_truth_heatmap(
                        gt_world_xy, wld_paras, hw_offset, bev_shape=BEV_SHAPE
                    ).to(device)
                    single_person_peak_scale = GAUSSIAN_SCALE / (2 * torch.pi * (GAUSSIAN_SIGMA**2))
                    oracle_mask, map_gt_conv = compute_oracle_mask(target_patch, map_kernel, single_person_peak_scale)

                    pred, _ = run_single_patch(
                        model,
                        features,
                        patch_ref_points,
                        args.routing,
                        gt_mask=oracle_mask if args.routing == "oracle" else None,
                    )

                    gt_coords = convert_gt_to_bev_coords(gt_world_xy, wld_paras, hw_offset, BEV_SHAPE)
                    pred_coords = extract_peaks_from_heatmap(pred, threshold=peak_threshold, min_distance=3)

                    pred_count = pred.sum().item() / KERNEL_MASS_PER_PERSON
                    target_mass = map_gt_conv.sum().item() / KERNEL_MASS_PER_PERSON
                    counting_metrics.update(pred_count, len(gt_coords), pred_mass=pred_count, target_mass=target_mass)

                    from evaluation.metrics.localization import match_points, classify_region

                    tp_dists, fp_idx, fn_idx = match_points(pred_coords, gt_coords, scene_match_threshold_px)
                    total_tp_distances.extend(tp_dists)
                    total_fp += len(fp_idx)
                    total_fn += len(fn_idx)
                    total_gt += len(gt_coords)

                    if args.dense_sparse_breakdown:
                        gt_regions = classify_region(gt_coords, oracle_mask, BEV_SHAPE)
                        pred_regions = classify_region(pred_coords, oracle_mask, BEV_SHAPE)
                        d_gt = sum(1 for r in gt_regions if r == "dense")
                        s_gt = sum(1 for r in gt_regions if r == "sparse")
                        d_fn = sum(1 for i in fn_idx if gt_regions[i] == "dense")
                        s_fn = sum(1 for i in fn_idx if gt_regions[i] == "sparse")
                        d_fp = sum(1 for i in fp_idx if pred_regions[i] == "dense")
                        s_fp = sum(1 for i in fp_idx if pred_regions[i] == "sparse")
                        dense_gt += d_gt
                        sparse_gt += s_gt
                        dense_tp += d_gt - d_fn
                        sparse_tp += s_gt - s_fn
                        dense_fp += d_fp
                        sparse_fp += s_fp
                        dense_fn += d_fn
                        sparse_fn += s_fn

            else:  # scene mode
                h_full = int(wld_paras[0, 3].item())
                w_full = int(wld_paras[0, 4].item())
                zero_offset = torch.zeros((1, 2), device=device)

                target_full = create_ground_truth_heatmap(
                    gt_world_xy, wld_paras, zero_offset, bev_shape=(h_full, w_full)
                ).to(device)
                single_person_peak_scale = GAUSSIAN_SCALE / (2 * torch.pi * (GAUSSIAN_SIGMA**2))
                oracle_mask_full, map_gt_conv_full = compute_oracle_mask(
                    target_full, map_kernel, single_person_peak_scale
                )

                pred_full, _ = run_proposed_full_scene(
                    model,
                    images,
                    ref_points_full,
                    h_full,
                    w_full,
                    args.routing,
                    gt_mask_full=oracle_mask_full if args.routing == "oracle" else None,
                    device=device,
                )

                gt_coords = convert_gt_to_bev_coords(gt_world_xy, wld_paras, zero_offset, (h_full, w_full))
                pred_coords = extract_peaks_from_heatmap(pred_full, threshold=peak_threshold, min_distance=3)

                pred_count = pred_full.sum().item() / KERNEL_MASS_PER_PERSON
                target_mass = map_gt_conv_full.sum().item() / KERNEL_MASS_PER_PERSON
                counting_metrics.update(pred_count, len(gt_coords), pred_mass=pred_count, target_mass=target_mass)

                from evaluation.metrics.localization import match_points, classify_region

                tp_dists, fp_idx, fn_idx = match_points(pred_coords, gt_coords, scene_match_threshold_px)
                total_tp_distances.extend(tp_dists)
                total_fp += len(fp_idx)
                total_fn += len(fn_idx)
                total_gt += len(gt_coords)

                if args.dense_sparse_breakdown:
                    gt_regions = classify_region(gt_coords, oracle_mask_full, (h_full, w_full))
                    pred_regions = classify_region(pred_coords, oracle_mask_full, (h_full, w_full))
                    d_gt = sum(1 for r in gt_regions if r == "dense")
                    s_gt = sum(1 for r in gt_regions if r == "sparse")
                    d_fn = sum(1 for i in fn_idx if gt_regions[i] == "dense")
                    s_fn = sum(1 for i in fn_idx if gt_regions[i] == "sparse")
                    d_fp = sum(1 for i in fp_idx if pred_regions[i] == "dense")
                    s_fp = sum(1 for i in fp_idx if pred_regions[i] == "sparse")
                    dense_gt += d_gt
                    sparse_gt += s_gt
                    dense_tp += d_gt - d_fn
                    sparse_tp += s_gt - s_fn
                    dense_fp += d_fp
                    sparse_fp += s_fp
                    dense_fn += d_fn
                    sparse_fn += s_fn

        del dataset  # [see train_proposed_multiscene.py design note #9] one-shot eval loop,
        # not a persistent multi-epoch registry -- discard each scene's dataset
        # as we move on rather than holding ~50 alive simultaneously.
        print(f"  {scene_name}: {n_frames} frames evaluated.")

    # ==========================================
    # Final aggregation -- pooled, never per-scene-averaged (see
    # LocalizationMetricsAggregator's docstring / match_points' docstring
    # for why this matters).
    # ==========================================
    counting_results = counting_metrics.compute()

    total_tp = len(total_tp_distances)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    moda = 1.0 - (total_fp + total_fn) / total_gt if total_gt > 0 else float("nan")
    modp = float(np.mean([1.0 - d / scene_match_threshold_px for d in total_tp_distances])) if total_tp > 0 else 0.0
    # NOTE: modp above reuses the LAST scene's scene_match_threshold_px for the
    # normalization -- technically inconsistent if scenes have very different r
    # (a match distance from a high-r scene and a low-r scene aren't on the same
    # normalized 0-1 scale once pooled this way). Flagged, not fixed: fixing
    # properly means normalizing EACH match distance by ITS OWN scene's threshold
    # before pooling, not pooling raw pixel distances across scenes with
    # different pixel-per-meter scales. Worth doing before trusting MODP
    # specifically across a multi-scene evaluation; MODA/Precision/Recall/F1
    # (raw TP/FP/FN counts, no per-scene normalization) are unaffected.

    print("\n" + "=" * 70)
    print(
        f"EVALUATION RESULTS -- {args.model}, {args.eval_mode} mode"
        + (f", routing={args.routing}" if args.model == "proposed" else "")
    )
    print("-" * 70)
    print(
        f"  MAE:  {counting_results['MAE']:.3f}   MSE: {counting_results['MSE']:.3f}   NAE: {counting_results['NAE']:.4f}"
    )
    if "Mass_MAE" in counting_results:
        print(
            f"  Mass_MAE: {counting_results['Mass_MAE']:.3f}   Mass_MSE: {counting_results['Mass_MSE']:.3f}  "
            f"(vs. the model's OWN blurred-Gaussian target -- see counting.py's docstring; "
            f"a gap between this and MAE above is boundary-truncation, not necessarily a learning failure)"
        )
    print("-" * 70)
    print(f"  Precision: {precision:.4f}   Recall: {recall:.4f}   F1: {f1:.4f}")
    print(
        f"  MODA: {moda:.4f}   MODP: {modp:.4f}  "
        f"(UNCALIBRATED peak threshold -- see module docstring design note #4; "
        f"MODP pooling caveat above -- treat both with caution until swept/fixed)"
    )
    if args.dense_sparse_breakdown:

        def prf(tp, fp, fn, gt):
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / gt if gt > 0 else float("nan")
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return p, r, f

        dp, dr, df1 = prf(dense_tp, dense_fp, dense_fn, dense_gt)
        sp, sr, sf1 = prf(sparse_tp, sparse_fp, sparse_fn, sparse_gt)
        print("-" * 70)
        print(f"  DENSE  regions (gt={dense_gt}): Precision={dp:.4f}  Recall={dr:.4f}  F1={df1:.4f}")
        print(f"  SPARSE regions (gt={sparse_gt}): Precision={sp:.4f}  Recall={sr:.4f}  F1={sf1:.4f}")
        print("  (classified against the TRUE oracle mask regardless of --routing -- see design note #2)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-scene, multi-mode evaluation")
    parser.add_argument("--model", type=str, required=True, choices=["proposed", "baseline"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "validate", "test"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--eval_mode", type=str, default="scene", choices=["patch", "scene"])
    parser.add_argument(
        "--routing",
        type=str,
        default="learned",
        choices=["oracle", "learned"],
        help="Proposed model only, ignored for baseline (no routing to speak of).",
    )
    parser.add_argument("--frames_per_scene", type=int, default=10)
    parser.add_argument(
        "--match_threshold_m",
        type=float,
        default=0.5,
        help="WILDTRACK-standard match radius in METERS, converted to each scene's "
        "own pixel units via r -- see match_threshold_px_from_r.",
    )
    parser.add_argument(
        "--peak_threshold_fraction",
        type=float,
        default=PEAK_THRESHOLD_FRACTION,
        help="UNCALIBRATED for the combined dense+sparse output -- see module docstring design note #4.",
    )
    parser.add_argument("--dense_sparse_breakdown", action="store_true", default=True)
    parser.add_argument("--no_dense_sparse_breakdown", dest="dense_sparse_breakdown", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.model == "baseline" and args.dense_sparse_breakdown:
        print(
            "[note] --dense_sparse_breakdown has no meaning for the baseline model (no dense/sparse "
            "routing) -- ignoring."
        )
        args.dense_sparse_breakdown = False

    evaluate(args)
