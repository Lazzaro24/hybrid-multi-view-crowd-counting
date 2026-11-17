"""
evaluation/evaluate_paper_protocol.py

Evaluation following the SynMVCrowd paper (IJCV 2026) and its released test code, so the baseline
and the proposed model are scored identically:

- Whole scene map (no patches), every frame of every scene in the split, and per frame
  `--view_samplings` (5) random draws of 5 camera views. Frame order and view draws are seeded.
- Ground truth: people visible in at least one of the drawn views (SynMVCrowdDataset), rasterised
  onto the 0.2 m ground-plane grid like the training targets (truncation, one mark per pixel).
- Counting: pred.sum() / 100 vs. the GT count; MAE, MSE (root of the mean squared error, the
  crowd-counting convention), NAE capped at 1 per sample (as in the original trainer).
- Localization (original trainer.test + MVDet evaluation): min-max normalise the predicted map per
  sample, keep pixels above --cls_thres (0.2), greedy NMS with a 2.5 px radius, then Hungarian
  matching against the GT with a 0.5 m radius (2.5 px at r = 5 px/m).
  MODA = 1 - (FP + FN) / GT, MODP = sum(1 - d / t) / TP, Precision, Recall, F1 -- all pooled over
  every sample, reported in percent like the paper's Table 3.

Usage (from src/):
    python evaluation/evaluate_paper_protocol.py --model baseline --weights runs/baseline/run_1/best.pth
    python evaluation/evaluate_paper_protocol.py --model proposed --weights multiscene_best_stage_3_<tag>.pth
    # a neck-ablation run (the baseline decoder on an ablation fusion, "H0"; its EMA weights, precise BN on one
    # K-means frame per train scene, exactly as eval_map_quality.py scores it)
    python evaluation/evaluate_paper_protocol.py --model ablation --weights runs/neck_ablation/ablation_v2/homography_256/last.pth
    # the density rule: dense expert where its own predicted density >= 0.25 p/m^2, sparse peaks elsewhere
    python evaluation/evaluate_paper_protocol.py --model proposed --routing dzone0.25p --weights <checkpoint>.pth
    # camera ablation: draws of 3 views instead of 5 (models trained on 5; per-camera-slot attention layers reuse
    # the trained slots cyclically, utils/camera_slots.py); the results file gets a _3v suffix
    python evaluation/evaluate_paper_protocol.py --model baseline --weights runs/baseline/ot/best.pth --num_views 3
    # per-sample rows (<out>.samples.csv) and a detection threshold sweep (doc/final_evaluations.md E1/E4)
    python evaluation/evaluate_paper_protocol.py --model baseline --weights runs/baseline/ot/best.pth --dump_samples         --cls_thres_list 0.1,0.2,0.3,0.4,0.5,0.6,0.8
"""

import os
import sys
import csv
import json
import math
import subprocess
import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
from utils.camera_slots import set_camera_count, renormalise_views
from utils.exclusions import load_excluded_scenes

SCALE = 100  # density maps are trained to sum to count * 100 (GaussianMSE_norm)


# ==========================================
# Models: each returns a full-scene density map [H_full, W_full]
# ==========================================


def load_model(name, weights, device):
    if name == "ablation":
        return load_ablation(weights, device)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if name == "baseline":
        from training_scripts.train_baseline import BaselineModel

        model = BaselineModel(num_cam=5)
    else:
        import training_scripts.train_proposed_multiscene as T

        # Build the checkpoint's architecture: backbone layout and whether the router has the contrast branch
        T.ROUTER_CONTRAST = (
            state["head.gating_net.fuse.0.weight"].shape[1] == 3 * state["head.gating_net.fine.0.weight"].shape[0]
        )
        model = T.build_model_for_checkpoint(checkpoint, num_cam=5)
        model.bev_shape = T.BEV_SHAPE
        # The experts are trained on dots blurred with a peak-normalised kernel, so one person sums to
        # kernel.sum() (~31.4), not SCALE -- rescale so pred.sum() / SCALE is a count, as for the baseline
        model.to_scale = SCALE / float(
            T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).sum()
        )
        model.T = T
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    epoch = checkpoint.get("epoch", "?") if isinstance(checkpoint, dict) else "?"
    return model, epoch


def load_ablation(weights, device, num_workers=4):
    """A neck-ablation run (training_scripts/neck_ablation.py; weights = <run>/<variant>/last.pth): its EMA weights,
    then precise BN of the fusion on one K-means-camera frame per train scene -- the same model eval_map_quality.py
    scores as H0. Its maps sum to count * SCALE like the baseline's."""
    for d in ("training_scripts", "scripts/diagnosis", "evaluation"):  # neck_ablation, diagnose_hca, eval_per_frame
        sys.path.append(os.path.normpath(os.path.join(src_dir, d)))
    import neck_ablation as NA
    import diagnose_hca as DH

    variant_dir = os.path.dirname(os.path.abspath(weights))
    assert os.path.basename(weights) == "last.pth", "--model ablation expects <run>/<variant>/last.pth"
    model, a = DH.load_model(
        os.path.dirname(variant_dir),
        os.path.basename(variant_dir),
        {"z_anchors": 4, "refs_file": "master_refs.pt"},
        device,
    )
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    train = NA.load_split("train", cams_cfg, load_excluded_scenes(), a.refs_file, a.z_anchors)
    NA.precise_bn(model, NA.make_loader(train, 1, a.val_seed + 7, num_workers), device)
    model.eval()
    epoch = torch.load(weights, map_location="cpu", weights_only=False).get("epoch", "?")
    return model, epoch


@torch.no_grad()
def predict_ablation(model, sample, device):
    """Whole-scene map of a neck-ablation model (200 x 200 tiles, as neck_ablation.evaluate)."""
    import eval_per_frame as PF

    return PF.whole_map(model, sample, device)


@torch.no_grad()
def density_zone(model, sample, device, thr, shape, win=25):
    """GT-free density rule (doc/head_ablation.md §6i): True where the dense expert's OWN predicted local density
    (mean over a win x win window, 25 px = 5 m at 5 px/m) is at least thr persons/m^2. Bool [H, W] on the CPU."""
    dense = predict_proposed(model, sample, device, torch.ones(shape, dtype=torch.bool)) / SCALE  # persons per px
    dens = F.avg_pool2d(dense[None, None], win, stride=1, padding=win // 2, count_include_pad=False)
    return (dens[0, 0] * win * win / (win / 5.0) ** 2 >= thr).cpu()


def parse_routing(routing):
    """'learned' | 'oracle' | 'dense' | 'sparse' | 'dzone<thr>', each with an optional trailing 'p' = the sparse
    expert rendered as NMS peaks (kernel 3, threshold 0.064, as eval_map_quality.py). Returns (base, peaks)."""
    peaks = routing.endswith("p") and routing != "learned"
    base = routing[:-1] if peaks else routing
    if base not in ("learned", "oracle", "dense", "sparse") and not base.startswith("dzone"):
        raise SystemExit(f"unknown --routing {routing!r}")
    if base.startswith("dzone"):
        float(base[len("dzone") :])  # validates the threshold
    return base, peaks


@torch.no_grad()
def predict_baseline(model, sample, device):
    images = sample["images"].to(device)
    cam_paras = sample["camera_paras"].to(device)
    wld_paras = sample["wld_map_paras"].to(device)
    h_full, w_full = int(wld_paras[0, 3].item()), int(wld_paras[0, 4].item())
    pred = model(images, cam_paras, wld_paras, full_scene_shape=(h_full, w_full))
    return pred.reshape(h_full, w_full)


def oracle_zone(T, gts, shape, device=None):
    """The ground-truth dense-zone mask the experts and the router are trained against
    (train_proposed_multiscene.py): GT dots blurred with the peak-normalised kernel, thresholded at
    DENSITY_PEAK_MULTIPLIER x a single person's peak, then smoothed (T.oracle_mask_from_conv). Returns a bool [H, W] tensor.
    device: blur on that device (~0.2 s per whole-scene map on the CPU); the result can differ from the CPU blur
    in the last float bits, so --routing oracle keeps the CPU (device=None) for unchanged results."""
    dots = torch.zeros((1, 1) + tuple(shape), device=device)
    if len(gts):
        dots[0, 0, gts[:, 0].astype(int), gts[:, 1].astype(int)] = 1.0
    kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(dots.device)
    conv = F.conv2d(dots, kernel, padding=kernel.shape[-1] // 2).cpu()
    return (T.oracle_mask_from_conv(conv) > 0.5)[0, 0]


@torch.no_grad()
def predict_proposed(model, sample, device, zone=None, return_mask=False):
    """Full scene: backbone once, then neck + head on 200x200 BEV tiles (the training patch size),
    non-overlapping except the last row/column, which is clamped to fit (as in evaluate.py).
    zone=None routes with the learned router; a bool [H, W] mask (oracle_zone) routes with it instead:
    zone * dense + (1 - zone) * sparse, the Stage 2 oracle fusion.
    return_mask (learned routing only): also return the router's soft mask [H, W], tiled like the map."""
    images = sample["images"].to(device)
    refs = sample["ref_points_full"].to(device)
    wld_paras = sample["wld_map_paras"]
    h_full, w_full = int(wld_paras[0, 3].item()), int(wld_paras[0, 4].item())
    tile_h, tile_w = model.bev_shape

    def starts(full, tile):
        return sorted(set(list(range(0, full - tile + 1, tile)) + [full - tile]))

    pred = torch.zeros((h_full, w_full), device=device)
    mask = torch.zeros((h_full, w_full), device=device) if return_mask and zone is None else None
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        B, N = images.shape[:2]
        features = model.backbone(images.flatten(0, 1))
        features = features.view(B, N, *features.shape[1:])
        for h in starts(h_full, tile_h):
            for w in starts(w_full, tile_w):
                extra = (
                    {
                        "ctx": {
                            "camera_paras": sample["camera_paras"].to(device),
                            "wld_map_paras": sample["wld_map_paras"].to(device),
                        },
                        "hw": (h, w),
                    }
                    if getattr(model.neck, "needs_ctx", False)
                    else {}
                )  # AblationNeck (homography)
                bev = model.neck(features, refs[:, :, h : h + tile_h, w : w + tile_w], **extra)
                if zone is None:
                    out, m = model.head(bev)
                    if mask is not None:
                        mask[h : h + tile_h, w : w + tile_w] = m[0, 0].float()
                else:
                    z = zone[h : h + tile_h, w : w + tile_w].to(device).float()
                    out = z * F.relu(model.head.dense_expert(bev)) + (1 - z) * model.head.sparse_map(bev)
                pred[h : h + tile_h, w : w + tile_w] = out[0, 0].float()
    return (pred * model.to_scale, mask) if return_mask else pred * model.to_scale


# ==========================================
# Ground truth / detection helpers
# ==========================================


def gt_pixels(gt_world_xy, wld_paras, shape):
    """Unique GT ground-plane pixels (row, col), rasterised exactly like the dataset's density_patches."""
    gt_world_xy = gt_world_xy.reshape(-1, 2).numpy()
    r, a, b = (wld_paras[0, i].item() for i in range(3))
    w_min, h_min = wld_paras[0, 6].item(), wld_paras[0, 7].item()
    cols = ((gt_world_xy[:, 0] - w_min + a) * r).astype(np.int64)
    rows = ((gt_world_xy[:, 1] - h_min + b) * r).astype(np.int64)
    valid = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    return np.unique(np.stack([rows[valid], cols[valid]], axis=1), axis=0).astype(np.float64)


def detect(pred, cls_thres, nms_radius):
    """Original trainer.test: min-max normalise, threshold, greedy NMS (suppress dist <= radius)."""
    norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    mask = norm > cls_thres
    points = mask.nonzero().cpu().numpy().astype(np.float64)
    scores = norm[mask].cpu().numpy()
    if len(points) == 0:
        return np.empty((0, 2))
    order = np.argsort(-scores, kind="stable")
    points = points[order]
    # Greedy in descending score order; a KD-tree finds the points each kept one suppresses
    # (same result as the original's pairwise loop, just fast enough for ~10k samples)
    tree = cKDTree(points)
    suppressed = np.zeros(len(points), dtype=bool)
    kept = []
    for i in range(len(points)):
        if suppressed[i]:
            continue
        kept.append(i)
        suppressed[tree.query_ball_point(points[i], nms_radius + 1e-9)] = True
    return points[kept]


def detect_multi(pred, thresholds, nms_radius):
    """detect() at several thresholds with one NMS pass. Greedy NMS visits points in descending score order and
    a point is only suppressed by a higher-scoring kept one, so the result at threshold t is the result at the lowest
    threshold restricted to scores > t. Returns {t: points}, each identical to detect(pred, t, nms_radius)."""
    norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    t_min = min(thresholds)
    mask = norm > t_min
    points = mask.nonzero().cpu().numpy().astype(np.float64)
    scores = norm[mask].cpu().numpy()
    if len(points) == 0:
        return {t: np.empty((0, 2)) for t in thresholds}
    order = np.argsort(-scores, kind="stable")
    points, scores = points[order], scores[order]
    tree = cKDTree(points)
    suppressed = np.zeros(len(points), dtype=bool)
    kept = []
    for i in range(len(points)):
        if suppressed[i]:
            continue
        kept.append(i)
        suppressed[tree.query_ball_point(points[i], nms_radius + 1e-9)] = True
    points, scores = points[kept], scores[kept]
    return {t: points[scores > np.float32(t)] for t in thresholds}  # float32, as norm > t in torch


def match(dets, gts, radius):
    """Hungarian matching (CLEAR MOD): pairs further apart than `radius` can't match.
    Solved per connected component of the "within radius" graph, which gives the same optimum as one
    global assignment (pairs across components can never match) at a fraction of the cost."""
    if len(dets) == 0 or len(gts) == 0:
        return 0, len(dets), len(gts), 0.0
    pairs = cKDTree(dets).sparse_distance_matrix(cKDTree(gts), radius, output_type="coo_matrix")
    if pairs.nnz == 0:
        return 0, len(dets), len(gts), 0.0
    n_det = len(dets)
    graph = coo_matrix((np.ones(pairs.nnz), (pairs.row, pairs.col + n_det)), shape=(n_det + len(gts),) * 2)
    _, labels = connected_components(graph, directed=False)
    matched = []
    for comp in np.unique(labels[pairs.row]):
        d_idx = np.where(labels[:n_det] == comp)[0]
        g_idx = np.where(labels[n_det:] == comp)[0]
        dist = np.linalg.norm(dets[d_idx][:, None, :] - gts[g_idx][None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(np.where(dist > radius, 1e6, dist))
        d = dist[rows, cols]
        matched.append(d[d <= radius])
    d = np.concatenate(matched)
    tp = len(d)
    return tp, len(dets) - tp, len(gts) - tp, float(np.sum(1.0 - d / radius))


# ==========================================
# Main
# ==========================================


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def new_stats():
    return {
        "n": 0,
        "abs": 0.0,
        "sq": 0.0,
        "nae": 0.0,
        "signed": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "gt": 0,
        "modp": 0.0,
        "dense": 0.0,
        "dense_n": 0,
    }


def thresholds(args):
    """Detection thresholds scored per sample: the headline --cls_thres first, then --cls_thres_list."""
    return [args.cls_thres] + [t for t in args.cls_thres_list if t != args.cls_thres]


def oracle_module(model):
    """train_proposed_multiscene (oracle_zone's mask definition) for every model type, imported once."""
    if not hasattr(model, "T"):
        import training_scripts.train_proposed_multiscene as T

        model.T = T
    return model.T


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def sample_row(sample, meta, pred, gts, oz, mask, gt_count, pred_count):
    """Conditions, zone split and router quality of one sample (doc/final_evaluations.md E1). oz = the oracle
    dense zone from the GT (the same area for every model); mask = the router's soft mask (learned routing) or None."""
    scene, frame, draw = meta
    images = sample["images"][0]  # [N, 3, H, W], ImageNet-normalised
    grey = ((images * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).mean() * 255).item()
    oz_dev = oz.to(pred.device)
    in_zone = oz[gts[:, 0].astype(int), gts[:, 1].astype(int)].numpy() if len(gts) else np.zeros(0, bool)
    pred_dense = (pred * oz_dev).sum().item() / SCALE
    row = {
        "scene": scene,
        "frame": frame,
        "snapshot": int(sample["snapshot_id"][0]),
        "draw": draw,
        "weather": sample["weather"][0],
        "brightness": round(grey, 2),
        "gt": gt_count,
        "pred": pred_count,
        "err": pred_count - gt_count,
        "oracle_dense_frac": float(oz.float().mean()),
        "gt_dense": int(in_zone.sum()),
        "pred_dense": pred_dense,
        "gt_sparse": int(gt_count - in_zone.sum()),
        "pred_sparse": pred_count - pred_dense,
    }
    if mask is not None:
        hard = mask > 0.5
        inter = (hard & oz_dev).sum().item()
        n_hard, n_oz = hard.sum().item(), oz_dev.sum().item()
        row.update(
            {
                "router_dense_frac": mask.mean().item(),
                "router_iou": inter / (n_hard + n_oz - inter) if n_hard + n_oz - inter else float("nan"),
                "router_prec": inter / n_hard if n_hard else float("nan"),
                "router_rec": inter / n_oz if n_oz else float("nan"),
                "soft_dice": 2 * (mask * oz_dev).sum().item() / (mask.sum().item() + n_oz + 1e-8),
            }
        )
    return row


def predict_map(model, model_name, routing, sample, gts, shape, device):
    """Whole-scene map [H, W] of one sample and the bool dense zone it was routed with (None: baseline, ablation,
    learned router). routing is the base token of parse_routing; gts are needed by the oracle only."""
    if model_name == "baseline":
        return predict_baseline(model, sample, device), None
    if model_name == "ablation":
        return predict_ablation(model, sample, device), None
    if routing == "oracle":
        zone = oracle_zone(model.T, gts, shape)
    elif routing in ("dense", "sparse"):  # one expert alone: the controls for the router
        zone = torch.full(shape, routing == "dense", dtype=torch.bool)
    elif routing.startswith("dzone"):
        zone = density_zone(model, sample, device, float(routing[len("dzone") :]), shape)
    else:
        zone = None
    return predict_proposed(model, sample, device, zone), zone


def run_samples(model, routing, loader, sample_meta, stats, args, device):
    """Predict every sample of `loader` and add its errors and detection counts to stats[scene].
    sample_meta[i] = (scene, frame index in the scene, draw) of the i-th sample. With --cls_thres_list, the
    detection counts per threshold go to stats[scene]["sweep"]; with --dump_samples, one row per sample to
    stats[scene]["rows"]."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # the previous call's (scene's) map size, see below
    prev_shape = None
    thrs = thresholds(args)
    for i, sample in enumerate(tqdm(loader, desc="[Eval]", mininterval=30)):
        wld_paras = sample["wld_map_paras"]
        shape = (int(wld_paras[0, 3].item()), int(wld_paras[0, 4].item()))
        if prev_shape is not None and shape != prev_shape and torch.cuda.is_available():
            # The map size differs per scene AND per frame (each snapshot's calibration gives its own
            # extent), so the caching allocator's blocks from the previous map rarely fit the next one
            # and the reserve grows until a 12 GB card spills into shared memory (seen at 3-9 s/sample
            # instead of ~0.2 on scene_13/scene_80). Release it whenever the map size changes.
            torch.cuda.empty_cache()
        prev_shape = shape
        gts = gt_pixels(sample["gt_world_xy"], wld_paras, shape)
        st = stats.setdefault(sample_meta[i][0], new_stats())
        if routing == "oracle":
            oz = oracle_zone(oracle_module(model), gts, shape)  # CPU: as before, the published oracle numbers
        elif args.dump_samples:
            oz = oracle_zone(oracle_module(model), gts, shape, device)  # only for the zone columns
        else:
            oz = None
        mask = None
        if args.model == "baseline":
            pred = predict_baseline(model, sample, device)
        elif args.model == "ablation":
            pred = predict_ablation(model, sample, device)
        else:
            if routing == "oracle":
                zone = oz
            elif routing in ("dense", "sparse"):  # one expert alone: the controls for the router
                zone = torch.full(shape, routing == "dense", dtype=torch.bool)
            elif routing.startswith("dzone"):
                zone = density_zone(model, sample, device, float(routing[len("dzone") :]), shape)
            else:
                zone = None
            if zone is None:  # learned: the router's mask too, for its dense fraction
                pred, mask = predict_proposed(model, sample, device, None, return_mask=True)
                st["dense"] += float(mask.mean())
            else:
                pred = predict_proposed(model, sample, device, zone)
                st["dense"] += float(zone.float().mean())
            st["dense_n"] += 1

        # counting
        pred_count, gt_count = pred.sum().item() / SCALE, len(gts)
        err = pred_count - gt_count
        st["n"] += 1
        st["abs"] += abs(err)
        st["sq"] += err**2
        st["signed"] += err
        st["nae"] += min(abs(err) / gt_count, 1.0) if gt_count > 0 else 0.0

        # localization: the headline threshold, then the sweep's
        radius_px = args.match_radius_m * wld_paras[0, 0].item()  # r = pixels per metre
        if len(thrs) == 1:
            det = {args.cls_thres: match(detect(pred, args.cls_thres, args.nms_radius_px), gts, radius_px)}
        else:
            det = {t: match(d, gts, radius_px) for t, d in detect_multi(pred, thrs, args.nms_radius_px).items()}
        s_tp, s_fp, s_fn, s_modp = det[args.cls_thres]
        st["tp"] += int(s_tp)
        st["fp"] += int(s_fp)
        st["fn"] += int(s_fn)
        st["gt"] += gt_count
        st["modp"] += float(s_modp)
        if args.cls_thres_list:
            sweep = st.setdefault("sweep", {})
            for t, (tp, fp, fn, modp) in det.items():
                c = sweep.setdefault(str(t), {"tp": 0, "fp": 0, "fn": 0, "modp": 0.0})
                c["tp"] += int(tp)
                c["fp"] += int(fp)
                c["fn"] += int(fn)
                c["modp"] += float(modp)

        if args.dump_samples:
            row = sample_row(sample, sample_meta[i], pred, gts, oz, mask, gt_count, pred_count)
            for t, (tp, fp, fn, modp) in det.items():
                row.update({f"tp_{t}": int(tp), f"fp_{t}": int(fp), f"fn_{t}": int(fn), f"modp_{t}": float(modp)})
            st.setdefault("rows", []).append(row)


def detection_metrics(tp, fp, fn, modp, n_gt):
    """MODA, MODP, Precision, Recall, F1 in percent, pooled over samples (CLEAR MOD)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    return {
        "MODA": 100.0 * (1.0 - (fp + fn) / n_gt),
        "MODP": 100.0 * modp / tp if tp else 0.0,
        "Precision": 100.0 * precision,
        "Recall": 100.0 * recall,
        "F1": 100.0 * (2 * precision * recall / (precision + recall) if precision + recall else 0.0),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
    }


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=src_dir, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = load_model(args.model, args.weights, device)
    if args.routing != "learned" and args.model != "proposed":
        raise SystemExit(f"--routing {args.routing} only applies to --model proposed")
    if args.routing.startswith("@"):  # routing token from a file (sweep_density_rule.py --write_routing)
        with open(args.routing[1:]) as f:
            args.routing = f.read().strip()
        print(f"[routing] {args.routing} (from file)")
    routing, peaks = parse_routing(args.routing)
    if peaks:
        model.T.apply_sparse_render(model, {"mode": "peaks", "kernel": 3, "threshold": 0.064})
        model.head.render_kernel = model.head.render_kernel.to(device)
    resized = set_camera_count(model, args.num_views)  # camera ablation; no-op at 5
    if resized:
        print(
            f"[cameras] {len(resized)} cross-attention(s) resized to {args.num_views} slots (trained slots reused cyclically)"
        )
    if args.view_norm:
        print(f"[cameras] renormalised to the trained 5 views: {renormalise_views(model, args.num_views)}")
    print(
        f"--- Paper-protocol evaluation | model={args.model} (epoch {epoch}) | routing={args.routing} | split={args.split} | "
        f"{args.view_samplings} draws of {args.num_views} views/frame | device={device} ---"
    )

    split_dir = os.path.join(src_dir, "datasets", "SynMVCrowd", "data", args.split)
    excluded = load_excluded_scenes()
    datasets, scene_names = [], []
    for scene in sorted(os.listdir(split_dir)):
        if not os.path.isdir(os.path.join(split_dir, scene)):
            continue
        if scene in excluded:
            print(f"[EXCLUDED] {args.split}/{scene}")
            continue
        datasets.append(
            SynMVCrowdDataset(
                scene_dir=os.path.join(split_dir, scene),
                camera_ids=None,
                num_sampled_cams=args.num_views,
                load_refs=(args.model != "baseline"),
            )
        )
        scene_names.append(scene)

    rng = np.random.RandomState(args.seed)
    indices, sample_meta, scene_frames, scene_meta, offset = [], [], {}, {}, 0
    for scene, ds in zip(scene_names, datasets):
        n = min(args.frames_per_scene, len(ds)) if args.frames_per_scene else len(ds)
        frames = sorted(rng.choice(len(ds), size=n, replace=False).tolist())
        scene_frames[scene] = [f for f in frames for _ in range(args.view_samplings)]
        scene_meta[scene] = [(scene, f, d) for f in frames for d in range(args.view_samplings)]  # (scene, frame, draw)
        for f in frames:
            indices.extend([offset + f] * args.view_samplings)
        sample_meta.extend(scene_meta[scene])
        offset += len(ds)
    print(f"{len(scene_names)} scenes -> {len(indices)} samples")

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.weights)),
        f"eval_{args.split}_{os.path.splitext(os.path.basename(args.weights))[0]}"
        f"{'' if args.routing == 'learned' else '_' + args.routing}"
        f"{'' if args.num_views == 5 else f'_{args.num_views}v'}"
        f"{'_vnorm' if args.view_norm else ''}.json",
    )
    stats = {}  # per scene: sums of the per-sample errors and the detection counts
    t0 = time.time()
    if not args.per_scene:
        loader = DataLoader(
            ConcatDataset(datasets),
            batch_size=1,
            sampler=indices,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed),
        )
        run_samples(model, routing, loader, sample_meta, stats, args, device)
    else:
        # scene by scene, saved after every scene; a rerun with the same settings continues after the last saved one
        partial = out + ".partial.json"
        config = {
            k: getattr(args, k)
            for k in (
                "model",
                "weights",
                "routing",
                "split",
                "view_samplings",
                "num_views",
                "view_norm",
                "frames_per_scene",
                "seed",
                "cls_thres",
                "nms_radius_px",
                "match_radius_m",
                "cls_thres_list",
                "dump_samples",
            )
        }
        if os.path.isfile(partial):
            with open(partial) as f:
                saved = json.load(f)
            if saved["config"] == config:
                stats = saved["scenes"]
                print(f"[resume] {len(stats)}/{len(scene_names)} scenes already done ({partial})")
            else:
                print(f"[resume] {partial} has other settings, starting over")
        for k, (scene, ds) in enumerate(zip(scene_names, datasets)):
            if scene in stats:
                continue
            t1 = time.time()
            loader = DataLoader(
                ds,
                batch_size=1,
                sampler=scene_frames[scene],
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(args.seed * 1000 + k),
            )
            run_samples(model, routing, loader, scene_meta[scene], stats, args, device)
            stats[scene]["seconds"] = round(time.time() - t1)
            with open(partial, "w") as f:
                json.dump({"config": config, "scenes": stats}, f, indent=2)
            print(
                f"[scene {k + 1}/{len(scene_names)}] {scene}: MAE {stats[scene]['abs'] / stats[scene]['n']:.2f} "
                f"({stats[scene]['seconds']} s), saved to {partial}"
            )

    tot = {key: sum(s[key] for s in stats.values()) for key in new_stats()}
    tp, fp, n_gt = tot["tp"], tot["fp"], tot["gt"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    results = {
        "model": args.model,
        "weights": args.weights,
        "epoch": epoch,
        "routing": args.routing,
        "split": args.split,
        "samples": tot["n"],
        "view_samplings": args.view_samplings,
        "num_views": args.num_views,
        "view_norm": args.view_norm,
        "seed": args.seed,
        "per_scene_loaders": args.per_scene,
        "MAE": tot["abs"] / tot["n"],
        "MSE": math.sqrt(tot["sq"] / tot["n"]),
        "NAE": tot["nae"] / tot["n"],
        "MODA": 100.0 * (1.0 - (fp + tot["fn"]) / n_gt),
        "MODP": 100.0 * tot["modp"] / tp if tp else 0.0,
        "Precision": 100.0 * precision,
        "Recall": 100.0 * recall,
        "F1": 100.0 * (2 * precision * recall / (precision + recall) if precision + recall else 0.0),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(tot["fn"]),
        "GT": int(n_gt),
        "bias": tot["signed"] / tot["n"],
        "per_scene_MAE": {s: stats[s]["abs"] / stats[s]["n"] for s in scene_names},
        "per_scene_bias": {s: stats[s]["signed"] / stats[s]["n"] for s in scene_names},
        "dense_zone_frac": tot["dense"] / tot["dense_n"] if tot["dense_n"] else None,
        "settings": {
            "cls_thres": args.cls_thres,
            "nms_radius_px": args.nms_radius_px,
            "match_radius_m": args.match_radius_m,
        },
        "seconds": round(time.time() - t0) if not args.per_scene else sum(s["seconds"] for s in stats.values()),
        "num_workers": args.num_workers,  # the view draws depend on it (one RNG per loader worker)
        "git_commit": git_commit(),
    }
    if args.cls_thres_list:  # detection threshold sweep (the headline above stays at --cls_thres)
        results["threshold_sweep"] = {}
        for t in thresholds(args):
            c = {k: sum(s["sweep"][str(t)][k] for s in stats.values()) for k in ("tp", "fp", "fn", "modp")}
            results["threshold_sweep"][str(t)] = detection_metrics(c["tp"], c["fp"], c["fn"], c["modp"], n_gt)
    if args.dump_samples:  # one row per sample, scenes in split order
        rows = [r for s in scene_names for r in stats[s]["rows"]]
        results["samples_csv"] = os.path.splitext(os.path.basename(out))[0] + ".samples.csv"
        with open(os.path.splitext(out)[0] + ".samples.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)

    print("\n" + "=" * 60)
    print(
        f"PAPER-PROTOCOL RESULTS -- {args.model} ({args.routing} routing), {args.split} ({results['samples']} samples)"
    )
    print("-" * 60)
    print(f"  MAE {results['MAE']:.2f} | MSE {results['MSE']:.2f} | NAE {results['NAE']:.3f}")
    print(
        f"  MODA {results['MODA']:.1f} | MODP {results['MODP']:.1f} | Precision {results['Precision']:.1f} | "
        f"Recall {results['Recall']:.1f} | F1 {results['F1']:.1f}"
    )
    print("-" * 60)
    for s, v in results["per_scene_MAE"].items():
        print(f"  {s}: MAE {v:.2f}")
    print("=" * 60)

    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out}" + (f" and {results['samples_csv']}" if args.dump_samples else ""))
    if args.per_scene and os.path.isfile(out + ".partial.json"):
        os.remove(out + ".partial.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SynMVCrowd paper-protocol evaluation")
    parser.add_argument("--model", type=str, required=True, choices=["baseline", "proposed", "ablation"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "validate", "test"])
    parser.add_argument("--view_samplings", type=int, default=5, help="Random 5-view draws per frame (paper: 5).")
    parser.add_argument(
        "--num_views",
        type=int,
        default=5,
        help="Views per draw (paper: 5; camera ablation: 3, 7, 10). Results get a _<n>v suffix.",
    )
    parser.add_argument(
        "--view_norm",
        action="store_true",
        help="Rescale the view softmax (x N/5) / ca_sa camera sum (x 5/N) to the trained 5-view "
        "magnitude (utils/camera_slots.renormalise_views). Results get a _vnorm suffix.",
    )
    parser.add_argument("--frames_per_scene", type=int, default=0, help="0 = all frames (paper).")
    parser.add_argument("--cls_thres", type=float, default=0.2, help="Original trainer default.")
    parser.add_argument("--nms_radius_px", type=float, default=2.5, help="Original trainer NMS radius.")
    parser.add_argument("--match_radius_m", type=float, default=0.5, help="0.5 m = 2.5 px at 0.2 m/px.")
    parser.add_argument(
        "--routing",
        type=str,
        default="learned",
        help="proposed only: 'oracle' replaces the router by the ground-truth dense-zone mask "
        "(upper bound for the router); 'dense' / 'sparse' use one expert alone (controls); "
        "'dzone<thr>' = the density rule (dense expert where its own density >= thr p/m^2); a "
        "trailing 'p' renders the sparse expert as NMS peaks (e.g. dzone0.25p). "
        "Non-learned routings add a _<routing> suffix to the results file.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--out", type=str, default=None, help="Results JSON (default: next to the weights).")
    parser.add_argument(
        "--per_scene",
        action="store_true",
        help="one seeded loader per scene, results saved to <out>.partial.json after every scene; "
        "a rerun with the same settings continues after the last saved scene (long camera-ablation "
        "evals). Same frames, but other view draws than a single-loader run (same distribution).",
    )
    parser.add_argument(
        "--dump_samples",
        action="store_true",
        help="write one row per sample (conditions, count, detections per threshold, GT-based dense / sparse "
        "zone split, router mask quality) to <out without .json>.samples.csv (doc/final_evaluations.md E1)",
    )
    parser.add_argument(
        "--cls_thres_list",
        type=lambda v: [float(t) for t in v.split(",") if t],
        default=[],
        help="extra detection thresholds, e.g. 0.1,0.2,0.4; results['threshold_sweep'] (the headline stays "
        "at --cls_thres)",
    )
    evaluate(parser.parse_args())
