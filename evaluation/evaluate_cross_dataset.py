"""
evaluation/evaluate_cross_dataset.py

Evaluates SynMVCrowd-trained models on Wildtrack and MultiviewX without fine-tuning (the "Baseline (test)"
rows of the SynMVCrowd paper's Table 4), with the scoring of evaluate_paper_protocol.py: whole-map
prediction, counting as pred.sum() / 100 vs. the GT count (MAE, root-MSE, NAE capped at 1, signed bias),
localization by min-max normalisation, --cls_thres, greedy NMS (--nms_radius_px) and Hungarian matching
within --match_radius_m (MODA, MODP, Precision, Recall, F1 pooled over all samples).

Differences to the SynMVCrowd protocol, all from the datasets (see datasets/multiview_detection.py):
- one scene, MVDet split (test = last 10 % of the frames, 40 frames on both datasets);
- only the annotated ground area is scored (Wildtrack 12 x 36 m, MultiviewX 25 x 16 m), since people outside it
  are not labelled. The models see a map widened to --min_map_px (200, the proposed models' tile; 0 = the annotated
  area only, as the 2026-10-22 baseline results), and the prediction is cut to the annotated area before counting
  and detection;
- views: --num_views 0 (default) feeds all cameras (7 / 6), the benchmark setting; --num_views 5 with
  --view_samplings 5 reproduces the SynMVCrowd draws (5 random views per draw, 5 draws per frame). Models trained
  on 5 views get their per-camera slots resized (utils/camera_slots.set_camera_count) and, with --view_norm, the
  trained 5-view magnitude restored (renormalise_views), as in the camera ablation;
- the projection plane is --head_height above the ground (the models were trained on the head plane;
  these datasets annotate feet at z = 0).

Usage (from src/):
    python evaluation/evaluate_cross_dataset.py --dataset wildtrack --weights runs/baseline/ot/best.pth
    python evaluation/evaluate_cross_dataset.py --dataset multiviewx --weights runs/baseline/ot/best.pth --num_views 5 --view_samplings 5
    python evaluation/evaluate_cross_dataset.py --dataset wildtrack --model proposed --routing dzone0.1p --view_norm --weights <checkpoint>.pth
"""

import os
import sys
import json
import math
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from datasets.Wildtrack.wildtrack import WildtrackDataset
from datasets.MultiviewX.multiviewx import MultiviewXDataset
from evaluation.evaluate_paper_protocol import (
    SCALE,
    load_model,
    parse_routing,
    predict_map,
    gt_pixels,
    detect,
    match,
    seed_worker,
)
from utils.camera_slots import set_camera_count, renormalise_views

DATASETS = {
    "wildtrack": (WildtrackDataset, os.path.join(src_dir, "datasets", "Wildtrack", "data")),
    "multiviewx": (MultiviewXDataset, os.path.join(src_dir, "datasets", "MultiviewX", "data")),
}


def crop_to_window(pred, gts, zone, window):
    """Cuts the map, the GT pixels and the dense zone to the annotated area window = [row0, col0, h, w]."""
    r0, c0, h, w = window
    gts = gts - np.array([r0, c0], dtype=np.float64)
    gts = gts[(gts[:, 0] >= 0) & (gts[:, 0] < h) & (gts[:, 1] >= 0) & (gts[:, 1] < w)]
    zone = None if zone is None else zone[r0 : r0 + h, c0 : c0 + w]
    return pred[r0 : r0 + h, c0 : c0 + w], gts, zone


def evaluate(args, model=None, epoch="?"):
    """Scores args.weights, or `model` when given (training_scripts/finetune_cross_dataset.py: a fine-tuned model
    already resized to the camera count, whose weights load_model could not rebuild). Returns the results dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model, epoch = load_model(args.model, args.weights, device)
    model.eval()
    if args.routing != "learned" and args.model != "proposed":
        raise SystemExit(f"--routing {args.routing} only applies to --model proposed")
    routing, peaks = parse_routing(args.routing)
    if peaks:
        model.T.apply_sparse_render(model, {"mode": "peaks", "kernel": 3, "threshold": 0.064})
        model.head.render_kernel = model.head.render_kernel.to(device)

    cls, root = DATASETS[args.dataset]
    dataset = cls(
        root,
        split=args.split,
        num_sampled_cams=args.num_views or None,
        head_height=args.head_height,
        min_map_px=(args.min_map_px, args.min_map_px) if args.min_map_px else None,
        ref_points=(args.model == "proposed"),
    )
    all_views = not args.num_views or args.num_views >= dataset.NUM_CAMS
    num_views = dataset.NUM_CAMS if all_views else args.num_views
    draws = 1 if all_views else args.view_samplings  # every draw would be identical
    views_tag = f"{num_views}v" if all_views else f"{num_views}v{draws}d"
    resized = set_camera_count(model, num_views)
    if resized:
        print(
            f"[cameras] {len(resized)} cross-attention(s) resized to {num_views} slots (trained slots reused cyclically)"
        )
    if args.view_norm:
        print(f"[cameras] renormalised to the trained 5 views: {renormalise_views(model, num_views)}")
    window = dataset.eval_window().tolist()
    print(
        f"--- Cross-dataset evaluation | {args.dataset} {args.split} ({len(dataset)} frames) | {args.model} "
        f"(epoch {epoch}) | routing={args.routing} | views {views_tag}{' vnorm' if args.view_norm else ''} | "
        f"map {dataset.wld_map_paras()[3:5].int().tolist()}, scored {window} | head plane {args.head_height} m | "
        f"device={device} ---"
    )

    indices = [i for i in range(len(dataset)) for _ in range(draws)]
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=indices,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
    )

    errs, nae, gt_counts, dense = [], [], [], []
    tp = fp = fn = n_gt = 0
    modp_sum = 0.0
    t0 = time.time()
    for sample in tqdm(loader, desc="[Eval]", mininterval=30):
        wld_paras = sample["wld_map_paras"]
        shape = (int(wld_paras[0, 3].item()), int(wld_paras[0, 4].item()))
        gts = gt_pixels(sample["gt_world_xy"], wld_paras, shape)
        pred, zone = predict_map(model, args.model, routing, sample, gts, shape, device)
        pred, gts, zone = crop_to_window(pred, gts, zone, sample["eval_window"][0].tolist())
        if zone is not None:
            dense.append(float(zone.float().mean()))

        # counting
        pred_count, gt_count = pred.sum().item() / SCALE, len(gts)
        errs.append(pred_count - gt_count)
        nae.append(min(abs(errs[-1]) / gt_count, 1.0) if gt_count > 0 else 0.0)
        gt_counts.append(gt_count)

        # localization
        dets = detect(pred, args.cls_thres, args.nms_radius_px)
        radius_px = args.match_radius_m * wld_paras[0, 0].item()  # r = pixels per metre
        s_tp, s_fp, s_fn, s_modp = match(dets, gts, radius_px)
        tp, fp, fn, n_gt, modp_sum = tp + s_tp, fp + s_fp, fn + s_fn, n_gt + gt_count, modp_sum + s_modp

    errs = np.array(errs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    results = {
        "dataset": args.dataset,
        "model": args.model,
        "routing": args.routing,
        "weights": args.weights,
        "epoch": epoch,
        "split": args.split,
        "frames": len(dataset),
        "samples": len(errs),
        "num_views": num_views,
        "view_samplings": draws,
        "view_norm": args.view_norm,
        "head_height": args.head_height,
        "map": dataset.wld_map_paras()[3:5].int().tolist(),
        "eval_window": window,
        "seed": args.seed,
        "MAE": float(np.mean(np.abs(errs))),
        "MSE": float(math.sqrt(np.mean(errs**2))),
        "NAE": float(np.mean(nae)),
        "bias": float(np.mean(errs)),
        "mean_GT_count": float(np.mean(gt_counts)),
        "MODA": 100.0 * (1.0 - (fp + fn) / n_gt),
        "MODP": 100.0 * modp_sum / tp if tp else 0.0,
        "Precision": 100.0 * precision,
        "Recall": 100.0 * recall,
        "F1": 100.0 * (2 * precision * recall / (precision + recall) if precision + recall else 0.0),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "GT": int(n_gt),
        "dense_zone_frac": float(np.mean(dense)) if dense else None,  # share of the annotated area routed to dense
        "settings": {
            "cls_thres": args.cls_thres,
            "nms_radius_px": args.nms_radius_px,
            "match_radius_m": args.match_radius_m,
        },
        "seconds": round(time.time() - t0),
    }

    print("\n" + "=" * 60)
    print(
        f"CROSS-DATASET RESULTS -- {args.model} ({args.routing}) on {args.dataset} {args.split} "
        f"({results['samples']} samples, views {views_tag}{' vnorm' if args.view_norm else ''})"
    )
    print("-" * 60)
    print(
        f"  MAE {results['MAE']:.2f} | MSE {results['MSE']:.2f} | NAE {results['NAE']:.3f} | "
        f"bias {results['bias']:+.2f} (mean GT {results['mean_GT_count']:.1f})"
    )
    print(
        f"  MODA {results['MODA']:.1f} | MODP {results['MODP']:.1f} | Precision {results['Precision']:.1f} | "
        f"Recall {results['Recall']:.1f} | F1 {results['F1']:.1f}"
        + (f" | dense zone {100 * results['dense_zone_frac']:.1f} %" if dense else "")
    )
    print("=" * 60)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.weights)),
        f"eval_{args.dataset}_{args.split}_{views_tag}"
        f"{'' if args.routing == 'learned' else '_' + args.routing}{'_vnorm' if args.view_norm else ''}_"
        f"{os.path.splitext(os.path.basename(args.weights))[0]}.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out}")
    return results


def build_parser():
    parser = argparse.ArgumentParser(description="Cross-dataset (Wildtrack / MultiviewX) evaluation")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS))
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "proposed"])
    parser.add_argument(
        "--routing",
        type=str,
        default="learned",
        help="proposed only: learned | oracle | dense | sparse | dzone<thr>, optional trailing p (evaluate_paper_protocol).",
    )
    parser.add_argument("--view_norm", action="store_true", help="Restore the trained 5-view magnitude.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--num_views", type=int, default=0, help="0 = all cameras; else random draws of this many.")
    parser.add_argument("--view_samplings", type=int, default=5, help="Draws per frame when --num_views < all.")
    parser.add_argument("--head_height", type=float, default=1.75, help="Projection plane above the ground (m).")
    parser.add_argument(
        "--min_map_px", type=int, default=200, help="Widen the map to at least this size (0 = annotated area only)."
    )
    parser.add_argument("--cls_thres", type=float, default=0.2, help="Original trainer default.")
    parser.add_argument("--nms_radius_px", type=float, default=2.5, help="Original trainer NMS radius.")
    parser.add_argument("--match_radius_m", type=float, default=0.5, help="0.5 m = 2.5 px at 0.2 m/px (MVDet).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out", type=str, default=None, help="Results JSON (default: next to the weights).")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
