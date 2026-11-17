"""
training_scripts/finetune_cross_dataset.py

Fine-tunes a SynMVCrowd-trained model on Wildtrack or MultiviewX with a share of the train split, then scores the
final weights on the test split with evaluation/evaluate_cross_dataset.py (doc/cross_dataset.md, WP4; the paper's
"Baseline (10% / 100%)" rows of Table 4).

Same for every model: all cameras, the trained 5-view magnitude restored at the start (--view_norm of the zero-shot
eval) and the per-camera slots resized to the camera count; the map widened to 200 x 200 (one patch for the proposed
models, the whole map for the baseline); Adam (--lr, --weight_decay), --iters steps of one frame each, gradient
clipping at the proposed trainer's MAX_GRAD_NORM, --seed (it also picks the --fraction subset, so the subset is the
same for every model); no validation, the FINAL weights are scored. Each model keeps its own trainer's precision
(proposed: autocast forward, fp32 losses; baseline: fp32).

The loss is each model's own, on the prediction multiplied by the annotated-area mask (eval_window): people outside
it are real but unlabelled, so a loss on the whole map would teach the model to suppress them.
- baseline (--model baseline): the baseline's OT loss (SynMVCrowdGeneralizedLoss, 'ed' cost, 100 per person), as
  train_baseline.py --loss ot;
- learned routing (h1_hom, h1_casa): the baseline's OT loss on the fused map (converted to its 100-per-person units)
  plus Stage 4's 0.1 Tversky anchor of the router to the oracle mask (alpha 0.7), neck + experts + router trainable.
  Not the trainer's Stage 4 with --ot_weight 1: its OT loss (GeneralizedLoss) renormalises the map to a distribution
  and cannot see the count, and with weight 1 nothing else does -- in the 2026-10-30 remote smoke the train count
  bias fell from +28 / +41 to -19 / -25 (GT ~24) within 50 steps;
- dzone routing (hybrid_hom): its rule is not trainable, so the experts train with their Stage 2 losses on the whole
  map (--stage2_unmasked; dense MSE + outside-weight 0.7, deep sparse expert with the S4 OT loss, count weight 0.01),
  neck + experts trainable; the rule stays as it is.
The proposed recipes go through the trainer's own command line (proposed_trainer.cli / config.apply_args), so the
loss settings are exactly the trainer's. --train_backbone also trains the backbone (each trainer's BatchNorm rule).

Output in --out_dir: last.pth (every --save_every steps; a rerun resumes from it), final.pth, eval_test.json and
result.json (the test scores + the fine-tuning settings; the launcher's "done" file), train_log.csv.

Usage (from src/):
    python training_scripts/finetune_cross_dataset.py --model baseline --weights runs/baseline/ot/best.pth \
        --dataset wildtrack --fraction 0.1 --out_dir runs/cross_dataset/finetune/baseline_ot_wildtrack_10
    python training_scripts/finetune_cross_dataset.py --model proposed --routing dzone0.1p --weights <checkpoint>.pth \
        --dataset multiviewx --fraction 1.0 --out_dir runs/cross_dataset/finetune/hybrid_hom_multiviewx_100
"""

import os
import sys
import csv
import json
import math
import argparse
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from evaluation import evaluate_cross_dataset as E
from evaluation.evaluate_paper_protocol import SCALE, load_model, parse_routing, gt_pixels, seed_worker
from losses.ot_loss.genloss_synmvcrowd import SynMVCrowdGeneralizedLoss, patch_points
from training_scripts.proposed_trainer import config as cfg
from training_scripts.proposed_trainer.cli import build_parser as build_trainer_parser
from training_scripts.proposed_trainer.losses import StageLosses, mask_loss
from training_scripts.proposed_trainer.targets import PatchTargets, extract_camera_locations
from training_scripts.proposed_trainer.training import set_train_modes
from utils.camera_slots import set_camera_count, renormalise_views

LOSS_NAMES = {None: "baseline OT", 4: "baseline OT on the fused map + 0.1 mask anchor", 2: "Stage 2 expert losses"}
MIN_MAP_PX = 200  # the proposed models' tile; the whole widened map is one patch
# the trainer flags of each proposed recipe (remote_scripts/run_remote_budget_ca_sa_h1.ps1 Stage 4,
# run_remote_budget_homography_hybrid.ps1 S4D2); Stage 4 takes only the anchor from them (its loss is the baseline's OT)
H1_RECIPE = ["--stage", "4", "--tversky_alpha", "0.7", "--stage4_mask_anchor_weight", "0.1"]
HYBRID_RECIPE = (
    "--stage 2 --stage2_unmasked --dense_out_weight 0.7 --count_loss_weight 0.01 --route_cell 10 "
    "--sparse_loss genloss --sparse_genloss_weight 1.0 --sparse_genloss_topk 1000 --sparse_genloss_count_weight 1.0 "
    "--sparse_nms_mult 1 --sparse_nms_kernel 3 --sparse_genloss_bg_weight 1.0 --sparse_genloss_bg_mode mass "
    "--sparse_bg_empty_thr 1e-3 --sparse_arch deep"
).split()


def recipe(args):
    """(stage, trainer argv) of the model's loss; stage None = the baseline's OT loss."""
    if args.model == "baseline":
        return None, None
    base, _ = parse_routing(args.routing)
    if base == "learned":
        argv = list(H1_RECIPE)
    elif base.startswith("dzone"):
        argv = list(HYBRID_RECIPE)
    else:
        raise SystemExit(f"--routing {args.routing}: no fine-tuning recipe (learned or dzone<thr> only)")
    argv += ["--no_router_contrast"]
    if args.train_backbone:
        argv += ["--train_backbone", "all"]
    return int(argv[1]), argv


def frame_order(n_frames, args):
    """The seeded --fraction subset of the train frames and the frame of every step (shuffled passes over it)."""
    n = max(1, round(args.fraction * n_frames))
    subset = sorted(np.random.RandomState(args.seed).choice(n_frames, size=n, replace=False).tolist())
    rng = np.random.RandomState(args.seed + 1)
    order = np.concatenate([rng.permutation(subset) for _ in range(math.ceil(args.iters / n))])[: args.iters]
    return subset, [int(i) for i in order]


def gt_points(sample, shape, device):
    """GT points of the sample's map (rasterised as the evaluation does) in the baseline OT loss's format; `shape`
    may be a crop from (0, 0) (the proposed models' 200 x 200 patch), which keeps every annotated person."""
    gts = gt_pixels(sample["gt_world_xy"], sample["wld_map_paras"], shape).astype(np.int64)
    points = torch.zeros(shape, device=device)
    points[gts[:, 0], gts[:, 1]] = 1.0
    return patch_points(points)


def window_mask(window, shape, device):
    """[1, 1, H, W] float mask of the annotated area window = [row0, col0, h, w] in a map of `shape`."""
    r0, c0, h, w = window
    mask = torch.zeros((1, 1) + tuple(shape), device=device)
    mask[..., r0 : r0 + h, c0 : c0 + w] = 1.0
    return mask


def finetune(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    proposed = args.model == "proposed"

    stage, argv = recipe(args)
    if proposed:  # before load_model: the recipe's settings, then the checkpoint's architecture on top
        parser = build_trainer_parser()
        cfg.apply_args(parser.parse_args(argv), parser)
    model, epoch0 = load_model(args.model, args.weights, device)

    cls, root = E.DATASETS[args.dataset]
    train_set = cls(
        root,
        split="train",
        head_height=args.head_height,
        min_map_px=(MIN_MAP_PX, MIN_MAP_PX),
        ref_points=proposed,
    )
    n_views = train_set.NUM_CAMS
    resized = set_camera_count(model, n_views)
    scales = renormalise_views(model, n_views)
    subset, order = frame_order(len(train_set), args)
    wld = train_set.wld_map_paras()
    shape = (int(wld[3]), int(wld[4]))
    window = train_set.eval_window().tolist()
    mask_full = window_mask(window, shape, device)
    mask_patch = mask_full[..., : cfg.BEV_SHAPE[0], : cfg.BEV_SHAPE[1]]  # MultiviewX is 200 x 201: patch at (0, 0)

    # trainable parts and the loss
    if proposed:
        model.set_stage_trainability(stage)
        losses = StageLosses(stage, device)
        from training_scripts.train_proposed_multiscene import TrainStats

        stats = TrainStats()
    else:
        model.train_backbone = args.train_backbone
        for name, p in model.named_parameters():
            p.requires_grad = args.train_backbone or not name.startswith("backbone.")
    ot_criterion = SynMVCrowdGeneralizedLoss(cost="ed", scale=SCALE)  # baseline and Stage 4
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=proposed and device.type == "cuda")
    to_count = getattr(model, "to_scale", 1.0) / SCALE  # map sum -> persons

    print(
        f"--- Cross-dataset fine-tuning | {args.model} ({args.routing}) from {args.weights} (epoch {epoch0}) | "
        f"{args.dataset} train: {len(subset)}/{len(train_set)} frames ({100 * args.fraction:g} %, seed {args.seed}) | "
        f"{args.iters} steps, Adam lr {args.lr:g}, wd {args.weight_decay:g} | "
        f"{LOSS_NAMES[stage]} | backbone "
        f"{'trainable' if args.train_backbone else 'frozen'} | {sum(p.numel() for p in params):,} trainable params | "
        f"{n_views} views, slots resized {resized}, view scales {scales} | map {list(shape)}, loss on {window} | "
        f"device={device} ---"
    )
    if proposed:
        print(f"[recipe] trainer flags: {' '.join(argv)}")

    last = os.path.join(args.out_dir, "last.pth")
    start = 0
    if os.path.isfile(last):
        ckpt = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start = ckpt["iteration"]
        print(f"[resume] {last}: step {start}/{args.iters}")

    def save(path, iteration):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "iteration": iteration,
                "args": vars(args),
            },
            path,
        )

    loader = DataLoader(
        train_set,
        batch_size=1,
        sampler=order[start:],
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + start),
    )
    log_path = os.path.join(args.out_dir, "train_log.csv")
    log_rows, window_loss, window_err, skipped = [], [], [], 0
    if proposed:
        set_train_modes(model, stage, args.train_backbone)
    else:
        model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for it, sample in enumerate(tqdm(loader, desc="[Fine-tune]", mininterval=30), start=start + 1):
        images = sample["images"].to(device)
        cam_paras = sample["camera_paras"].to(device)
        wld_paras = sample["wld_map_paras"].to(device)
        if proposed:
            hw = torch.zeros((1, 2, 1), dtype=torch.int32, device=device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                patches, masks = model(
                    images,
                    sample["ref_points_full"].to(device),
                    hw,
                    stage=stage,
                    camera_paras=cam_paras,
                    wld_map_paras=wld_paras,
                )
            pred, pred_mask = patches[:, 0].float() * mask_patch, masks[:, 0].float() * mask_patch
            with torch.amp.autocast("cuda", enabled=False):
                cam_locations = extract_camera_locations(cam_paras).squeeze(0)
                t = PatchTargets(
                    sample["gt_world_xy"], wld_paras, hw[:, :, 0], cam_locations, losses.map_kernel, device
                )
                if stage == 4:  # the baseline's OT loss on the fused map (in its units) + the router anchor
                    loss = ot_criterion(pred * model.to_scale, gt_points(sample, cfg.BEV_SHAPE, device))
                    anchor = mask_loss(torch.clamp(pred_mask, min=1e-6, max=1.0 - 1e-6), t.gt_mask)
                    loss = loss + cfg.STAGE4_MASK_ANCHOR_WEIGHT * anchor
                else:
                    loss = losses.patch_loss(model, pred, pred_mask, t, stats)
        else:
            pred = model(images, cam_paras, wld_paras, full_scene_shape=shape).reshape(1, 1, *shape) * mask_full
            loss = ot_criterion(pred, gt_points(sample, shape, device))

        if not torch.isfinite(loss):
            skipped += 1
            print(f"WARNING: non-finite loss at step {it}, skipping the update")
            continue
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, max_norm=cfg.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        window_loss.append(loss.item())
        window_err.append(pred[:, 0].sum().item() * to_count - len(sample["gt_world_xy"][0]))
        if it % args.log_every == 0 or it == args.iters:
            row = {
                "step": it,
                "loss": float(np.mean(window_loss)),
                "count_bias": float(np.mean(window_err)),  # first channel (dense expert in Stage 2)
                "s_per_step": (time.time() - t0) / len(window_loss),
                "peak_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
            }
            log_rows.append(row)
            print(
                f"[step {it}/{args.iters}] loss {row['loss']:.4f} | count bias {row['count_bias']:+.2f} | "
                f"{row['s_per_step']:.2f} s/step | peak {row['peak_gb']:.1f} GB"
            )
            with open(log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            window_loss, window_err, t0 = [], [], time.time()
        if it % args.save_every == 0:
            save(last, it)
    save(last, args.iters)
    torch.save(
        {"model_state_dict": model.state_dict(), "num_cameras": n_views, "view_scales": scales, "args": vars(args)},
        os.path.join(args.out_dir, "final.pth"),
    )

    # test: the final weights, scored like the zero-shot evaluation (view_norm keeps the scales trained with)
    eval_args = E.build_parser().parse_args(
        [
            "--dataset", args.dataset, "--weights", args.weights, "--model", args.model, "--routing", args.routing,
            "--view_norm", "--head_height", str(args.head_height), "--min_map_px", str(MIN_MAP_PX),
            "--num_workers", str(args.num_workers), "--out", os.path.join(args.out_dir, "eval_test.json"),
        ]
    )  # fmt: skip
    results = E.evaluate(eval_args, model=model, epoch=f"finetuned {args.iters} steps")
    results["finetune"] = {
        **{k: getattr(args, k) for k in ("fraction", "iters", "lr", "weight_decay", "seed", "train_backbone")},
        "frames": len(subset),
        "train_frames": len(train_set),
        "stage": stage,
        "loss": LOSS_NAMES[stage],
        "recipe": argv,
        "skipped_steps": skipped,
        "last_log": log_rows[-1] if log_rows else None,
    }
    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {os.path.join(args.out_dir, 'result.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-dataset fine-tuning (Wildtrack / MultiviewX)")
    parser.add_argument("--model", type=str, required=True, choices=["baseline", "proposed"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--routing", type=str, default="learned", help="learned (h1) | dzone<thr>[p] (hybrid_hom).")
    parser.add_argument("--dataset", type=str, required=True, choices=list(E.DATASETS))
    parser.add_argument("--fraction", type=float, required=True, help="Share of the train split (0.1, 1.0).")
    parser.add_argument("--iters", type=int, default=3600, help="Optimiser steps of one frame (all cameras) each.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_backbone", action="store_true", help="Also train the backbone.")
    parser.add_argument("--head_height", type=float, default=1.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    finetune(parser.parse_args())
