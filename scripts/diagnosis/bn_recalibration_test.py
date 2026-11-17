"""
bn_recalibration_test.py

Tests one hypothesis about the learned router: its epoch-to-epoch noise comes from the BatchNorm RUNNING STATISTICS
in gating_net, not from its weights. In Stage 3/4 the router trains in train() mode with one 200x200 patch per
batch, so its running mean/var are an exponential average over roughly the last ten patches -- i.e. over the last
one or two SCENES of the epoch -- and the eval-mode router used for validation inherits that scene-dependent
normalisation. That would shift the mean mask level, and the fused mass error is hypersensitive to it (dense-expert
leakage is weighted by the mask).

For every checkpoint the script evaluates the learned-routing mass error on the fixed val patches
  (a) as saved, and
  (b) after RE-ESTIMATING gating_net's BatchNorm statistics as a cumulative average over training patches from
      ALL scenes (forward passes only; no weight is changed).
If (b) is much lower and much more similar across checkpoints than (a), the noise is a BatchNorm-statistics
artefact and the fix is cheap (freeze/recalibrate BN, or GroupNorm). If (b) ~ (a), the hypothesis is wrong.
If gating_net has no BatchNorm layers, the hypothesis cannot apply and the script says so.

Composition (same as Stage 3 validation):  fused = m*relu(dense) + (1-m)*relu(sparse).
Do NOT run while training is running.

Usage (repo root):
    uv run src/scripts/diagnosis/bn_recalibration_test.py
    uv run src/scripts/diagnosis/bn_recalibration_test.py --checkpoints src/multiscene_best_stage_3.pth,src/multiscene_final_stage_4_ro.pth
"""

import os
import sys
import json
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

DEFAULT_CKPTS = [
    "multiscene_checkpoint_stage_3_epoch_25.pth",  # a BAD Stage 3 epoch in the log (pooled MAE ~210)
    "multiscene_checkpoint_stage_3_epoch_30.pth",  # pooled MAE ~107
    "multiscene_best_stage_3.pth",  # epoch 24, pooled MAE ~87
    "multiscene_best_stage_4_ro.pth",  # router-only Stage 4, epoch 10, ~68.5
    "multiscene_final_stage_4_ro.pth",  # router-only Stage 4, final epoch, ~78.8
]


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import train_proposed_multiscene as T

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints", default=None, help="comma-separated paths (default: five from the Stage 3 / router-only runs)"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--val_seed", type=int, default=1234)
    parser.add_argument("--val_frames_per_scene", type=int, default=10)
    parser.add_argument(
        "--recal_frames_per_scene",
        type=int,
        default=2,
        help="training frames per scene used to re-estimate BN statistics",
    )
    parser.add_argument("--sparse_arch", choices=["shallow", "deep"], default="shallow")
    args = parser.parse_args()
    T.SPARSE_ARCH = args.sparse_arch

    ckpts = (
        [c.strip() for c in args.checkpoints.split(",")]
        if args.checkpoints
        else [os.path.join(T.src_dir, c) for c in DEFAULT_CKPTS]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json"), "r") as f:
        cams_cfg = json.load(f)
    print("Building registries (train + val; ~1 min)...")
    train_reg = T.build_scene_registry("train", args.k, cams_cfg, {}, validate_setup=False)
    val_reg = T.build_scene_registry("validate", args.k, cams_cfg, {}, validate_setup=False)
    val_samples, _ = T.build_cached_val_samples(val_reg, args.val_frames_per_scene, val_seed=args.val_seed)
    recal_samples, _ = T.build_cached_val_samples(train_reg, args.recal_frames_per_scene, val_seed=args.val_seed + 7)
    print(
        f"{len(val_samples)} val samples, {len(recal_samples)} recalibration samples from {len(train_reg)} train scenes"
    )

    map_kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(device)
    KMP = float(map_kernel.sum())
    pad = map_kernel.shape[-1] // 2

    def target_mass(sample, p):
        wld = sample["wld_map_paras"].to(device)
        hw_offset = sample["hw_random"].to(device)[:, :, p]
        tpatch = T.create_ground_truth_heatmap(sample["gt_world_xy"], wld, hw_offset, bev_shape=T.BEV_SHAPE).to(device)
        return float(F.conv2d(tpatch, map_kernel, padding=pad).sum()) / KMP

    # target masses do not depend on the checkpoint: compute once (crops are fixed in the cache)
    targets = [[target_mass(s, p) for p in range(s["hw_random"].shape[-1])] for s in val_samples]

    def eval_learned(model):
        errs = []
        for s, tm in zip(val_samples, targets):
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                proj, masks = model(
                    s["images"].to(device), s["ref_points_full"].to(device), s["hw_random"].to(device), stage=3
                )
            proj, masks = proj.float(), masks.float()
            for p in range(proj.shape[1]):
                with torch.no_grad():
                    D = torch.relu(model.head.dense_expert(proj[:, p])).float()
                    S = torch.relu(model.head.sparse_expert(proj[:, p])).float()
                m = masks[:, p]
                errs.append(float((m * D + (1 - m) * S).sum()) / KMP - tm[p])
        e = np.array(errs)
        t = np.array([x for tm in targets for x in tm])
        return float(np.abs(e).mean()), float(np.abs(e).sum() / t.sum()), float(e.mean())

    rows = []
    for path in ckpts:
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        model = T.ProposedModel(num_cam=args.k).to(device)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        bns = [
            (n, m) for n, m in model.head.gating_net.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
        ]
        if not bns:
            print("gating_net has NO BatchNorm layers -> the BatchNorm hypothesis cannot explain the noise. Stopping.")
            return
        before = eval_learned(model)
        for _, m in bns:
            m.reset_running_stats()
            m.momentum = None  # cumulative average over all recalibration batches
        model.eval()
        model.head.gating_net.train()  # only gating_net's BN layers use/update batch statistics
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            for s in recal_samples:
                model(s["images"].to(device), s["ref_points_full"].to(device), s["hw_random"].to(device), stage=3)
        model.eval()
        after = eval_learned(model)
        rows.append((os.path.basename(path), before, after, len(bns)))
        print(
            f"{os.path.basename(path)}: as saved MAE {before[0]:.1f} (NAE {100 * before[1]:.0f}%, bias {before[2]:+.1f}) | "
            f"BN recalibrated MAE {after[0]:.1f} (NAE {100 * after[1]:.0f}%, bias {after[2]:+.1f})"
        )
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 92)
    print(
        "BATCHNORM RECALIBRATION TEST (learned-routing mass error per patch; mean target "
        f"{np.mean([x for tm in targets for x in tm]):.1f})"
    )
    print("=" * 92)
    print(f"{'checkpoint':<42}{'as saved: MAE / NAE / bias':>28}{'recalibrated: MAE / NAE / bias':>32}")
    for name, b, a, n_bn in rows:
        print(
            f"{name:<42}{b[0]:>9.1f} {100 * b[1]:>5.0f}% {b[2]:>+7.1f}   {a[0]:>12.1f} {100 * a[1]:>5.0f}% {a[2]:>+7.1f}"
        )
    if rows:
        sb = [r[1][0] for r in rows]
        sa = [r[2][0] for r in rows]
        print(
            f"\nspread across checkpoints (max-min MAE): as saved {max(sb) - min(sb):.1f} -> recalibrated {max(sa) - min(sa):.1f}; "
            f"mean MAE {np.mean(sb):.1f} -> {np.mean(sa):.1f}   ({rows[0][3]} BatchNorm layers in gating_net)"
        )


if __name__ == "__main__":
    main()
