"""
constant_dense_test.py

Zero-training test of one question: does the TRAINED dense expert add anything over a spatially CONSTANT density level?

Evidence so far: the dense expert outputs a near-constant plateau (about 1.3 density units/px outside the mask), the mask area predicts
the in-mask mass with correlation 1.00, and every attempt to suppress the plateau traded in-mask fit for leak (constant-output formula).
If a constant c in place of the dense expert scores the same as the trained expert, the dense head can be replaced by a constant level,
and the "mass outside the mask" problem disappears by construction (outside the true mask the dense contribution is then c times the
router's false-positive mask mass, which the fused loss trains directly).

For each checkpoint (gating_net BatchNorm statistics are re-estimated first, as in bn_recalibration_test.py, so the router is measured
at its honest level), on the fixed val patches, the fused map is evaluated with

    learned routing:  m * D + (1 - m) * S          with D = trained dense expert   or   D = c (constant), for several c
    oracle routing:   gt_mask * D + (1 - gt_mask) * S   likewise

and the mass error vs the true total mass (persons/patch) is reported: MAE, NAE (sum|err| / sum true), bias.

Reading:
  learned/const ~ learned/trained  -> under learned routing the dense expert is replaceable by a constant.
  oracle/const  ~ oracle/trained   -> the oracle-routed accuracy comes from the mask, not from the dense expert.
  c = 0 shows how much the dense contribution matters at all.
Do NOT run while training is running.

Usage (from src):
    uv run scripts/diagnosis/constant_dense_test.py
    uv run scripts/diagnosis/constant_dense_test.py --checkpoints multiscene_final_stage_4_ro.pth --consts 0,0.6,1.0,1.34,1.39
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


def summarize(errs, true_masses):
    e = np.asarray(errs, dtype=np.float64)
    t = np.asarray(true_masses, dtype=np.float64)
    return {"mae": float(np.abs(e).mean()), "nae": float(np.abs(e).sum() / max(t.sum(), 1e-9)), "bias": float(e.mean())}


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import train_proposed_multiscene as T

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="comma-separated (default: router-only Stage 4 best and final, Stage 3 best)",
    )
    parser.add_argument(
        "--consts", default="0,0.6,1.0,1.34", help="comma-separated constant density levels c (density units per pixel)"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--val_seed", type=int, default=1234)
    parser.add_argument("--val_frames_per_scene", type=int, default=10)
    parser.add_argument("--recal_frames_per_scene", type=int, default=2)
    parser.add_argument("--sparse_arch", choices=["shallow", "deep"], default="shallow")
    parser.add_argument("--out_dir", default=os.path.join(T.src_dir, "diagnostics"))
    args = parser.parse_args()
    T.SPARSE_ARCH = args.sparse_arch
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    consts = [float(c) for c in args.consts.split(",") if c.strip() != ""]
    ckpts = (
        [c.strip() for c in args.checkpoints.split(",")]
        if args.checkpoints
        else [
            os.path.join(T.src_dir, n)
            for n in (
                "multiscene_best_stage_4_ro.pth",
                "multiscene_final_stage_4_ro.pth",
                "multiscene_best_stage_3.pth",
            )
        ]
    )

    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json"), "r") as f:
        cams_cfg = json.load(f)
    print("Building registries (train + val; ~1 min)...")
    train_reg = T.build_scene_registry("train", args.k, cams_cfg, {}, validate_setup=False)
    val_reg = T.build_scene_registry("validate", args.k, cams_cfg, {}, validate_setup=False)
    val_samples, _ = T.build_cached_val_samples(val_reg, args.val_frames_per_scene, val_seed=args.val_seed)
    recal_samples, _ = T.build_cached_val_samples(train_reg, args.recal_frames_per_scene, val_seed=args.val_seed + 7)
    print(f"{len(val_samples)} val samples, {len(recal_samples)} recalibration samples")

    map_kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(device)
    KMP = float(map_kernel.sum())
    pad = map_kernel.shape[-1] // 2

    # ground-truth quantities do not depend on the checkpoint: compute once
    gm_cache, tm_cache = {}, {}
    for i, s in enumerate(val_samples):
        for p in range(s["hw_random"].shape[-1]):
            wld = s["wld_map_paras"].to(device)
            hw_offset = s["hw_random"].to(device)[:, :, p]
            tpatch = T.create_ground_truth_heatmap(s["gt_world_xy"], wld, hw_offset, bev_shape=T.BEV_SHAPE).to(device)
            conv = F.conv2d(tpatch, map_kernel, padding=pad)
            gm_cache[(i, p)] = T.oracle_mask_from_conv(conv)
            tm_cache[(i, p)] = float(conv.sum()) / KMP
    all_true = list(tm_cache.values())
    print(
        f"mean true mass per patch {np.mean(all_true):.1f}; mean oracle-mask area {np.mean([float(g.sum()) for g in gm_cache.values()]):.0f} px"
    )

    def autocast():
        return torch.amp.autocast("cuda", enabled=(device.type == "cuda"))

    results = {}
    for path in ckpts:
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        label = os.path.basename(path)
        model = T.ProposedModel(num_cam=args.k).to(device)
        ck = torch.load(path, map_location=device)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.eval()
        bns = [m for m in model.head.gating_net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        for m in bns:
            m.reset_running_stats()
            m.momentum = None
        model.eval()
        model.head.gating_net.train()
        with torch.no_grad(), autocast():
            for s in recal_samples:
                model(s["images"].to(device), s["ref_points_full"].to(device), s["hw_random"].to(device), stage=3)
        model.eval()

        names = (
            ["learned / trained dense"]
            + [f"learned / const {c:g}" for c in consts]
            + ["oracle  / trained dense"]
            + [f"oracle  / const {c:g}" for c in consts]
        )
        errs = {n: [] for n in names}
        trues = []
        m_mean = []
        for i, s in enumerate(val_samples):
            with torch.no_grad(), autocast():
                proj, masks = model(
                    s["images"].to(device), s["ref_points_full"].to(device), s["hw_random"].to(device), stage=3
                )
            proj, masks = proj.float(), masks.float()
            for p in range(proj.shape[1]):
                with torch.no_grad():
                    D = torch.relu(model.head.dense_expert(proj[:, p])).float()
                    S = torch.relu(model.head.sparse_expert(proj[:, p])).float()
                m, gm, tm = masks[:, p], gm_cache[(i, p)], tm_cache[(i, p)]
                trues.append(tm)
                m_mean.append(float(m.mean()))
                sm_m = float(((1 - m) * S).sum()) / KMP
                sm_g = float(((1 - gm) * S).sum()) / KMP
                errs["learned / trained dense"].append(float((m * D).sum()) / KMP + sm_m - tm)
                errs["oracle  / trained dense"].append(float((gm * D).sum()) / KMP + sm_g - tm)
                for c in consts:
                    errs[f"learned / const {c:g}"].append(c * float(m.sum()) / KMP + sm_m - tm)
                    errs[f"oracle  / const {c:g}"].append(c * float(gm.sum()) / KMP + sm_g - tm)
        res = {n: summarize(errs[n], trues) for n in names}
        results[label] = res
        print("\n" + "=" * 78)
        print(
            f"{label}   (gating BatchNorm re-estimated; mean router mask value {np.mean(m_mean):.3f}; mean true mass {np.mean(trues):.1f})"
        )
        print("=" * 78)
        print(f"{'fused map':<28}{'MAE':>8}{'NAE':>8}{'bias':>9}")
        for n in names:
            r = res[n]
            print(f"{n:<28}{r['mae']:>8.1f}{100 * r['nae']:>7.0f}%{r['bias']:>+9.1f}")
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "constant_dense_test.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nWrote {os.path.join(args.out_dir, 'constant_dense_test.json')}")


if __name__ == "__main__":
    main()
