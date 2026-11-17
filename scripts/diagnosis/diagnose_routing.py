"""
diagnose_routing.py

Post-hoc diagnostics for the LEARNED ROUTER on the fixed validation patches (Stage 3 or Stage 4 checkpoint).
Composition used everywhere (same as the Stage 3 validation code):  fused = m * relu(dense) + (1 - m) * relu(sparse),
where m is the learned soft mask. "Oracle" mask gm = the density-threshold mask built from ground truth.

It answers:
  1. Mask quality: mean learned mask inside / outside the oracle dense region, Dice/IoU at 0.5, coverage.
  2. Mass error of the learned routing vs counterfactuals that isolate WHERE the error comes from:
       learned            m*D + (1-m)*S
       oracle             gm*D + (1-gm)*S                       (upper bound; uses ground truth)
       hard mask          [m>0.5]*D + (1-[m>0.5])*S
       no dense leak      m*D*gm + (1-m)*S                       (dense output zeroed outside the oracle dense region)
       no sparse leak     m*D + (1-m)*S*(1-gm)                   (sparse output zeroed inside the oracle dense region)
       no leaks           m*D*gm + (1-m)*S*(1-gm)
  3. Mass decomposition by region (dense-region vs sparse-region), predicted vs target.
  4. Per scene: learned vs oracle MAE and bias.

Do NOT run while training is running (RAM/GPU contention).
Usage (repo root):
    uv run src/scripts/diagnosis/diagnose_routing.py --checkpoint src/multiscene_best_stage_3.pth
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))


def summarize(errs, targets):
    """errs: signed (pred - target) masses per patch; targets: target masses. Returns MAE, NAE, mean bias."""
    e = np.asarray(errs, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    return float(np.abs(e).mean()), float(np.abs(e).sum() / max(t.sum(), 1e-9)), float(e.mean())


def main():
    import torch
    import torch.nn.functional as F
    import train_proposed_multiscene as T

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(T.src_dir, "multiscene_best_stage_3.pth"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--val_seed", type=int, default=1234)
    parser.add_argument("--val_frames_per_scene", type=int, default=10)
    parser.add_argument("--sparse_arch", choices=["shallow", "deep"], default="shallow")
    parser.add_argument("--out_dir", default=os.path.join(T.src_dir, "diagnostics"))
    args = parser.parse_args()
    T.SPARSE_ARCH = args.sparse_arch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json"), "r") as f:
        cams_cfg = json.load(f)
    val_reg = T.build_scene_registry("validate", args.k, cams_cfg, {}, validate_setup=False)
    val_samples, _ = T.build_cached_val_samples(val_reg, args.val_frames_per_scene, val_seed=args.val_seed)

    model = T.ProposedModel(num_cam=args.k).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded {args.checkpoint} (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()

    map_kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(device)
    KMP = float(map_kernel.sum())
    pad = map_kernel.shape[-1] // 2

    def oracle_mask(sample, p):
        wld = sample["wld_map_paras"].to(device)
        hw_offset = sample["hw_random"].to(device)[:, :, p]
        tpatch = T.create_ground_truth_heatmap(sample["gt_world_xy"], wld, hw_offset, bev_shape=T.BEV_SHAPE).to(device)
        conv = F.conv2d(tpatch, map_kernel, padding=pad)
        return conv, T.oracle_mask_from_conv(conv)

    # 'sharpG' = m**G blended (G>1 pushes ambiguous mid-range mask values toward 0 without any
    # retraining -- tests whether the router's failure is mostly "unconfidently wrong" (mushy mask,
    # fixable by sharpening) vs. genuinely confidently-wrong coverage (not fixable this way).
    SHARPEN_GAMMAS = [2, 4, 8]
    VARIANTS = ["learned", "oracle", "hard mask", "no dense leak", "no sparse leak", "no leaks"] + [
        f"sharp{g}" for g in SHARPEN_GAMMAS
    ]
    errs = {v: [] for v in VARIANTS}
    targets, scenes = [], []
    parts = dict(dd=0.0, ds=0.0, ss=0.0, sd=0.0, t_dense=0.0, t_sparse=0.0, n=0)
    ms = dict(m_in_dense=0.0, n_dense=0, m_in_sparse=0.0, n_sparse=0, tp=0.0, fp=0.0, fn=0.0, cov_pred=0.0, cov_orc=0.0)
    # per-scene mask-quality accumulator, same fields as `ms` above -- the pooled numbers
    # can average away a failure mode that's concentrated on a few (typically high-density) scenes.
    from collections import defaultdict

    ms_scene = defaultdict(
        lambda: dict(
            m_in_dense=0.0,
            n_dense=0,
            m_in_sparse=0.0,
            n_sparse=0,
            tp=0.0,
            fp=0.0,
            fn=0.0,
            cov_pred=0.0,
            cov_orc=0.0,
            n_patch=0,
        )
    )

    for sample in val_samples:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            proj, masks = model(
                sample["images"].to(device),
                sample["ref_points_full"].to(device),
                sample["hw_random"].to(device),
                stage=3,
            )
        proj, masks = proj.float(), masks.float()
        for p in range(proj.shape[1]):
            conv, gm = oracle_mask(sample, p)
            feat, m = proj[:, p], masks[:, p]
            with torch.no_grad():
                D = torch.relu(model.head.dense_expert(feat)).float()
                S = torch.relu(model.head.sparse_expert(feat)).float()
            mh = (m > 0.5).float()
            fused = {
                "learned": m * D + (1 - m) * S,
                "oracle": gm * D + (1 - gm) * S,
                "hard mask": mh * D + (1 - mh) * S,
                "no dense leak": m * D * gm + (1 - m) * S,
                "no sparse leak": m * D + (1 - m) * S * (1 - gm),
                "no leaks": m * D * gm + (1 - m) * S * (1 - gm),
            }
            for g in SHARPEN_GAMMAS:
                ms_ = m**g
                fused[f"sharp{g}"] = ms_ * D + (1 - ms_) * S
            tmass = float(conv.sum()) / KMP
            targets.append(tmass)
            scenes.append(sample["scene_name"])
            for v in VARIANTS:
                errs[v].append(float(fused[v].sum()) / KMP - tmass)
            parts["dd"] += float((m * D * gm).sum()) / KMP
            parts["ds"] += float((m * D * (1 - gm)).sum()) / KMP
            parts["ss"] += float(((1 - m) * S * (1 - gm)).sum()) / KMP
            parts["sd"] += float(((1 - m) * S * gm).sum()) / KMP
            parts["t_dense"] += float((conv * gm).sum()) / KMP
            parts["t_sparse"] += float((conv * (1 - gm)).sum()) / KMP
            parts["n"] += 1
            nd, ns_ = float(gm.sum()), float((1 - gm).sum())
            m_in_d, m_in_s = float((m * gm).sum()), float((m * (1 - gm)).sum())
            tp_, fp_, fn_ = float((mh * gm).sum()), float((mh * (1 - gm)).sum()), float(((1 - mh) * gm).sum())
            cov_p, cov_o = float(mh.mean()), float(gm.mean())
            ms["m_in_dense"] += m_in_d
            ms["n_dense"] += nd
            ms["m_in_sparse"] += m_in_s
            ms["n_sparse"] += ns_
            ms["tp"] += tp_
            ms["fp"] += fp_
            ms["fn"] += fn_
            ms["cov_pred"] += cov_p
            ms["cov_orc"] += cov_o
            sm = ms_scene[sample["scene_name"]]
            sm["m_in_dense"] += m_in_d
            sm["n_dense"] += nd
            sm["m_in_sparse"] += m_in_s
            sm["n_sparse"] += ns_
            sm["tp"] += tp_
            sm["fp"] += fp_
            sm["fn"] += fn_
            sm["cov_pred"] += cov_p
            sm["cov_orc"] += cov_o
            sm["n_patch"] += 1

    n = parts["n"]
    print("\n" + "=" * 78)
    print(f"ROUTING DIAGNOSTICS -- val ({n} patches), checkpoint {os.path.basename(args.checkpoint)}")
    print("=" * 78)
    dice = 2 * ms["tp"] / max(2 * ms["tp"] + ms["fp"] + ms["fn"], 1e-9)
    iou = ms["tp"] / max(ms["tp"] + ms["fp"] + ms["fn"], 1e-9)
    print(
        f"Mask: mean m inside oracle-dense region {ms['m_in_dense'] / max(ms['n_dense'], 1):.2f} (ideal 1.0) | "
        f"outside {ms['m_in_sparse'] / max(ms['n_sparse'], 1):.2f} (ideal 0.0) | Dice@0.5 {dice:.2f} | IoU@0.5 {iou:.2f} | "
        f"predicted coverage {100 * ms['cov_pred'] / n:.1f}% vs oracle {100 * ms['cov_orc'] / n:.1f}%"
    )
    print(
        f"\n{'variant':<16}{'MAE':>8}{'NAE':>8}{'bias':>9}   (per patch, persons; mean target {np.mean(targets):.1f})"
    )
    res = {}
    for v in VARIANTS:
        mae, nae, bias = summarize(errs[v], targets)
        res[v] = dict(mae=mae, nae=nae, bias=bias)
        print(f"{v:<16}{mae:>8.1f}{100 * nae:>7.0f}%{bias:>+9.1f}")
    print("\nMass decomposition of the learned fusion (mean persons per patch):")
    print(
        f"   dense-region  : dense-expert part {parts['dd'] / n:7.1f} + sparse-expert leak {parts['sd'] / n:7.1f} "
        f"= {(parts['dd'] + parts['sd']) / n:7.1f}   vs target {parts['t_dense'] / n:7.1f}"
    )
    print(
        f"   sparse-region : sparse-expert part {parts['ss'] / n:7.1f} + dense-expert leak {parts['ds'] / n:7.1f} "
        f"= {(parts['ss'] + parts['ds']) / n:7.1f}   vs target {parts['t_sparse'] / n:7.1f}"
    )
    # Per-scene mask quality, sorted by scene density (meanGT) so a density correlation with
    # the router's over/under-coverage is visible at a glance rather than needing a separate plot.
    print(
        "\nPer scene, sorted by density (learned MAE/bias, oracle MAE/bias, mask mean-in/out, Dice, IoU, pred vs oracle coverage):"
    )
    print(
        f"   {'scene':<10}{'meanGT':>8}{'MAE lrn':>9}{'bias':>8}{'MAE orc':>9}{'bias':>8}{'m-in':>7}{'m-out':>7}{'Dice':>6}{'IoU':>6}{'cov%':>7}{'orcCov%':>8}"
    )
    sc = np.array(scenes)
    scene_order = sorted(set(scenes), key=lambda s: -np.array(targets)[sc == s].mean())
    scene_mask_rows = {}
    for s in scene_order:
        idx = np.where(sc == s)[0]
        tg = np.array(targets)[idx]
        el, eo = np.array(errs["learned"])[idx], np.array(errs["oracle"])[idx]
        sm = ms_scene[s]
        s_dice = 2 * sm["tp"] / max(2 * sm["tp"] + sm["fp"] + sm["fn"], 1e-9)
        s_iou = sm["tp"] / max(sm["tp"] + sm["fp"] + sm["fn"], 1e-9)
        s_min = sm["m_in_dense"] / max(sm["n_dense"], 1)
        s_mout = sm["m_in_sparse"] / max(sm["n_sparse"], 1)
        s_covp = 100 * sm["cov_pred"] / max(sm["n_patch"], 1)
        s_covo = 100 * sm["cov_orc"] / max(sm["n_patch"], 1)
        scene_mask_rows[s] = dict(dice=s_dice, iou=s_iou, m_in=s_min, m_out=s_mout, cov_pred=s_covp, cov_orc=s_covo)
        print(
            f"   {s:<10}{tg.mean():>8.1f}{np.abs(el).mean():>9.1f}{el.mean():>+8.1f}{np.abs(eo).mean():>9.1f}{eo.mean():>+8.1f}"
            f"{s_min:>7.2f}{s_mout:>7.2f}{s_dice:>6.2f}{s_iou:>6.2f}{s_covp:>7.1f}{s_covo:>8.1f}"
        )

    # Mask-sharpening check (m**gamma, no retraining): does pushing ambiguous mid-range mask
    # values toward 0 recover most of the oracle-vs-learned gap on the worst (highest-density) scenes,
    # or is the over-coverage too confident for sharpening alone to fix?
    sharp_vars = [f"sharp{g}" for g in SHARPEN_GAMMAS]
    print("\nMask-sharpening check (m**gamma blend, no retraining) -- pooled and on the worst scenes:")
    print(f"   {'':<10}{'learned':>9}" + "".join(f"{v:>9}" for v in sharp_vars) + f"{'oracle':>9}   (MAE)")

    def row(label_, idx_):
        vals = (
            [np.abs(np.array(errs["learned"])[idx_]).mean()]
            + [np.abs(np.array(errs[v])[idx_]).mean() for v in sharp_vars]
            + [np.abs(np.array(errs["oracle"])[idx_]).mean()]
        )
        print(f"   {label_:<10}" + "".join(f"{v_:>9.1f}" for v_ in vals))

    row("pooled", np.arange(len(scenes)))
    for s in scene_order[:3]:  # 3 highest-density scenes -- where the over-coverage problem concentrates
        row(s, np.where(sc == s)[0])

    out = os.path.join(args.out_dir, f"routing_diag_{tag}.json")
    with open(out, "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "variants": res,
                "dice": dice,
                "iou": iou,
                "per_scene_mask": scene_mask_rows,
                "parts_per_patch": {k_: v_ / n for k_, v_ in parts.items() if k_ != "n"},
            },
            f,
            indent=1,
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
