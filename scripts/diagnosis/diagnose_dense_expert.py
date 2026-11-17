"""
diagnose_dense_expert.py

Question: how much does the DENSE expert actually learn, versus how much of its in-mask accuracy comes for free from
the ORACLE mask?

Background. The suppression-sweep control arm shows the dense expert outputs ~1,500 persons/patch (5.7x the true total)
OUTSIDE the oracle mask, i.e. a mean of ~1.3 density units per pixel there -- about the same as the mean TARGET density
inside the mask (~1.4). Inside the mask the target is > ~1.27 by construction (the mask is a density threshold), so a
near-constant output already gives a small in-mask error; the mask itself encodes the area. This script tests that:

  1. in-mask MASS error of the expert vs two MASK-ONLY baselines that never look at the image:
       constant density  : c * mask_area                 (c = mean in-mask target density)
       linear in area    : a * mask_area + b             (least squares)
     (both fitted IN-SAMPLE on the val patches, which flatters the baselines -> a conservative test)
  2. pixel-level R^2 of the expert inside the mask (share of within-mask target variance it explains)
  3. expert output statistics inside vs outside the mask (mean, std, coefficient of variation): a small CV outside means
     a flat plateau, not a blob.

If the expert barely beats the mask-only baselines, the oracle-routed hybrid's accuracy is partly ground-truth information
carried by the mask, and the "oracle" numbers overstate what the experts learned.

Do NOT run while training is running.
Usage (repo root):
    uv run src/scripts/diagnosis/diagnose_dense_expert.py --checkpoint src/multiscene_checkpoint_stage_2_p0_epoch_10.pth
"""

import os
import sys
import json
import math
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))


def summarize_dense(areas, t_in, d_in, pix):
    """areas/t_in/d_in: per-patch mask area (px), target mass and expert mass inside the mask (persons).
    pix: dict with sums over in-mask pixels (n, sT, sT2, sE2, sD) and outside pixels (n_out, s_out, s_out2)."""
    areas, t_in, d_in = (np.asarray(x, dtype=np.float64) for x in (areas, t_in, d_in))
    out = {}
    out["mean_target_mass"] = float(t_in.mean())
    out["expert_mae"] = float(np.abs(d_in - t_in).mean())
    out["expert_bias"] = float((d_in - t_in).mean())
    # mask-only baselines (in-sample)
    c_per_px = float(t_in.sum() / max(areas.sum(), 1e-9))  # persons per mask pixel
    base_const = c_per_px * areas
    out["const_density_mae"] = float(np.abs(base_const - t_in).mean())
    A = np.stack([areas, np.ones_like(areas)], axis=1)
    coef, *_ = np.linalg.lstsq(A, t_in, rcond=None)
    out["linear_area_mae"] = float(np.abs(A @ coef - t_in).mean())
    out["zero_mae"] = float(np.abs(t_in).mean())
    out["expert_vs_mass_corr"] = float(np.corrcoef(d_in, t_in)[0, 1]) if len(t_in) > 2 else float("nan")
    out["area_vs_mass_corr"] = float(np.corrcoef(areas, t_in)[0, 1]) if len(t_in) > 2 else float("nan")
    # pixel level
    n, sT, sT2, sE2 = pix["n"], pix["sT"], pix["sT2"], pix["sE2"]
    var_T = sT2 - sT**2 / max(n, 1)
    out["pixel_r2_in_mask"] = float(1 - sE2 / max(var_T, 1e-12))
    out["mean_target_in_mask"] = float(sT / max(n, 1))
    out["mean_expert_in_mask"] = float(pix["sD"] / max(n, 1))
    m_out = pix["s_out"] / max(pix["n_out"], 1)
    out["mean_expert_outside"] = float(m_out)
    out["std_expert_outside"] = float(math.sqrt(max(pix["s_out2"] / max(pix["n_out"], 1) - m_out**2, 0.0)))
    out["cv_outside"] = out["std_expert_outside"] / max(m_out, 1e-9)
    return out


def main():
    import torch
    import torch.nn.functional as F
    import train_proposed_multiscene as T

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=os.path.join(T.src_dir, "multiscene_checkpoint_stage_2_p0_epoch_10.pth")
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--val_seed", type=int, default=1234)
    parser.add_argument("--val_frames_per_scene", type=int, default=10)
    parser.add_argument("--sparse_arch", choices=["shallow", "deep"], default="shallow")
    parser.add_argument("--out_dir", default=os.path.join(T.src_dir, "diagnostics"))
    args = parser.parse_args()
    T.SPARSE_ARCH = args.sparse_arch
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json"), "r") as f:
        cams_cfg = json.load(f)
    val_reg = T.build_scene_registry("validate", args.k, cams_cfg, {}, validate_setup=False)
    val_samples, _ = T.build_cached_val_samples(val_reg, args.val_frames_per_scene, val_seed=args.val_seed)

    model = T.ProposedModel(num_cam=args.k).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
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

    areas, t_in, d_in = [], [], []
    pix = dict(n=0.0, sT=0.0, sT2=0.0, sE2=0.0, sD=0.0, n_out=0.0, s_out=0.0, s_out2=0.0)
    for sample in val_samples:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            proj, _ = model(
                sample["images"].to(device),
                sample["ref_points_full"].to(device),
                sample["hw_random"].to(device),
                stage=3,
            )
        proj = proj.float()
        for p in range(proj.shape[1]):
            conv, gm = oracle_mask(sample, p)
            with torch.no_grad():
                D = torch.relu(model.head.dense_expert(proj[:, p])).float()
            inside = gm[0, 0] > 0.5
            areas.append(float(inside.sum()))
            t_in.append(float((conv * gm).sum()) / KMP)
            d_in.append(float((D * gm).sum()) / KMP)
            if inside.any():
                Tp, Dp = conv[0, 0][inside].double(), D[0, 0][inside].double()
                pix["n"] += Tp.numel()
                pix["sT"] += float(Tp.sum())
                pix["sT2"] += float((Tp * Tp).sum())
                pix["sE2"] += float(((Dp - Tp) ** 2).sum())
                pix["sD"] += float(Dp.sum())
            Do = D[0, 0][~inside].double()
            pix["n_out"] += Do.numel()
            pix["s_out"] += float(Do.sum())
            pix["s_out2"] += float((Do * Do).sum())

    r = summarize_dense(areas, t_in, d_in, pix)
    print("\n" + "=" * 78)
    print(f"DENSE EXPERT DIAGNOSTICS -- val ({len(areas)} patches), {os.path.basename(args.checkpoint)}")
    print("=" * 78)
    print(f"In-mask MASS error (persons/patch; mean target mass in mask {r['mean_target_mass']:.1f}):")
    print(
        f"   dense expert            MAE {r['expert_mae']:6.1f}   (bias {r['expert_bias']:+.1f}, corr with target {r['expert_vs_mass_corr']:.2f})"
    )
    print(
        f"   constant density x area MAE {r['const_density_mae']:6.1f}   (mask-only, never sees the image; in-sample fit)"
    )
    print(
        f"   linear in area          MAE {r['linear_area_mae']:6.1f}   (mask-only; corr(area, target mass) = {r['area_vs_mass_corr']:.2f})"
    )
    print(f"   predict zero            MAE {r['zero_mae']:6.1f}")
    print(
        f"Pixel level inside the mask: R^2 = {r['pixel_r2_in_mask']:.2f} | mean target {r['mean_target_in_mask']:.2f} | mean expert {r['mean_expert_in_mask']:.2f}"
    )
    print(
        f"Outside the mask: mean expert output {r['mean_expert_outside']:.2f} (std {r['std_expert_outside']:.2f}, CV {r['cv_outside']:.2f}) "
        f"= {100 * r['mean_expert_outside'] / max(r['mean_target_in_mask'], 1e-9):.0f}% of the mean in-mask TARGET level"
    )
    print(
        "\nReading: if the expert's MAE is close to the mask-only baselines and the pixel R^2 is low, the expert mostly outputs a\n"
        "near-constant density and its oracle-routed accuracy comes from the mask's area; a low CV outside means a flat plateau."
    )
    out = os.path.join(args.out_dir, "dense_expert_diag.json")
    with open(out, "w") as f:
        json.dump(r, f, indent=1)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
