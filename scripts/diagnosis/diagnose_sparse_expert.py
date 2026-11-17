"""
diagnose_sparse_expert.py

Post-hoc diagnostics for the Stage 2 SPARSE expert, run on a saved checkpoint over (a) the fixed
validation cache and (b) a sample of TRAINING patches. No training happens. It answers:

  1. Is the sparse MSE actually better than trivial predictors (zero / per-patch mean)?     -> "skill"
  2. What localisation error does that MSE correspond to?  (GT people displaced by N(0, s^2) px)
  3. Is the output diffuse (background leakage), amplitude-miscalibrated, or dead (ReLU zeros)?
  4. At which spatial scale does the sparse output become accurate? (multi-scale cell masses)
  5. Is the NMS peak metric just miscalibrated?  (threshold-multiplier x kernel sweep, matched F1)
  6. Which scenes / how much of each patch is really "sparse"?

Do NOT run it while training is running (RAM/GPU contention).

Usage (from the repo root):
    uv run src/scripts/diagnosis/diagnose_sparse_expert.py --checkpoint src/multiscene_best_stage_2.pth
Outputs: console report + src/diagnostics/sparse_diag_<tag>.json (+ PNG panels if matplotlib is installed).
"""

import os
import sys
import json
import math
import argparse
import datetime
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))


# ----------------------------------------------------------------------------------------------
# Pure-numpy helpers (no torch needed)
# ----------------------------------------------------------------------------------------------
class Stats:
    """Streaming statistics of prediction P vs target T (float arrays of identical shape)."""

    def __init__(self):
        self.n = 0
        self.sT = self.sT2 = self.sE2 = self.sAE = self.sP = self.sTP = self.sP2 = 0.0

    def add(self, P, T):
        P = np.asarray(P, dtype=np.float64).ravel()
        T = np.asarray(T, dtype=np.float64).ravel()
        E = P - T
        self.n += T.size
        self.sT += T.sum()
        self.sT2 += (T * T).sum()
        self.sE2 += (E * E).sum()
        self.sAE += np.abs(E).sum()
        self.sP += P.sum()
        self.sTP += (T * P).sum()
        self.sP2 += (P * P).sum()

    def mse(self):
        return self.sE2 / max(self.n, 1)

    def zero_mse(self):
        return self.sT2 / max(self.n, 1)

    def skill_zero(self):
        return 1.0 - self.sE2 / max(self.sT2, 1e-12)

    def skill_mean(self):
        var = self.sT2 - self.sT**2 / max(self.n, 1)
        return 1.0 - self.sE2 / max(var, 1e-12)

    def slope(self):  # least-squares amplitude a in T ~ a * P
        return self.sTP / max(self.sP2, 1e-12)

    def mass_ratio(self):  # sum(P) / sum(T)
        return self.sP / max(self.sT, 1e-12)

    def nae(self):  # sum|P-T| / sum(T)
        return self.sAE / max(self.sT, 1e-12)

    def as_dict(self):
        return {
            "mse": self.mse(),
            "zero_mse": self.zero_mse(),
            "skill_vs_zero": self.skill_zero(),
            "skill_vs_mean": self.skill_mean(),
            "amp_slope": self.slope(),
            "mass_ratio": self.mass_ratio(),
            "nae": self.nae(),
            "n": int(self.n),
        }


def count_matches(pred_xy, gt_xy, radius):
    """Number of one-to-one matches (Hungarian if scipy is available, else greedy) within `radius` px."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return 0
    d = np.linalg.norm(pred_xy[:, None, :] - gt_xy[None, :, :], axis=2)
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(d)
        return int((d[r, c] <= radius).sum())
    except ImportError:
        used_p, used_g, tp = set(), set(), 0
        for flat in np.argsort(d, axis=None):
            i, j = divmod(int(flat), d.shape[1])
            if d[i, j] > radius:
                break
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            tp += 1
        return tp


def equivalent_jitter(jitter_curve, model_mse):
    """jitter_curve: list of (s_px, mse). Linear interpolation of the s where mse == model_mse (None if outside)."""
    pts = sorted(jitter_curve)
    for (s0, m0), (s1, m1) in zip(pts[:-1], pts[1:]):
        if m0 <= model_mse <= m1 and m1 > m0:
            return s0 + (s1 - s0) * (model_mse - m0) / (m1 - m0)
    return None


# ----------------------------------------------------------------------------------------------
# Main (needs torch + the training module)
# ----------------------------------------------------------------------------------------------
def main():
    import torch
    import torch.nn.functional as F
    import train_proposed_multiscene as T

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(T.src_dir, "multiscene_best_stage_2.pth"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--val_seed", type=int, default=1234, help="must match the training run to reuse its exact val patches"
    )
    parser.add_argument("--val_frames_per_scene", type=int, default=10)
    parser.add_argument("--train_frames_per_scene", type=int, default=2)
    parser.add_argument(
        "--match_px", type=float, default=5.0, help="NMS-vs-GT matching radius in BEV pixels (5 px = 1 m)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=os.path.join(T.src_dir, "diagnostics"))
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument(
        "--sparse_arch",
        choices=["shallow", "deep"],
        default="shallow",
        help="must match the checkpoint's sparse expert",
    )
    parser.add_argument(
        "--sparse_activation",
        choices=["relu", "sigmoid"],
        default="relu",
        help="use 'sigmoid' for checkpoints trained with --sparse_loss focal",
    )
    args = parser.parse_args()
    T.SPARSE_ACTIVATION = args.sparse_activation
    T.SPARSE_ARCH = args.sparse_arch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json"), "r") as f:
        cams_cfg = json.load(f)
    print("Building registries (this loads all scenes; ~30 s)...")
    train_reg = T.build_scene_registry("train", args.k, cams_cfg, {}, validate_setup=False)
    val_reg = T.build_scene_registry("validate", args.k, cams_cfg, {}, validate_setup=False)
    val_samples, _ = T.build_cached_val_samples(val_reg, args.val_frames_per_scene, val_seed=args.val_seed)
    train_samples, _ = T.build_cached_val_samples(train_reg, args.train_frames_per_scene, val_seed=args.val_seed + 1)

    model = T.ProposedModel(num_cam=args.k).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded {args.checkpoint} (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()
    x0 = val_samples[0]["images"].to(device)
    b0, n0, c0, h0, w0 = x0.shape
    with torch.no_grad():
        f0 = model.backbone(x0.view(b0 * n0, c0, h0, w0))
    print(f"Backbone: input {h0}x{w0} -> feature map {tuple(f0.shape)} (stride ~{h0 / f0.shape[2]:.1f} px)")

    map_kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(device)
    pad = map_kernel.shape[-1] // 2
    single_peak = T.GAUSSIAN_SCALE / (2 * math.pi * T.GAUSSIAN_SIGMA**2)
    H, W = T.BEV_SHAPE
    SCALES = [1, 2, 4, 8, 16, 32]
    JITTERS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    NMS_MULTS = [0.5, 1, 2, 3, 5, 10]
    NMS_KERNELS = [3, 5, 7, 9]

    def gt_maps(sample, p):
        wld = sample["wld_map_paras"].to(device)
        hw_offset = sample["hw_random"].to(device)[:, :, p]
        gxy = sample["gt_world_xy"]
        dots = T.extract_target_coordinates(gxy, wld, hw_offset, bev_shape=T.BEV_SHAPE).to(device)
        tpatch = T.create_ground_truth_heatmap(gxy, wld, hw_offset, bev_shape=T.BEV_SHAPE).to(device)
        conv = F.conv2d(tpatch, map_kernel, padding=pad)
        return dots, tpatch, conv, T.oracle_mask_from_conv(conv)

    def jitter_sse(tpatch, tm, gm, s):
        ys, xs = tpatch[0, 0].nonzero(as_tuple=True)
        cnt = tpatch[0, 0][ys, xs].long()
        ys, xs = ys.repeat_interleave(cnt), xs.repeat_interleave(cnt)
        ys = (ys.float() + torch.randn(ys.shape, device=ys.device) * s).round().long().clamp(0, H - 1)
        xs = (xs.float() + torch.randn(xs.shape, device=xs.device) * s).round().long().clamp(0, W - 1)
        new = torch.zeros_like(tpatch)
        new[0, 0].index_put_((ys, xs), torch.ones(ys.shape, device=ys.device, dtype=new.dtype), accumulate=True)
        jm = F.conv2d(new, map_kernel, padding=pad) * (1.0 - gm)
        return float(((jm - tm) ** 2).sum())

    def analyse(samples, split):
        overall = Stats()
        per_scene = {}
        per_scale = {s: Stats() for s in SCALES}
        jit_sse = {s: 0.0 for s in JITTERS}
        jit_n = 0
        nms = {(m, k): [0, 0, 0] for m in NMS_MULTS for k in NMS_KERNELS}  # tp, n_pred, n_gt
        c = dict(region=0, dead=0, bg_n=0, bg_P=0.0, fg_n=0, fg_missed=0, fg_P=0.0, fg_T=0.0, patches=0, dense_frac=0.0)
        scene_extra = {}
        worst = []  # (mse, scene, T, P)
        per_weather = {}
        patch_rows = []  # (scene, weather, sumP, sumT)
        for i, sample in enumerate(samples):
            scene = sample["scene_name"]
            weather = str(sample["weather"][0]) if "weather" in sample else "UNKNOWN"
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds, _ = model(
                    sample["images"].to(device),
                    sample["ref_points_full"].to(device),
                    sample["hw_random"].to(device),
                    stage=2,
                )
            preds = preds.float()
            for p in range(preds.shape[1]):
                dots, tpatch, conv, gm = gt_maps(sample, p)
                inv = 1.0 - gm
                Pm = preds[:, p, 1:2] * inv  # masked sparse prediction (same as training)
                Tm = conv * inv  # masked sparse target
                P_np, T_np = Pm[0, 0].cpu().numpy(), Tm[0, 0].cpu().numpy()
                overall.add(P_np, T_np)
                per_scene.setdefault(scene, Stats()).add(P_np, T_np)
                per_weather.setdefault(weather, Stats()).add(P_np, T_np)
                patch_rows.append((scene, weather, float(P_np.sum()), float(T_np.sum())))
                for s in SCALES:
                    Pc = F.avg_pool2d(Pm, s, s) * (s * s)
                    Tc = F.avg_pool2d(Tm, s, s) * (s * s)
                    per_scale[s].add(Pc[0, 0].cpu().numpy(), Tc[0, 0].cpu().numpy())
                # region statistics
                R = inv[0, 0].cpu().numpy() > 0.5
                c["patches"] += 1
                c["dense_frac"] += float(gm.mean())
                c["region"] += int(R.sum())
                c["dead"] += int(((P_np == 0) & R).sum())
                bg = R & (T_np < 0.05)
                c["bg_n"] += int(bg.sum())
                c["bg_P"] += float(P_np[bg].sum())
                fg = R & (T_np > 0.5)
                c["fg_n"] += int(fg.sum())
                c["fg_missed"] += int((P_np[fg] < 0.05).sum())
                c["fg_P"] += float(P_np[fg].sum())
                c["fg_T"] += float(T_np[fg].sum())
                # GT points inside the sparse region
                gt_xy = np.zeros((0, 2))
                if dots.shape[2] > 0:
                    d2 = dots[0, 0].cpu().numpy() * np.array([W, H])
                    cx = np.clip(d2[:, 0].astype(int), 0, W - 1)
                    cy = np.clip(d2[:, 1].astype(int), 0, H - 1)
                    gt_xy = d2[R[cy, cx]]
                ex = scene_extra.setdefault(scene, dict(dense_frac=0.0, sparse_gt=0, patches=0))
                ex["dense_frac"] += float(gm.mean())
                ex["sparse_gt"] += len(gt_xy)
                ex["patches"] += 1
                # NMS sweep
                for m, k in nms:
                    peaks = T.extract_peaks_via_nms(
                        Pm, threshold=single_peak * T.PEAK_THRESHOLD_FRACTION * m, kernel_size=k
                    )
                    pred_xy = peaks.cpu().numpy() * np.array([W, H]) if peaks.shape[0] else np.zeros((0, 2))
                    nms[(m, k)][0] += count_matches(pred_xy, gt_xy, args.match_px)
                    nms[(m, k)][1] += len(pred_xy)
                    nms[(m, k)][2] += len(gt_xy)
                # GT-only jitter baseline
                for s in JITTERS:
                    jit_sse[s] += jitter_sse(tpatch, Tm, gm, s)
                jit_n += T_np.size
                sse = float(((P_np - T_np) ** 2).sum())
                worst.append((sse, scene, T_np.copy(), P_np.copy()))
                worst = sorted(worst, key=lambda x: -x[0])[:3]
        # ---- report
        m = overall.as_dict()
        curve = [(s, jit_sse[s] / jit_n) for s in JITTERS]
        s_eq = equivalent_jitter(curve, m["mse"])
        print("\n" + "=" * 78)
        print(f"SPARSE EXPERT DIAGNOSTICS -- {split} ({c['patches']} patches, {len(per_scene)} scenes)")
        print("=" * 78)
        print(
            f"MSE {m['mse']:.4f} | zero-predictor {m['zero_mse']:.4f} -> skill vs zero {100 * m['skill_vs_zero']:.0f}% "
            f"| skill vs per-set mean {100 * m['skill_vs_mean']:.0f}%"
        )
        print(
            f"amplitude slope (T ~ a*P): a = {m['amp_slope']:.2f} (1.0 = calibrated) | sum(P)/sum(T) = {m['mass_ratio']:.2f} "
            f"| NAE of raw maps {100 * m['nae']:.0f}%"
        )
        reg = max(c["region"], 1)
        print(
            f"sparse region = {100 * (1 - c['dense_frac'] / c['patches']):.0f}% of pixels | dead outputs (P==0) in region: "
            f"{100 * c['dead'] / reg:.0f}%"
        )
        print(
            f"background leakage: mean P where T<0.05 = {c['bg_P'] / max(c['bg_n'], 1):.4f} | "
            f"missed bump pixels (T>0.5, P<0.05): {100 * c['fg_missed'] / max(c['fg_n'], 1):.0f}% | "
            f"P/T mass at bump pixels: {c['fg_P'] / max(c['fg_T'], 1e-12):.2f}"
        )
        print("\nGT-only jitter baseline (each person displaced by N(0, s^2), correct count) -> MSE:")
        for s, v in curve:
            print(f"   s = {s:>4.1f} px ({s * 0.2:.1f} m): MSE {v:.4f}")
        if s_eq is None:
            print(
                f"   model MSE {m['mse']:.4f} lies outside this curve (>{curve[-1][1]:.4f} = worse than 8 px jitter, or below the smallest)"
            )
        else:
            print(
                f"   => model MSE equals ~{s_eq:.1f} px ({s_eq * 0.2:.1f} m) of pure localisation noise "
                f"(a LOWER bound on its error: misses / false detections add to it)"
            )
        print("\nMulti-scale agreement inside the sparse region (cell masses; 5 px = 1 m):")
        print(f"   {'cell px':>8}{'cell m':>8}{'skill vs zero':>15}{'NAE':>8}{'sum P / sum T':>15}")
        for s in SCALES:
            d = per_scale[s].as_dict()
            print(
                f"   {s:>8}{s * 0.2:>8.1f}{100 * d['skill_vs_zero']:>14.0f}%{100 * d['nae']:>7.0f}%{d['mass_ratio']:>15.2f}"
            )
        print(f"\nNMS sweep (matching radius {args.match_px:g} px; GT = people in the sparse region):")
        rows = []
        for (mm, k), (tp, npred, ngt) in nms.items():
            prec, rec = tp / max(npred, 1), tp / max(ngt, 1)
            rows.append((2 * prec * rec / max(prec + rec, 1e-12), mm, k, npred, ngt, prec, rec))
        rows.sort(reverse=True)
        print(
            f"   {'mult':>5}{'kernel':>7}{'pred/GT':>10}{'precision':>11}{'recall':>8}{'F1':>7}   (top 6 of {len(rows)})"
        )
        for f1, mm, k, npred, ngt, prec, rec in rows[:6]:
            print(f"   {mm:>5}{k:>7}{npred / max(ngt, 1):>10.2f}{prec:>11.2f}{rec:>8.2f}{f1:>7.2f}")
        cur = [r for r in rows if r[1] == T.SPARSE_PEAK_THRESHOLD_MULTIPLIER and r[2] == T.SPARSE_NMS_KERNEL_SIZE]
        if cur:
            f1, mm, k, npred, ngt, prec, rec = cur[0]
            print(
                f"   current training setting (mult {mm}, kernel {k}): pred/GT {npred / max(ngt, 1):.2f}, F1 {f1:.2f}"
            )
        print("\nPer scene:")
        print(f"   {'scene':<10}{'dense%':>8}{'sparse GT/patch':>17}{'MSE':>9}{'skill0':>8}{'sumP/sumT':>11}{'NAE':>6}")
        for scene, st in sorted(per_scene.items(), key=lambda kv: kv[1].skill_zero()):
            ex = scene_extra[scene]
            print(
                f"   {scene:<10}{100 * ex['dense_frac'] / ex['patches']:>7.0f}%{ex['sparse_gt'] / ex['patches']:>17.1f}"
                f"{st.mse():>9.4f}{100 * st.skill_zero():>7.0f}%{st.mass_ratio():>11.2f}{100 * st.nae():>5.0f}%"
            )
        # per-patch on/off behaviour
        KMP = float(map_kernel.sum())
        ratios = np.array(
            [sp / st for (_, _, sp, st) in patch_rows if st > KMP]
        )  # patches with >= 1 sparse-region person
        if len(ratios):
            q = np.percentile(ratios, [10, 25, 50, 75, 90])
            print(
                f"\nPer-patch sum(P)/sum(T) ({len(ratios)} patches with sparse-region people): "
                f"p10 {q[0]:.2f} | p25 {q[1]:.2f} | median {q[2]:.2f} | p75 {q[3]:.2f} | p90 {q[4]:.2f} | "
                f"<0.3: {100 * (ratios < 0.3).mean():.0f}% | >2: {100 * (ratios > 2).mean():.0f}%"
            )
        print("\nPer weather (patches pooled over scenes; small groups are noisy):")
        print(f"   {'weather':<12}{'patches':>8}{'MSE':>9}{'skill0':>8}{'sumP/sumT':>11}{'share ratio<0.3':>17}")
        by_w = {}
        for _, w, sp, st in patch_rows:
            by_w.setdefault(w, []).append(sp / st if st > KMP else np.nan)
        for w, st_ in sorted(per_weather.items(), key=lambda kv: -kv[1].n):
            r_ = np.array(by_w[w])
            r_ = r_[~np.isnan(r_)]
            print(
                f"   {w:<12}{st_.n // (H * W):>8}{st_.mse():>9.4f}{100 * st_.skill_zero():>7.0f}%{st_.mass_ratio():>11.2f}"
                f"{(100 * (r_ < 0.3).mean() if len(r_) else float('nan')):>16.0f}%"
            )
        result = {
            "overall": m,
            "jitter_curve": curve,
            "per_weather": {w: st_.as_dict() for w, st_ in per_weather.items()},
            "patch_mass_ratio_quantiles": (
                np.percentile(ratios, [10, 25, 50, 75, 90]).tolist() if len(ratios) else None
            ),
            "equivalent_jitter_px": s_eq,
            "scales": {s: per_scale[s].as_dict() for s in SCALES},
            "nms": [
                dict(f1=r[0], mult=r[1], kernel=r[2], pred=r[3], gt=r[4], precision=r[5], recall=r[6]) for r in rows
            ],
            "per_scene": {sc: st.as_dict() for sc, st in per_scene.items()},
            "region": {k_: (float(v) if not isinstance(v, int) else v) for k_, v in c.items()},
        }
        if not args.no_plots:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                for j, (sse, scene, Tn, Pn) in enumerate(worst):
                    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
                    vmax = max(Tn.max(), Pn.max(), 1e-6)
                    for a, img, ttl in zip(
                        ax, (Tn, Pn, Pn - Tn), ("sparse target", "sparse prediction (masked)", "pred - target")
                    ):
                        im = a.imshow(
                            img,
                            cmap="magma" if "diff" not in ttl and "-" not in ttl else "coolwarm",
                            vmin=None if "-" in ttl else 0,
                            vmax=None if "-" in ttl else vmax,
                        )
                        a.set_title(ttl)
                        plt.colorbar(im, ax=a, fraction=0.046)
                    fig.suptitle(f"{split} | {scene} | patch SSE {sse:.1f}")
                    out = os.path.join(args.out_dir, f"sparse_panel_{split}_{j}_{tag}.png")
                    fig.savefig(out, dpi=110, bbox_inches="tight")
                    plt.close(fig)
                print(f"\nSaved {len(worst)} worst-patch panels to {args.out_dir}")
            except ImportError:
                print("(matplotlib not installed -- panels skipped)")
        return result

    results = {
        "checkpoint": args.checkpoint,
        "val_seed": args.val_seed,
        "val": analyse(val_samples, "val"),
        "train": analyse(train_samples, "train"),
    }
    out_json = os.path.join(args.out_dir, f"sparse_diag_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
