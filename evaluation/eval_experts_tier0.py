"""
eval_experts_tier0.py

Tier-0 evaluation of the DENSE / SPARSE experts (and optionally the Stage 1 unified head) of one or more checkpoints on the
fixed validation cache. No training. Reports per scene and pooled: shape (corr, AUROC), background pedestal, regional mass,
oracle-routed count error, router-leak sensitivity, response curve, and PASS/FAIL against pre-declared criteria.

Do NOT run while training is running.
Usage (from src):
    uv run evaluation/eval_experts_tier0.py --checkpoints multiscene_final_stage_2_ring.pth --stage1_checkpoint multiscene_best_stage_1.pth --precise_bn
Use --precise_bn for multi-scene checkpoints (matches training-time validation); do not use it for single-scene checkpoints.
"""

import os
import sys
import json
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))
from visualize_experts import pearson, BinAccumulator, BIN_EDGES

GUIDE = """
READING GUIDE
  T          true density = annotated points convolved with the training kernel (peak 1.0 per isolated person, sum 31.4159 per person).
             mass (persons) = sum(map) / 31.4159. Every prediction is compared against T.
  zone       oracle dense zone (T > 1.273, smoothed): where the dense expert is trained.   empty = pixels with T < {empty}.
  corr       Pearson correlation with T over all pixels of a patch. Invariant to scale AND offset -> measures SHAPE only.
  bg         mean output on empty pixels (the pedestal). Ideal ~ 0.
  auc_crowd  P(random crowd pixel [T >= {crowd}] scores higher than random empty pixel). 0.5 = no information, 1.0 = perfect. Offset-invariant.
  auc_zone   same idea for 'inside the oracle zone' vs 'outside': does the output track the shape of the mask?
  corr_in    correlation with T inside the zone only (does the expert resolve structure there?).
  layout     no-image baseline: mean true density of the OTHER frames of the same scene at the same crop (uses annotations).
  fused      oracle-routed map = zone*dense + (1-zone)*sparse.        unified = Stage 1 head (RQ2 baseline), converted to kernel units.
  count      predicted total persons per patch minus true total (bias = signed mean, MAE = mean absolute).
  leak_out(e) fused count if the router gives the dense expert weight e OUTSIDE the zone; leak_sym(e): also gives sparse weight e inside.
"""


def auroc(score, label):
    score = np.asarray(score).ravel()
    label = np.asarray(label).ravel().astype(bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    uniq, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(cnt) - (cnt - 1) / 2.0
    ranks = avg_rank[inv]
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def nm(vals):
    a = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def fmt(x, d=2, w=7):
    if x is None or x != x:
        return f"{'nan':>{w}}"
    return f"{x:>{w}.{d}f}"


def scene_num(name):
    try:
        return int(name.split("_")[1])
    except Exception:
        return 0


def ek(e):
    return f"{e:g}"


def summarize(rows, eps_list):
    out = {"n_patches": len(rows)}
    for k in [k for k, v in rows[0].items() if isinstance(v, float)]:
        out[k] = nm([r[k] for r in rows])
    for tag in ("fused", "unified", "dense", "sparse"):
        if f"{tag}_tot" in rows[0]:
            err = np.array([r[f"{tag}_tot"] - r["t_tot"] for r in rows])
            out[f"cnt_bias_{tag}"] = float(err.mean())
            out[f"cnt_mae_{tag}"] = float(np.abs(err).mean())
    for mode in ("leak_out", "leak_sym"):
        for e in eps_list:
            err = np.array([r[f"{mode}_{ek(e)}"] for r in rows])
            out[f"{mode}_bias_{ek(e)}"] = float(err.mean())
            out[f"{mode}_mae_{ek(e)}"] = float(np.abs(err).mean())
    return out


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import train_proposed_multiscene as T

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True, help="comma-separated Stage>=2 checkpoints")
    ap.add_argument("--stage1_checkpoint", default=None, help="adds the unified-head baseline")
    ap.add_argument("--val_scenes", default="all", help="comma-separated, 'all' or 'none'")
    ap.add_argument(
        "--train_scenes",
        default="none",
        help="comma-separated, 'all' or 'none' (train scenes are not a generalisation test)",
    )
    ap.add_argument("--frames_per_scene", type=int, default=10)
    ap.add_argument("--val_seed", type=int, default=1234)
    ap.add_argument(
        "--weak_scenes", default="scene_23,scene_38", help="also report pooled results without these scenes"
    )
    ap.add_argument("--precise_bn", action="store_true")
    ap.add_argument("--precise_bn_frames", type=int, default=1)
    ap.add_argument("--no_layout_prior", action="store_true")
    ap.add_argument("--leak_eps", default="0,0.03,0.1")
    ap.add_argument("--empty_thr", type=float, default=0.02)
    ap.add_argument("--crowd_thr", type=float, default=0.1)
    ap.add_argument("--target_bg", type=float, default=0.15, help="criterion 1: pooled dense pedestal <= this")
    ap.add_argument(
        "--min_frac_scenes",
        type=float,
        default=0.7,
        help="criterion 2/3: dense corr must beat unified / layout in at least this fraction of scenes",
    )
    ap.add_argument(
        "--max_leak_ratio",
        type=float,
        default=1.5,
        help="criterion 4: fused count MAE at max eps / MAE at eps 0 must be <= this",
    )
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sparse_arch", choices=["shallow", "deep"], default="shallow")
    ap.add_argument("--allow_missing", action="store_true")
    ap.add_argument("--out_dir", default=os.path.join(T.src_dir, "diagnostics", "tier0"))
    args = ap.parse_args()
    T.SPARSE_ARCH = args.sparse_arch
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps_list = [float(x) for x in args.leak_eps.split(",")]
    weak = {n.strip() for n in args.weak_scenes.split(",") if n.strip()}
    print(GUIDE.format(empty=args.empty_thr, crowd=args.crowd_thr))

    # ---------------------------------------------------------------- samples
    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json"), "r") as f:
        cams_cfg = json.load(f)
    train_all = T.build_scene_registry("train", args.k, cams_cfg, {}, validate_setup=False)
    val_all = T.build_scene_registry("validate", args.k, cams_cfg, {}, validate_setup=False)

    def pick(reg, csv, split):
        raw = [n.strip() for n in csv.split(",") if n.strip()]
        if not raw or raw[0].lower() == "none":
            return []
        if raw[0].lower() == "all":
            return list(reg)
        chosen = [e for e in reg if e["scene_name"] in raw]
        missing = sorted(set(raw) - {e["scene_name"] for e in chosen})
        if missing:
            print(f"[warning] {split} scenes not in the registry: {missing}")
        return chosen

    samples = []
    for sel, split in (
        (pick(train_all, args.train_scenes, "train"), "train"),
        (pick(val_all, args.val_scenes, "val"), "val"),
    ):
        if not sel:
            continue
        ss, _ = T.build_cached_val_samples(sel, args.frames_per_scene, val_seed=args.val_seed)
        for s in ss:
            s["split"] = split
        samples += ss
    if not samples:
        raise SystemExit("no samples selected")
    recal_samples = None
    if args.precise_bn:
        recal_samples, _ = T.build_cached_val_samples(train_all, args.precise_bn_frames, val_seed=args.val_seed + 7)

    kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(device)
    KMP = float(kernel.sum())
    pad = kernel.shape[-1] // 2
    thr = T.oracle_density_threshold()
    print(f"[units] kernel max {float(kernel.max()):.3f}, sum {KMP:.4f}, zone threshold {thr:.4f}")

    # ---------------------------------------------------------------- ground truth (once, reused for every checkpoint)
    def dens(src, wld, hw):
        tp = T.create_ground_truth_heatmap(src["gt_world_xy"], wld, hw, bev_shape=T.BEV_SHAPE).to(device)
        return F.conv2d(tp, kernel, padding=pad)

    gt_cache, n_patch = {}, 0
    for i, s in enumerate(samples):
        wld = s["wld_map_paras"].to(device)
        for p in range(s["hw_random"].shape[-1]):
            hw = s["hw_random"].to(device)[:, :, p]
            conv = dens(s, wld, hw)
            gm = T.oracle_mask_from_conv(conv)
            gt_cache[(i, p)] = (conv[0, 0].cpu().numpy(), gm[0, 0].cpu().numpy())
            n_patch += 1
    mean_true = float(np.mean([v[0].sum() / KMP for v in gt_cache.values()]))
    print(
        f"[data] {len(samples)} samples, {n_patch} patches, mean true mass {mean_true:.1f} persons/patch "
        f"(training val cache with defaults: 300 patches, ~269.1)"
    )
    scene_counts = {}
    for s in samples:
        scene_counts.setdefault((s["split"], s["scene_name"]), []).append(float(s["wld_map_paras"].reshape(-1)[0]))
    for (sp, sc), r0s in sorted(scene_counts.items(), key=lambda kv: (kv[0][0], scene_num(kv[0][1]))):
        print(f"       {sp:<5} {sc:<10} {len(r0s)} frames, r0 {np.mean(r0s):.2f}{'  (weak)' if sc in weak else ''}")

    layout_corr = {}
    if not args.no_layout_prior:
        print("[prior] computing layout-prior baseline (leave-one-frame-out per scene)...")
        by_scene = {}
        for i, s in enumerate(samples):
            by_scene.setdefault((s["split"], s["scene_name"]), []).append(i)
        for idxs in by_scene.values():
            if len(idxs) < 2:
                continue
            for i in idxs:
                wld = samples[i]["wld_map_paras"].to(device)
                for p in range(samples[i]["hw_random"].shape[-1]):
                    hw = samples[i]["hw_random"].to(device)[:, :, p]
                    acc = None
                    for j in idxs:
                        if j != i:
                            d = dens(samples[j], wld, hw)
                            acc = d if acc is None else acc + d
                    prior = (acc / (len(idxs) - 1))[0, 0].cpu().numpy()
                    layout_corr[(i, p)] = pearson(prior, gt_cache[(i, p)][0])

    # ---------------------------------------------------------------- model helpers
    def load_model(path, need):
        from collections import Counter

        ck = torch.load(path, map_location=device, weights_only=False)
        sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
        model = T.build_model_for_checkpoint(ck, num_cam=args.k).to(device)
        # Drop shape-mismatched keys (e.g. gating_net after its contrast branch; unused under oracle routing)
        # instead of crashing: strict=False only tolerates missing/extra keys.
        msd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k not in msd or msd[k].shape == v.shape}
        res = model.load_state_dict(sd, strict=False)

        def grp(ks):
            return dict(Counter(".".join(k.split(".")[:2]) for k in ks))

        print(
            f"[load] {os.path.basename(path)}: missing={len(res.missing_keys)} {grp(res.missing_keys)} | unexpected={len(res.unexpected_keys)} {grp(res.unexpected_keys)}"
        )
        bad = [k for k in res.missing_keys if k.startswith(need)]
        if bad and not args.allow_missing:
            raise RuntimeError(
                f"{len(bad)} required keys missing in {path} (e.g. {bad[:3]}); use --allow_missing to override"
            )
        model.eval()
        return model

    def recalibrate_bn(model, stage):
        mods = [
            m
            for part in (model.backbone.feature_extractor[-1], model.neck)
            for m in part.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)
        ]
        if not mods:
            return
        old = [m.momentum for m in mods]
        for m in mods:
            m.reset_running_stats()
            m.momentum = None
        model.eval()
        for m in mods:
            m.train()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            for s in recal_samples:
                model(s["images"].to(device), s["ref_points_full"].to(device), s["hw_random"].to(device), stage=stage)
        for m, mom in zip(mods, old):
            m.momentum = mom
        model.eval()

    def forward(model, s, stage):
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out, _ = model(
                s["images"].to(device), s["ref_points_full"].to(device), s["hw_random"].to(device), stage=stage
            )
        return out.float()

    unified = None
    if args.stage1_checkpoint:
        m1 = load_model(args.stage1_checkpoint, ("neck.", "backbone."))
        if args.precise_bn:
            recalibrate_bn(m1, 1)
        unified = [(forward(m1, s, 1) * (KMP / T.GAUSSIAN_SCALE))[0, :, 0].cpu().numpy() for s in samples]
        del m1
        torch.cuda.empty_cache()
    has_u = unified is not None

    # ---------------------------------------------------------------- evaluation of one checkpoint
    def evaluate(path):
        label = os.path.splitext(os.path.basename(path))[0]
        model = load_model(path, ("head.dense_expert", "head.sparse_expert", "neck.", "backbone."))
        if args.precise_bn:
            recalibrate_bn(model, 3)
        names = ["dense", "sparse", "fused"] + (["unified"] if has_u else [])
        acc = BinAccumulator(names)
        rows = []
        for i, s in enumerate(samples):
            if i % 20 == 0:
                print(f"[eval] {label}: sample {i + 1}/{len(samples)}")
            proj = forward(model, s, 3)
            for p in range(proj.shape[1]):
                Tn, gm = gt_cache[(i, p)]
                with torch.no_grad():
                    D = torch.relu(model.head.dense_expert(proj[:, p]))[0, 0].float().cpu().numpy()
                    S = torch.relu(model.head.sparse_expert(proj[:, p]))[0, 0].float().cpu().numpy()
                maps = {"dense": D, "sparse": S, "fused": gm * D + (1 - gm) * S}
                if has_u:
                    maps["unified"] = unified[i][p]
                acc.add(Tn, maps)
                inside, empty = gm > 0.5, Tn < args.empty_thr
                crowd = Tn >= args.crowd_thr
                sel = empty | crowd
                t_tot = float(Tn.sum() / KMP)
                row = {
                    "scene": s["scene_name"],
                    "split": s["split"],
                    "snap": int(s["snapshot_id"][0]),
                    "patch": p,
                    "r0": float(s["wld_map_paras"].reshape(-1)[0]),
                    "zone_cov": float(100 * gm.mean()),
                    "t_tot": t_tot,
                    "t_in": float((Tn * gm).sum() / KMP),
                    "t_out": float((Tn * (1 - gm)).sum() / KMP),
                    "layout_corr": float(layout_corr.get((i, p), float("nan"))),
                }
                for nme, M in maps.items():
                    row[f"{nme}_corr"] = float(pearson(M, Tn))
                    row[f"{nme}_bg"] = float(M[empty].mean()) if empty.any() else float("nan")
                    row[f"{nme}_in"] = float((M * gm).sum() / KMP)
                    row[f"{nme}_out"] = float((M * (1 - gm)).sum() / KMP)
                    row[f"{nme}_tot"] = float(M.sum() / KMP)
                    if nme != "fused":
                        row[f"{nme}_auc_crowd"] = (
                            auroc(M[sel], crowd[sel]) if (empty.any() and crowd.any()) else float("nan")
                        )
                        ok_zone = inside.sum() >= 50 and (~inside).sum() >= 50
                        row[f"{nme}_auc_zone"] = auroc(M, inside) if ok_zone else float("nan")
                        row[f"{nme}_corr_in"] = (
                            float(pearson(M[inside], Tn[inside])) if inside.sum() >= 50 else float("nan")
                        )
                for e in eps_list:
                    m1_ = gm * (1 - e) + e
                    m2_ = e + (1 - 2 * e) * gm
                    row[f"leak_out_{ek(e)}"] = float((m1_ * D + (1 - m1_) * S).sum() / KMP - t_tot)
                    row[f"leak_sym_{ek(e)}"] = float((m2_ * D + (1 - m2_) * S).sum() / KMP - t_tot)
                rows.append(row)
        del model
        torch.cuda.empty_cache()
        return label, rows, acc, names

    # ---------------------------------------------------------------- reporting
    def report(label, rows, acc, names):
        by = {}
        for r in rows:
            by.setdefault((r["split"], r["scene"]), []).append(r)
        keys = sorted(by, key=lambda k: (k[0], scene_num(k[1])))
        per = {k: summarize(v, eps_list) for k, v in by.items()}
        bar = "=" * 132
        print(f"\n{bar}\n{label}\n{bar}")

        print("\n[1] SHAPE  (corr / AUC ignore offset; 'layout' = no-image baseline)")
        print(
            f"{'scene':<10}{'sp':<6}{'n':>3}{'r0':>6}{'zone%':>6} | {'layout':>6} | corr: {'D':>5}{'S':>6}{'U':>6}{'F':>6} | auc_crowd: {'D':>5}{'S':>6}{'U':>6} | auc_zone: {'D':>5}{'S':>6}{'U':>6} | corr_in {'D':>5}{'U':>6}"
        )
        for k in keys:
            s = per[k]
            print(
                f"{k[1]:<10}{k[0]:<6}{s['n_patches']:>3}{fmt(s['r0'], 2, 6)}{fmt(s['zone_cov'], 1, 6)} | {fmt(s['layout_corr'], 2, 6)} | "
                f"      {fmt(s['dense_corr'], 2, 5)}{fmt(s['sparse_corr'], 2, 6)}{fmt(s.get('unified_corr'), 2, 6)}{fmt(s['fused_corr'], 2, 6)} | "
                f"           {fmt(s['dense_auc_crowd'], 2, 5)}{fmt(s['sparse_auc_crowd'], 2, 6)}{fmt(s.get('unified_auc_crowd'), 2, 6)} | "
                f"          {fmt(s['dense_auc_zone'], 2, 5)}{fmt(s['sparse_auc_zone'], 2, 6)}{fmt(s.get('unified_auc_zone'), 2, 6)} | "
                f"        {fmt(s['dense_corr_in'], 2, 5)}{fmt(s.get('unified_corr_in'), 2, 6)}"
            )

        print("\n[2] BACKGROUND  (bg = mean output on empty pixels, ideal ~0; mass outside the zone in persons/patch)")
        print(
            f"{'scene':<10}{'sp':<6} | bg: {'D':>6}{'S':>7}{'U':>7} | out-zone mass: {'true':>7}{'D':>8}{'S':>8}{'U':>8}"
        )
        for k in keys:
            s = per[k]
            print(
                f"{k[1]:<10}{k[0]:<6} |     {fmt(s['dense_bg'], 2, 6)}{fmt(s['sparse_bg'], 2, 7)}{fmt(s.get('unified_bg'), 2, 7)} |                 "
                f"{fmt(s['t_out'], 1, 7)}{fmt(s['dense_out'], 1, 8)}{fmt(s['sparse_out'], 1, 8)}{fmt(s.get('unified_out'), 1, 8)}"
            )

        print("\n[3] MASS / COUNT with ORACLE routing  (persons/patch; count MAE = mean |pred_total - true_total|)")
        print(
            f"{'scene':<10}{'sp':<6} | in-zone: {'true':>6}{'D':>7}{'U':>7} | total: {'true':>6}{'fused':>7}{'U':>7}{'D-alone':>8}{'S-alone':>8} | count MAE: {'fused':>6}{'U':>7}"
        )
        for k in keys:
            s = per[k]
            print(
                f"{k[1]:<10}{k[0]:<6} |          {fmt(s['t_in'], 1, 6)}{fmt(s['dense_in'], 1, 7)}{fmt(s.get('unified_in'), 1, 7)} |        "
                f"{fmt(s['t_tot'], 1, 6)}{fmt(s['fused_tot'], 1, 7)}{fmt(s.get('unified_tot'), 1, 7)}{fmt(s['dense_tot'], 1, 8)}{fmt(s['sparse_tot'], 1, 8)} |            "
                f"{fmt(s['cnt_mae_fused'], 1, 6)}{fmt(s.get('cnt_mae_unified'), 1, 7)}"
            )

        def pooled(title, rr):
            if not rr:
                return None
            s = summarize(rr, eps_list)
            print(f"\n[4] POOLED: {title}  ({s['n_patches']} patches)")
            print(
                f"    corr        dense {fmt(s['dense_corr'])}  sparse {fmt(s['sparse_corr'])}  "
                + (f"unified {fmt(s['unified_corr'])}  " if has_u else "")
                + f"fused {fmt(s['fused_corr'])}  layout {fmt(s['layout_corr'])}"
            )
            print(
                f"    bg          dense {fmt(s['dense_bg'])}  sparse {fmt(s['sparse_bg'])}  "
                + (f"unified {fmt(s['unified_bg'])}" if has_u else "")
            )
            print(
                f"    auc_crowd   dense {fmt(s['dense_auc_crowd'])}  sparse {fmt(s['sparse_auc_crowd'])}  "
                + (f"unified {fmt(s['unified_auc_crowd'])}" if has_u else "")
            )
            print(
                f"    auc_zone    dense {fmt(s['dense_auc_zone'])}  sparse {fmt(s['sparse_auc_zone'])}  "
                + (f"unified {fmt(s['unified_auc_zone'])}" if has_u else "")
            )
            print(
                f"    corr_in     dense {fmt(s['dense_corr_in'])}  "
                + (f"unified {fmt(s['unified_corr_in'])}" if has_u else "")
            )
            print(
                f"    mass        true total {fmt(s['t_tot'], 1)} (in-zone {fmt(s['t_in'], 1)}, out {fmt(s['t_out'], 1)})   dense in-zone {fmt(s['dense_in'], 1)}   sparse out {fmt(s['sparse_out'], 1)}   dense OUT {fmt(s['dense_out'], 1)}"
            )
            for tag in ["fused", "unified", "dense", "sparse"]:
                if f"cnt_mae_{tag}" in s:
                    print(
                        f"    count {tag:<8} bias {fmt(s[f'cnt_bias_{tag}'], 1, 8)}   MAE {fmt(s[f'cnt_mae_{tag}'], 1, 8)}   ({100 * s[f'cnt_mae_{tag}'] / max(s['t_tot'], 1e-6):.1f}% of mean true mass)"
                    )
            print("    router leak (fused count error, persons/patch):")
            for e in eps_list:
                print(
                    f"       eps={ek(e):<5} leak_out bias {fmt(s[f'leak_out_bias_{ek(e)}'], 1, 8)} MAE {fmt(s[f'leak_out_mae_{ek(e)}'], 1, 8)}   |   leak_sym bias {fmt(s[f'leak_sym_bias_{ek(e)}'], 1, 8)} MAE {fmt(s[f'leak_sym_mae_{ek(e)}'], 1, 8)}"
                )
            return s

        p_all = pooled("all scenes", rows)
        p_ex = (
            pooled(f"excluding weak scenes {sorted(weak)}", [r for r in rows if r["scene"] not in weak])
            if weak
            else p_all
        )

        print(
            "\n[5] RESPONSE CURVE  mean output per true-density bin (pooled over all patches). Ideal: output = mean T."
        )
        mean_t, outs, cnt = acc.curves()
        print(f"    {'bin':<16}{'pixels':>10}{'mean T':>9}  " + "  ".join(f"{n:>8}" for n in names))
        for b in range(len(cnt)):
            if cnt[b] == 0:
                continue
            hi = "+" if not np.isfinite(BIN_EDGES[b + 1]) else f"{BIN_EDGES[b + 1]:.2f}"
            print(
                f"    [{BIN_EDGES[b]:.2f},{hi:>5}){int(cnt[b]):>12}{mean_t[b]:>9.3f}  "
                + "  ".join(f"{outs[n][b]:>8.3f}" for n in names)
            )

        print("\n[6] CRITERIA (thresholds set by flags; excluding weak scenes where given)")
        ref = p_ex
        c1 = ref["dense_bg"]
        print(
            f"    1. dense pedestal on empty pixels <= {args.target_bg}:  {c1:.3f}  -> {'PASS' if c1 <= args.target_bg else 'FAIL'}"
        )
        v_scenes = [k for k in keys if k[1] not in weak]
        if has_u and v_scenes:
            frac = np.mean([per[k]["dense_corr"] >= per[k]["unified_corr"] for k in v_scenes])
            print(
                f"    2. dense corr >= unified corr in >= {args.min_frac_scenes:.0%} of scenes:  {frac:.0%}  -> {'PASS' if frac >= args.min_frac_scenes else 'FAIL'}"
            )
        if v_scenes and any(per[k]["layout_corr"] == per[k]["layout_corr"] for k in v_scenes):
            frac = np.mean(
                [
                    per[k]["dense_corr"] > per[k]["layout_corr"]
                    for k in v_scenes
                    if per[k]["layout_corr"] == per[k]["layout_corr"]
                ]
            )
            print(
                f"    3. dense corr > layout baseline in >= {args.min_frac_scenes:.0%} of scenes:  {frac:.0%}  -> {'PASS' if frac >= args.min_frac_scenes else 'FAIL'}"
            )
        e_max = max(eps_list)
        m0, m1 = ref[f"leak_out_mae_{ek(min(eps_list))}"], ref[f"leak_out_mae_{ek(e_max)}"]
        ratio = m1 / max(m0, 1e-6)
        print(
            f"    4. count MAE with leak_out(eps={ek(e_max)}) / MAE at eps={ek(min(eps_list))} <= {args.max_leak_ratio}:  {ratio:.2f}  ({m1:.1f} vs {m0:.1f})  -> {'PASS' if ratio <= args.max_leak_ratio else 'FAIL'}"
        )
        return {"pooled_all": p_all, "pooled_excl_weak": p_ex, "per_scene": {f"{k[0]}/{k[1]}": per[k] for k in keys}}

    results, order = {}, []
    for path in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        label, rows, acc, names = evaluate(path)
        results[label] = report(label, rows, acc, names)
        results[label]["rows"] = rows
        order.append(label)

    if len(order) > 1:
        print("\n" + "=" * 100 + "\nCOMPARISON (pooled, excluding weak scenes)\n" + "=" * 100)
        print(
            f"{'checkpoint':<44}{'D bg':>7}{'D corr':>8}{'S corr':>8}{'fused MAE':>10}{'leak MAE':>10}{'D-alone bias':>13}"
        )
        for lb in order:
            s = results[lb]["pooled_excl_weak"]
            print(
                f"{lb:<44}{fmt(s['dense_bg'], 2, 7)}{fmt(s['dense_corr'], 2, 8)}{fmt(s['sparse_corr'], 2, 8)}"
                f"{fmt(s['cnt_mae_fused'], 1, 10)}{fmt(s[f'leak_out_mae_{ek(max(eps_list))}'], 1, 10)}{fmt(s['cnt_bias_dense'], 1, 13)}"
            )
        if has_u:
            s = results[order[0]]["pooled_excl_weak"]
            print(
                f"{'[unified head, Stage 1]':<44}{fmt(s['unified_bg'], 2, 7)}{fmt(s['unified_corr'], 2, 8)}{'':>8}{fmt(s['cnt_mae_unified'], 1, 10)}"
            )

    out_path = os.path.join(args.out_dir, "tier0_summary.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=1)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
