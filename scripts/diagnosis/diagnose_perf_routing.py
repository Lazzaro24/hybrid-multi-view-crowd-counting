"""
diagnose_perf_routing.py -- offline diagnostics of the performance-routing head (doc/head_ablation.md, step 1).

No training. For each Stage 3 performance-router checkpoint, on the fixed val patches (same cache as training, val_seed
1234) or on test patches drawn the same way, with the K-means cameras:

  0. Sanity: dense alone / sparse alone / per-cell upper bound / learned soft / hard (must match the Stage 3 log).
  1. Upper bound vs routing cell size (10 px ... 200 px = whole patch): how much of the per-cell bound is lost to the
     cells' signed errors not cancelling, and how much is real headroom.
  2. Sparse expert variants without retraining:
       raw         relu(sparse)
       peaks       one person (one map kernel) per NMS peak -- the pedestal is gone by construction
       near-peaks  relu(sparse) kept only within `--near_px` of a peak (pedestal removed, peak mass kept)
     each scored alone, in the upper bound and under the learned router; plus sparse F1 and peaks per person.
  3. Where sparse wins: 10 px cells binned by GT persons per cell and by local density (25 px window, persons/m^2);
     per bin the share of cells / people, the label share, and the people each expert's error costs.
  4. Router-free control: fixed global blend a*dense + (1-a)*sparse, a fitted on these patches (optimistic).

Usage (src/):
    uv run python -u scripts/diagnosis/diagnose_perf_routing.py \
        --ckpt homography_256=multiscene_best_stage_3_abl_homography_256_unmasked_joint_genloss_ep30_perf_c10_sel.pth \
        --ckpt ca_sa=multiscene_best_stage_3_abl_ca_sa_unmasked_joint_genloss_ep30_perf_c10_sel.pth --split validate
Do not run while a training is running.
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

CELLS = (10, 20, 25, 40, 50, 100, 200)
COUNT_BINS = [(0.0, 0.05, "empty"), (0.05, 1.0, "<1"), (1.0, 3.0, "1-3"), (3.0, 6.0, "3-6"), (6.0, 1e9, ">6")]
DENS_BINS = [
    (0.0, 0.05, "<0.05"),
    (0.05, 0.15, "0.05-0.15"),
    (0.15, 0.3, "0.15-0.3"),
    (0.3, 0.5, "0.3-0.5"),
    (0.5, 1e9, ">=0.5"),
]
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
SPARSE_VARIANTS = ("raw", "peaks", "near")


def main():
    import torch
    import torch.nn.functional as F
    import train_proposed_multiscene as T

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action="append", required=True, help="name=checkpoint (Stage 3, performance router)")
    parser.add_argument("--split", choices=["validate", "test"], default="validate")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--val_seed", type=int, default=1234)
    parser.add_argument("--frames_per_scene", type=int, default=10)
    parser.add_argument("--near_px", type=int, default=3, help="'near-peaks': keep sparse mass within this many px")
    parser.add_argument("--out_dir", default=os.path.join(T.src_dir, "diagnostics", "perf_routing"))
    parser.add_argument("--nms_kernel", type=int, default=7, help="NMS kernel of the 'peaks' / 'near' sparse variants")
    parser.add_argument(
        "--no_router", action="store_true", help="Stage 2 checkpoints (untrained router): skip the learned-router rows"
    )
    parser.add_argument("--out_name", default=None, help="default perf_routing_<split>")
    args = parser.parse_args()

    T.ROUTER_CONTRAST = False  # every perf-router checkpoint was trained with --no_router_contrast
    T.SPARSE_PEAK_THRESHOLD_MULTIPLIER, T.SPARSE_NMS_KERNEL_SIZE = 1, args.nms_kernel  # threshold 0.064
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    with open(os.path.join(T.src_dir, "utils", "z_sign_corrections_full.json")) as f:
        z_sign = json.load(f)
    registry = T.build_scene_registry(args.split, args.k, cams_cfg, z_sign, validate_setup=False)
    samples, _ = T.build_cached_val_samples(registry, args.frames_per_scene, val_seed=args.val_seed)
    print(f"[data] {args.split}: {len(samples)} frames from {len(registry)} scenes")

    map_kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(device)
    KMP = float(map_kernel.sum())
    pad = map_kernel.shape[-1] // 2
    single_person_peak = T.GAUSSIAN_SCALE / (2 * np.pi * T.GAUSSIAN_SIGMA**2)
    peak_thr = single_person_peak * T.PEAK_THRESHOLD_FRACTION * T.SPARSE_PEAK_THRESHOLD_MULTIPLIER
    nms_k = T.SPARSE_NMS_KERNEL_SIZE
    win = 25  # px = 5 m, the dense expert's receptive field (oracle_mask_redesign.md §5)

    def peak_mask(s):
        local_max = F.max_pool2d(s, nms_k, stride=1, padding=nms_k // 2)
        return ((local_max == s) & (s > peak_thr)).float()

    def to_cells(x, c):  # map in map_gt_conv units -> persons per c x c cell
        return F.avg_pool2d(x, c, stride=c) * (c * c) / KMP

    all_results = {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        ckpt = torch.load(
            path if os.path.isabs(path) else os.path.join(T.src_dir, path), map_location="cpu", weights_only=False
        )
        model = T.build_model_for_checkpoint(ckpt, num_cam=args.k).to(device)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"\n[load] {name}: {path} (missing {len(missing)}, unexpected {len(unexpected)})")
        model.eval()

        defaultdict(list)  # per-patch absolute count errors
        signed = defaultdict(list)  # per-patch signed errors
        gt_counts, scenes = [], []
        cell_sum_abs = defaultdict(list)  # sum over 10 px cells of |err| (no cancellation)
        blend_pred = {v: [] for v in SPARSE_VARIANTS}  # (dense count, sparse count) per patch for the blend fit
        f1 = dict(tp=0, pred=0, gt=0)
        empty_mass = dict(dense=0.0, sparse=0.0, near=0.0, n=0)
        bins = {kind: defaultdict(lambda: defaultdict(float)) for kind in ("count", "density")}

        for sample in samples:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                proj, masks = model(
                    sample["images"].to(device),
                    sample["ref_points_full"].to(device),
                    sample["hw_random"].to(device),
                    stage=3,
                    camera_paras=sample["camera_paras"].to(device),
                    wld_map_paras=sample["wld_map_paras"].to(device),
                )
            wld = sample["wld_map_paras"].to(device)
            for p in range(proj.shape[1]):
                hw = sample["hw_random"].to(device)[:, :, p]
                feat = proj[:, p].float()
                m = torch.clamp(masks[:, p].float(), 1e-6, 1 - 1e-6)
                with torch.no_grad():
                    D = torch.relu(model.head.dense_expert(feat)).float()
                    S = torch.relu(model.head.sparse_expert(feat)).float()
                    pk = peak_mask(S)
                    S_var = {
                        "raw": S,
                        "peaks": F.conv2d(pk, map_kernel, padding=pad),
                        "near": S * (F.max_pool2d(pk, 2 * args.near_px + 1, stride=1, padding=args.near_px) > 0),
                    }
                    tpatch = T.create_ground_truth_heatmap(sample["gt_world_xy"], wld, hw, bev_shape=T.BEV_SHAPE).to(
                        device
                    )
                    G = F.conv2d(tpatch, map_kernel, padding=pad)
                g = G.sum().item() / KMP
                gt_counts.append(g)
                scenes.append(sample["scene_name"])

                def cnt(x):
                    return x.sum().item() / KMP

                # 0. sanity, exactly as the Stage 3 validation
                ct = T.cell_routing_targets(D, S, G, 10, KMP)
                cell_prob = F.avg_pool2d(m, 10, stride=10)
                hard = cell_prob > 0.5
                for key, val in (
                    ("dense alone", cnt(D)),
                    ("learned soft", cnt(m * D + (1 - m) * S)),
                    ("learned hard", torch.where(hard, ct["dense"], ct["sparse"]).sum().item()),
                ):
                    signed[key].append(val - g)
                cell_sum_abs["dense"].append((ct["dense"] - ct["gt"]).abs().sum().item())

                # 1 + 2. sparse variants: alone, upper bound per cell size, under the learned router
                for v, Sv in S_var.items():
                    signed[f"sparse alone [{v}]"].append(cnt(Sv) - g)
                    signed[f"learned soft [{v}]"].append(cnt(m * D + (1 - m) * Sv) - g)
                    for c in CELLS:
                        d_, s_, g_ = to_cells(D, c), to_cells(Sv, c), to_cells(G, c)
                        best = torch.where((d_ - g_).abs() < (s_ - g_).abs(), d_, s_)
                        signed[f"upper c{c} [{v}]"].append(best.sum().item() - g)
                    blend_pred[v].append((cnt(D), cnt(Sv)))
                    if v != "raw":
                        cell_sum_abs[f"sparse {v}"].append((to_cells(Sv, 10) - ct["gt"]).abs().sum().item())
                cell_sum_abs["sparse raw"].append((ct["sparse"] - ct["gt"]).abs().sum().item())

                # sparse F1 (whole patch, 1 m match radius) and peaks per person
                dots = T.extract_target_coordinates(sample["gt_world_xy"], wld, hw, bev_shape=T.BEV_SHAPE)
                gt_xy = (
                    (dots.reshape(-1, 2).numpy() * np.array([T.BEV_SHAPE[1], T.BEV_SHAPE[0]]))
                    if dots.numel()
                    else np.zeros((0, 2))
                )
                pr = pk[0, 0].nonzero(as_tuple=False).cpu().numpy()[:, ::-1].astype(np.float64)  # (x, y)
                f1["tp"] += T.count_matches_np(pr, gt_xy, T.SPARSE_MATCH_PX)
                f1["pred"] += len(pr)
                f1["gt"] += len(gt_xy)

                # pedestal: mass on empty cells (GT < 0.05 persons per 10 px cell)
                empty = (ct["gt"] < 0.05).float()
                empty_mass["dense"] += (ct["dense"] * empty).sum().item()
                empty_mass["sparse"] += (ct["sparse"] * empty).sum().item()
                empty_mass["near"] += (to_cells(S_var["near"], 10) * empty).sum().item()
                empty_mass["n"] += 1

                # 3. where sparse wins (10 px cells, raw sparse = what the router actually chose between)
                dens = (
                    F.avg_pool2d(G, win, stride=1, padding=win // 2)
                    * (win * win)
                    / KMP
                    / (win * win / T.BEV_PX_PER_M**2)
                )
                dens_c = F.avg_pool2d(dens, 10, stride=10)
                for kind, key_map, edges in (("count", ct["gt"], COUNT_BINS), ("density", dens_c, DENS_BINS)):
                    for lo, hi, lab in edges:
                        sel = ((key_map >= lo) & (key_map < hi)).float()
                        b = bins[kind][lab]
                        b["cells"] += sel.sum().item()
                        b["people"] += (ct["gt"] * sel).sum().item()
                        b["label_dense"] += (ct["label"] * sel).sum().item()
                        b["routed_dense"] += (hard.float() * sel).sum().item()
                        b["err_dense"] += (ct["err_dense"] * sel).sum().item()
                        b["err_sparse"] += (ct["err_sparse"] * sel).sum().item()
                        b["err_best"] += (torch.minimum(ct["err_dense"], ct["err_sparse"]) * sel).sum().item()
                        b["signed_dense"] += ((ct["dense"] - ct["gt"]) * sel).sum().item()
                        b["signed_sparse"] += ((ct["sparse"] - ct["gt"]) * sel).sum().item()
                        b["gain_sparse"] += (ct["weight"] * (1 - ct["label"]) * sel).sum().item()

        n = len(gt_counts)

        def mae(key):
            return float(np.mean(np.abs(signed[key])))

        def bias(key):
            return float(np.mean(signed[key]))

        res = {"n_patches": n, "mean_gt": float(np.mean(gt_counts)), "rows": {}}
        print(
            f"\n{'=' * 96}\n{name} -- {args.split}, {n} patches, mean GT {np.mean(gt_counts):.1f} persons/patch\n{'=' * 96}"
        )

        print("\n0. Sanity (should match the Stage 3 log)")
        for key in ("dense alone", "sparse alone [raw]", "upper c10 [raw]") + (
            () if args.no_router else ("learned soft", "learned hard")
        ):
            res["rows"][key] = dict(mae=mae(key), bias=bias(key))
            print(f"   {key:<22} MAE {mae(key):7.1f}  bias {bias(key):+7.1f}")

        print("\n1+2. Upper bound (best expert per cell, uses GT) by cell size and sparse variant -- MAE")
        print(f"   {'cell px':<10}" + "".join(f"{v:>12}" for v in SPARSE_VARIANTS) + f"{'dense alone':>14}")
        for c in CELLS:
            row = {v: mae(f"upper c{c} [{v}]") for v in SPARSE_VARIANTS}
            res["rows"][f"upper c{c}"] = row
            print(f"   {c:<10}" + "".join(f"{row[v]:>12.1f}" for v in SPARSE_VARIANTS) + f"{mae('dense alone'):>14.1f}")
        print(f"\n   {'':<22}" + "".join(f"{v:>12}" for v in SPARSE_VARIANTS))
        for lab, fmt in (("sparse alone", "sparse alone [{}]"),) + (
            () if args.no_router else (("learned soft router", "learned soft [{}]"),)
        ):
            print(f"   {lab + ' MAE':<22}" + "".join(f"{mae(fmt.format(v)):>12.1f}" for v in SPARSE_VARIANTS))
            print(f"   {lab + ' bias':<22}" + "".join(f"{bias(fmt.format(v)):>+12.1f}" for v in SPARSE_VARIANTS))
            for v in SPARSE_VARIANTS:
                res["rows"][fmt.format(v)] = dict(mae=mae(fmt.format(v)), bias=bias(fmt.format(v)))
        prec = f1["tp"] / max(f1["pred"], 1)
        rec = f1["tp"] / max(f1["gt"], 1)
        res["sparse_detection"] = dict(
            precision=prec,
            recall=rec,
            f1=2 * prec * rec / max(prec + rec, 1e-9),
            peaks_per_person=f1["pred"] / max(f1["gt"], 1),
        )
        print(
            f"   sparse peaks: precision {prec:.2f} | recall {rec:.2f} | F1 {res['sparse_detection']['f1']:.2f} | "
            f"{f1['pred'] / max(f1['gt'], 1):.2f} peaks per person (NMS {nms_k}, threshold {peak_thr:.3f})"
        )
        en = empty_mass["n"]
        res["empty_cell_mass"] = {k: v / en for k, v in empty_mass.items() if k != "n"}
        print(
            f"   mass on empty 10 px cells (persons/patch): dense {empty_mass['dense'] / en:.1f} | sparse raw "
            f"{empty_mass['sparse'] / en:.1f} | sparse near-peaks {empty_mass['near'] / en:.1f}"
        )
        print("   error cancellation, 10 px cells (persons/patch): sum over cells of |err| vs |patch err|:")
        for k_, key in (
            ("dense", "dense alone"),
            ("sparse raw", "sparse alone [raw]"),
            ("sparse peaks", "sparse alone [peaks]"),
            ("sparse near", "sparse alone [near]"),
        ):
            res["rows"][f"cellsum {k_}"] = float(np.mean(cell_sum_abs[k_]))
            print(f"      {k_:<14} {np.mean(cell_sum_abs[k_]):7.1f}  vs  {mae(key):7.1f}")

        print("\n3. Where sparse wins (10 px cells, raw sparse; 'gain' = people saved in cells where sparse is better)")
        res["bins"] = {}
        for kind, edges, title in (
            ("count", COUNT_BINS, "GT persons per cell"),
            ("density", DENS_BINS, "local density p/m^2 (25 px window)"),
        ):
            print(f"   by {title}")
            print(
                f"   {'bin':<11}{'cells%':>8}{'people%':>9}{'lbl dense%':>11}{'rtd dense%':>11}"
                f"{'errD':>8}{'errS':>8}{'best':>8}{'biasD':>8}{'biasS':>8}{'gainS':>8}   (per patch)"
            )
            tc = sum(b["cells"] for b in bins[kind].values())
            tp_ = sum(b["people"] for b in bins[kind].values())
            res["bins"][kind] = {}
            for _, _, lab in edges:
                b = bins[kind][lab]
                if b["cells"] == 0:
                    continue
                row = dict(
                    cells_pct=100 * b["cells"] / tc,
                    people_pct=100 * b["people"] / max(tp_, 1e-9),
                    label_dense_pct=100 * b["label_dense"] / b["cells"],
                    routed_dense_pct=100 * b["routed_dense"] / b["cells"],
                    err_dense=b["err_dense"] / n,
                    err_sparse=b["err_sparse"] / n,
                    err_best=b["err_best"] / n,
                    bias_dense=b["signed_dense"] / n,
                    bias_sparse=b["signed_sparse"] / n,
                    gain_sparse=b["gain_sparse"] / n,
                )
                res["bins"][kind][lab] = row
                print(
                    f"   {lab:<11}{row['cells_pct']:>8.1f}{row['people_pct']:>9.1f}{row['label_dense_pct']:>11.1f}"
                    f"{row['routed_dense_pct']:>11.1f}{row['err_dense']:>8.1f}{row['err_sparse']:>8.1f}{row['err_best']:>8.1f}"
                    f"{row['bias_dense']:>+8.1f}{row['bias_sparse']:>+8.1f}{row['gain_sparse']:>8.1f}"
                )

        print("\n4. Fixed global blend a*dense + (1-a)*sparse (a fitted on these patches -- optimistic)")
        g_arr = np.array(gt_counts)
        res["blend"] = {}
        for v in SPARSE_VARIANTS:
            dc, sc = np.array(blend_pred[v]).T
            maes = [float(np.mean(np.abs(a * dc + (1 - a) * sc - g_arr))) for a in ALPHAS]
            i = int(np.argmin(maes))
            res["blend"][v] = dict(alpha=float(ALPHAS[i]), mae=maes[i], curve=dict(zip(map(float, ALPHAS), maes)))
            print(
                f"   sparse {v:<6}: best a = {ALPHAS[i]:.2f} -> MAE {maes[i]:.1f}  (a=1 dense alone {maes[-1]:.1f}, "
                f"a=0.5 {maes[10]:.1f})"
            )

        # per scene: dense alone vs best upper bound, to see whether one scene drives it
        sc_arr = np.array(scenes)
        res["per_scene"] = {}
        for s in sorted(set(scenes), key=lambda s_: -g_arr[sc_arr == s_].mean()):
            idx = sc_arr == s
            res["per_scene"][s] = {
                k_: float(np.mean(np.abs(np.array(signed[k_])[idx])))
                for k_ in (
                    "dense alone",
                    "learned soft",
                    "upper c10 [raw]",
                    "upper c200 [raw]",
                    "sparse alone [raw]",
                    "sparse alone [peaks]",
                    "upper c10 [peaks]",
                )
            }
            res["per_scene"][s]["gt"] = float(g_arr[idx].mean())

        # 5. Predictability of each expert's count error across scenes (doc/head_ablation.md §6d): a router can only
        # use the headroom if the sparse expert's error has a stable sign. Per scene: mean signed error per patch.
        BIAS_KEYS = ("dense alone", "sparse alone [raw]", "sparse alone [peaks]", "sparse alone [near]")
        print(
            "\n5. Per-scene bias (mean signed error per patch, persons): a predictable expert errs the same way everywhere"
        )
        print(
            f"   {'scene':<10}{'GT':>7}" + "".join(f"{k_.replace('sparse alone ', 'sparse '):>17}" for k_ in BIAS_KEYS)
        )
        sb = {k_: [] for k_ in BIAS_KEYS}
        for s in res["per_scene"]:
            idx = sc_arr == s
            vals = {k_: float(np.mean(np.array(signed[k_])[idx])) for k_ in BIAS_KEYS}
            for k_ in BIAS_KEYS:
                sb[k_].append(vals[k_])
                res["per_scene"][s][f"bias {k_}"] = vals[k_]
            print(f"   {s:<10}{res['per_scene'][s]['gt']:>7.0f}" + "".join(f"{vals[k_]:>+17.1f}" for k_ in BIAS_KEYS))
        res["bias_spread"] = {}
        for lab, fn in (
            ("mean", np.mean),
            ("std across scenes", np.std),
            ("scenes > 0 (of n)", lambda x: float(np.sum(np.array(x) > 0))),
        ):
            vals = {k_: float(fn(sb[k_])) for k_ in BIAS_KEYS}
            res["bias_spread"][lab] = vals
            print(f"   {lab:<17}" + "".join(f"{vals[k_]:>17.1f}" for k_ in BIAS_KEYS))
        all_results[name] = res
        del model
        torch.cuda.empty_cache()

    out = os.path.join(args.out_dir, f"{args.out_name or 'perf_routing'}_{args.split}.json")
    with open(out, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=1)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
