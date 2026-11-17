"""
diagnose_sparse_detection.py -- where and how the sparse expert fails as a person detector (doc/head_ablation.md §6d/§8).

No training. For a Stage 2 (or later) checkpoint, on fixed patches of the train, validate and test scenes (K-means
cameras, seeded per scene like the training val cache), NMS peaks of relu(sparse) are matched one-to-one to the GT
heads within 1 m (Hungarian). Everything is broken down by each person's / peak's local crowd density (persons per m^2
in a 5 x 5 m window, the dense expert's scale) and by each person's nearest-neighbour distance:

  recall          share of people matched by a peak
  miss type       no local maximum of the sparse map within 1 m at all / a local maximum exists but below the
                  threshold / above threshold but suppressed by the NMS kernel (merged into a neighbour's peak) /
                  a peak exists within 1 m but was assigned to another person
  precision       share of peaks matched
  false-peak type duplicate (within 1 m of a person whose match is another peak) / near (1-2 m from a person) /
                  background (> 2 m from everyone)
  resp            mean of the sparse map's maximum within 1 m of each person (a single person's target peak is 1.0)
  peaks per person per density bin
  target ceiling  the same detector run on the TARGET map (GT dots blurred with the training kernel): the recall any
                  detector trained to reproduce that map could reach

Usage (src/):
    uv run python -u scripts/diagnosis/diagnose_sparse_detection.py \
        --ckpt homography_256=multiscene_best_stage_2_abl_homography_256_unmasked_joint_genloss_ep30.pth
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

DENS_BINS = [
    (0.0, 0.1, "<0.1"),
    (0.1, 0.25, "0.1-0.25"),
    (0.25, 0.5, "0.25-0.5"),
    (0.5, 1.0, "0.5-1"),
    (1.0, 1e9, ">=1"),
]
NN_BINS = [(0.0, 0.5, "<0.5m"), (0.5, 1.0, "0.5-1m"), (1.0, 2.0, "1-2m"), (2.0, 1e9, ">2m")]
MISS_TYPES = ("no_local_max", "below_threshold", "nms_suppressed", "taken_by_other")
FP_TYPES = ("duplicate", "near", "background")


def bin_of(v, bins):
    for lo, hi, lab in bins:
        if lo <= v < hi:
            return lab
    return bins[-1][2]


def main():
    import torch
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
    import train_proposed_multiscene as T

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help="name=checkpoint[:sigmoid] (':sigmoid' for a --sparse_loss focal heatmap expert)",
    )
    parser.add_argument("--splits", default="train,validate,test")
    parser.add_argument("--frames", default="train=3,validate=10,test=10", help="frames per scene per split")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--val_seed", type=int, default=1234)
    parser.add_argument("--nms_kernel", type=int, default=3)
    parser.add_argument("--threshold", default="0.064", help="peak threshold(s), comma separated")
    parser.add_argument("--out_name", default="sparse_detection")
    parser.add_argument("--match_m", type=float, default=1.0)
    parser.add_argument("--window_px", type=int, default=25, help="local density window (25 px = 5 m)")
    parser.add_argument("--out_dir", default=os.path.join(T.src_dir, "diagnostics", "sparse_detection"))
    args = parser.parse_args()

    T.ROUTER_CONTRAST = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    frames = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.frames.split(",")}
    px_per_m = T.BEV_PX_PER_M
    match_px, near_px = args.match_m * px_per_m, 2.0 * px_per_m
    win = args.window_px
    win_area_m2 = (win / px_per_m) ** 2

    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    with open(os.path.join(T.src_dir, "utils", "z_sign_corrections_full.json")) as f:
        z_sign = json.load(f)
    samples = {}
    for split in args.splits.split(","):
        reg = T.build_scene_registry(split, args.k, cams_cfg, z_sign, validate_setup=False)
        samples[split], _ = T.build_cached_val_samples(reg, frames[split], val_seed=args.val_seed)
        print(f"[data] {split}: {len(samples[split])} frames from {len(reg)} scenes")

    map_kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(device)
    float(map_kernel.sum())
    pad = map_kernel.shape[-1] // 2
    k = args.nms_kernel
    thresholds = [float(t) for t in args.threshold.split(",")]
    TARGET_THR = 0.064  # target map: one person peaks at 1.0

    def detect(m, thr):
        """m [1,1,H,W] -> (peaks above thr after NMS k, local maxima kernel 3 at any height > 1e-4,
        local maxima kernel 3 above thr) as (x, y) arrays"""
        is_max_k = F.max_pool2d(m, k, stride=1, padding=k // 2) == m
        is_max_3 = F.max_pool2d(m, 3, stride=1, padding=1) == m

        def xy(b):
            return b[0, 0].nonzero(as_tuple=False).cpu().numpy()[:, ::-1].astype(float)

        return xy(is_max_k & (m > thr)), xy(is_max_3 & (m > 1e-4)), xy(is_max_3 & (m > thr))

    def match(pred, gt):
        if len(pred) == 0 or len(gt) == 0:
            return np.full(len(gt), -1), np.full(len(pred), -1)
        d = np.linalg.norm(gt[:, None] - pred[None], axis=2)
        r, c = linear_sum_assignment(d)
        g2p, p2g = np.full(len(gt), -1), np.full(len(pred), -1)
        ok = d[r, c] <= match_px
        g2p[r[ok]], p2g[c[ok]] = c[ok], r[ok]
        return g2p, p2g

    all_results = {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        activation = "relu"
        if path.endswith(":sigmoid"):
            path, activation = path[: -len(":sigmoid")], "sigmoid"
        T.SPARSE_ACTIVATION = activation
        ckpt = torch.load(
            path if os.path.isabs(path) else os.path.join(T.src_dir, path), map_location="cpu", weights_only=False
        )
        model = T.build_model_for_checkpoint(ckpt, num_cam=args.k).to(device)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"\n[load] {name}: {path} (missing {len(missing)}, unexpected {len(unexpected)}, sparse {activation})")
        model.eval()
        res_model = {}
        for split, smp in samples.items():
            # per person: (density bin, nn bin, matched?, miss type, target-map matched?); per peak: (density bin, fp type)
            PP = {t: defaultdict(lambda: defaultdict(float)) for t in thresholds}
            QQ = {t: defaultdict(lambda: defaultdict(float)) for t in thresholds}
            for sample in smp:
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    out, _ = model(
                        sample["images"].to(device),
                        sample["ref_points_full"].to(device),
                        sample["hw_random"].to(device),
                        stage=2,
                        camera_paras=sample["camera_paras"].to(device),
                        wld_map_paras=sample["wld_map_paras"].to(device),
                    )
                wld = sample["wld_map_paras"].to(device)
                for p in range(out.shape[1]):
                    hw = sample["hw_random"].to(device)[:, :, p]
                    S = out[:, p, 1:2].float()
                    with torch.no_grad():
                        tpatch = T.create_ground_truth_heatmap(
                            sample["gt_world_xy"], wld, hw, bev_shape=T.BEV_SHAPE
                        ).to(device)
                        G = F.conv2d(tpatch, map_kernel, padding=pad)
                        dens = (
                            F.avg_pool2d(tpatch, win, stride=1, padding=win // 2, count_include_pad=False)
                            * win
                            * win
                            / win_area_m2
                        )
                    dots = T.extract_target_coordinates(sample["gt_world_xy"], wld, hw, bev_shape=T.BEV_SHAPE)
                    gt = (
                        dots.reshape(-1, 2).numpy() * np.array([T.BEV_SHAPE[1], T.BEV_SHAPE[0]])
                        if dots.numel()
                        else np.zeros((0, 2))
                    )
                    dens_np = dens[0, 0].cpu().numpy()
                    S_np = S[0, 0].cpu().numpy()

                    def dens_at(xy):
                        return dens_np[min(int(xy[1]), 199), min(int(xy[0]), 199)]

                    gpeaks, _, _ = detect(G, TARGET_THR)
                    g2p_G, _ = match(gpeaks, gt)
                    if len(gt) > 1:
                        nn = cKDTree(gt).query(gt, k=2)[0][:, 1] / px_per_m
                    else:
                        nn = np.full(len(gt), 99.0)
                    for thr in thresholds:
                        P, Q = PP[thr], QQ[thr]
                        peaks, lmax, above3 = detect(S, thr)
                        g2p, p2g = match(peaks, gt)
                        tree_l = cKDTree(lmax) if len(lmax) else None
                        tree_a = cKDTree(above3) if len(above3) else None
                        tree_p = cKDTree(peaks) if len(peaks) else None
                        for i, xy in enumerate(gt):
                            db, nb = bin_of(dens_at(xy), DENS_BINS), bin_of(nn[i], NN_BINS)
                            for key in (("dens", db), ("nn", nb), ("all", "all")):
                                row = P[key]
                                row["people"] += 1
                                row["tp"] += g2p[i] >= 0
                                row["target_tp"] += g2p_G[i] >= 0
                                row["resp"] += float(
                                    S_np[
                                        max(int(xy[1]) - 5, 0) : int(xy[1]) + 6, max(int(xy[0]) - 5, 0) : int(xy[0]) + 6
                                    ].max()
                                )
                                if g2p[i] < 0:
                                    if tree_l is None or not tree_l.query_ball_point(xy, match_px):
                                        row["no_local_max"] += 1
                                    elif tree_a is None or not tree_a.query_ball_point(xy, match_px):
                                        row["below_threshold"] += 1
                                    elif tree_p is None or not tree_p.query_ball_point(xy, match_px):
                                        row["nms_suppressed"] += 1
                                    else:
                                        row["taken_by_other"] += 1
                        tree_g = cKDTree(gt) if len(gt) else None
                        for j, xy in enumerate(peaks):
                            db = bin_of(dens_at(xy), DENS_BINS)
                            for key in (("dens", db), ("all", "all")):
                                row = Q[key]
                                row["peaks"] += 1
                                if p2g[j] >= 0:
                                    row["tp"] += 1
                                    continue
                                dist = tree_g.query(xy)[0] if tree_g is not None else 1e9
                                if dist <= match_px:
                                    row["duplicate"] += 1
                                elif dist <= near_px:
                                    row["near"] += 1
                                else:
                                    row["background"] += 1

            res_split = {}
            for thr in thresholds:
                P, Q = PP[thr], QQ[thr]
                res = {"people": {}, "peaks": {}}
                print(
                    f"\n{'=' * 110}\n{name} -- {split} ({len(smp)} frames x {out.shape[1]} patches), NMS kernel {k}, "
                    f"threshold {thr}, match {args.match_m} m\n{'=' * 110}"
                )
                print(
                    f"{'group':<14}{'people%':>8}{'recall':>8}{'target':>8}{'resp':>8}   misses: "
                    + "".join(f"{m:>16}" for m in MISS_TYPES)
                    + "   (share of people in the group)"
                )
                tot = P[("all", "all")]["people"]
                for kind, bins in (("all", [(0, 0, "all")]), ("dens", DENS_BINS), ("nn", NN_BINS)):
                    for _, _, lab in bins:
                        row = P[(kind, lab)]
                        if not row["people"]:
                            continue
                        n_ = row["people"]
                        r = dict(
                            people_pct=100 * n_ / tot,
                            recall=row["tp"] / n_,
                            target_recall=row["target_tp"] / n_,
                            max_response=row["resp"] / n_,
                            **{m: row[m] / n_ for m in MISS_TYPES},
                        )
                        res["people"][f"{kind}:{lab}"] = r
                        print(
                            f"{(lab if kind != 'all' else 'ALL'):<14}{r['people_pct']:>8.1f}{r['recall']:>8.2f}"
                            f"{r['target_recall']:>8.2f}{r['max_response']:>8.2f}            "
                            + "".join(f"{r[m]:>16.2f}" for m in MISS_TYPES)
                        )
                    if kind == "all":
                        print("  by local density (persons / m^2, 5 x 5 m):")
                    elif kind == "dens":
                        print("  by nearest-neighbour distance:")
                print(
                    f"\n{'peaks at':<14}{'peaks%':>8}{'prec':>8}{'pk/pers':>9}   false peaks: "
                    + "".join(f"{t:>12}" for t in FP_TYPES)
                    + "   (share of peaks in the group)"
                )
                totp = Q[("all", "all")]["peaks"]
                for kind, bins in (("all", [(0, 0, "all")]), ("dens", DENS_BINS)):
                    for _, _, lab in bins:
                        row = Q[(kind, lab)]
                        if not row["peaks"]:
                            continue
                        n_ = row["peaks"]
                        ppl = P[(kind, lab)]["people"] if kind == "dens" else tot
                        r = dict(
                            peaks_pct=100 * n_ / max(totp, 1),
                            precision=row["tp"] / n_,
                            peaks_per_person=n_ / max(ppl, 1),
                            **{t: row[t] / n_ for t in FP_TYPES},
                        )
                        res["peaks"][f"{kind}:{lab}"] = r
                        print(
                            f"{(lab if kind != 'all' else 'ALL'):<14}{r['peaks_pct']:>8.1f}{r['precision']:>8.2f}"
                            f"{r['peaks_per_person']:>9.2f}                "
                            + "".join(f"{r[t]:>12.2f}" for t in FP_TYPES)
                        )
                    if kind == "all":
                        print("  by local density at the peak:")
                res_split[str(thr)] = res
            print(
                f"\n{name} -- {split} SUMMARY (F1 overall | people below 0.25 p/m^2: recall, and precision of the peaks there)"
            )
            for thr in thresholds:
                P, Q = PP[thr], QQ[thr]
                tp, n, pk = P[("all", "all")]["tp"], P[("all", "all")]["people"], Q[("all", "all")]["peaks"]
                lo = [("dens", "<0.1"), ("dens", "0.1-0.25")]
                tpl, nl = sum(P[b]["tp"] for b in lo), sum(P[b]["people"] for b in lo)
                qtl, qnl = sum(Q[b]["tp"] for b in lo), sum(Q[b]["peaks"] for b in lo)
                rec, prec = tp / max(n, 1), Q[("all", "all")]["tp"] / max(pk, 1)
                rl, pl = tpl / max(nl, 1), qtl / max(qnl, 1)
                f1, f1l = 2 * rec * prec / max(rec + prec, 1e-9), 2 * rl * pl / max(rl + pl, 1e-9)
                res_split[str(thr)]["summary"] = dict(
                    recall=rec, precision=prec, f1=f1, recall_low=rl, precision_low=pl, f1_low=f1l
                )
                print(
                    f"  thr {thr:<6} F1 {f1:.3f} (R {rec:.2f} P {prec:.2f}) | <0.25 p/m^2: F1 {f1l:.3f} (R {rl:.2f} P {pl:.2f})"
                )
            res_model[split] = res_split
        all_results[name] = res_model
        del model
        torch.cuda.empty_cache()

    out = os.path.join(args.out_dir, f"{args.out_name}.json")
    with open(out, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=1)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
