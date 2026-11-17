"""
sweep_sparse_nms.py -- offline NMS threshold / kernel sweep for the sparse expert read as a detector
(doc/head_ablation.md §6b, next step 1). No training.

The 'peaks' sparse output = one map kernel (one person) per NMS peak of relu(sparse). For every (threshold, kernel)
pair, on the same patches as diagnose_perf_routing.py (K-means cameras, val_seed 1234):
  detection   precision / recall / F1 at 1 m, peaks per person
  count       peaks alone MAE / bias
  routing     upper bound (best expert per cell, uses GT) at 10 / 25 / 50 px cells, and the trained router's soft mask
              applied to dense vs peaks
Expert outputs are computed once per patch; every pair is scored on them.

Usage (src/):
    uv run python -u scripts/diagnosis/sweep_sparse_nms.py \
        --ckpt homography_256=multiscene_best_stage_3_abl_homography_256_unmasked_joint_genloss_ep30_perf_c10_sel.pth \
        --split validate
"""

import os
import sys
import json
import argparse
import itertools
from collections import defaultdict

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

THRESHOLDS = (0.01, 0.02, 0.04, 0.064, 0.1, 0.15, 0.2, 0.3, 0.5)
KERNELS = (3, 5, 7, 9)
CELLS = (10, 25, 50)


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
    parser.add_argument("--out_dir", default=os.path.join(T.src_dir, "diagnostics", "perf_routing"))
    args = parser.parse_args()

    T.ROUTER_CONTRAST = False  # every perf-router checkpoint was trained with --no_router_contrast
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

    def to_cells(x, c):
        return F.avg_pool2d(x, c, stride=c) * (c * c) / KMP

    combos = list(itertools.product(THRESHOLDS, KERNELS))

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

        acc = {c: defaultdict(list) for c in combos}
        det = {c: dict(tp=0, pred=0, gt=0) for c in combos}
        dense_err, peak_values = [], []
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
                    tpatch = T.create_ground_truth_heatmap(sample["gt_world_xy"], wld, hw, bev_shape=T.BEV_SHAPE).to(
                        device
                    )
                    G = F.conv2d(tpatch, map_kernel, padding=pad)
                g = G.sum().item() / KMP
                d_cnt = D.sum().item() / KMP
                dense_err.append(d_cnt - g)
                dots = T.extract_target_coordinates(sample["gt_world_xy"], wld, hw, bev_shape=T.BEV_SHAPE)
                gt_xy = (
                    (dots.reshape(-1, 2).numpy() * np.array([T.BEV_SHAPE[1], T.BEV_SHAPE[0]]))
                    if dots.numel()
                    else np.zeros((0, 2))
                )
                dc = {c: to_cells(D, c) for c in CELLS}
                gc_ = {c: to_cells(G, c) for c in CELLS}
                # value of the sparse map at its local maxima (kernel 3, any height): where a threshold can cut
                lm = F.max_pool2d(S, 3, stride=1, padding=1)
                peak_values.append(S[(lm == S) & (S > 1e-4)].cpu().numpy())
                for kern in KERNELS:
                    is_max = F.max_pool2d(S, kern, stride=1, padding=kern // 2) == S
                    for thr in THRESHOLDS:
                        pk = (is_max & (S > thr)).float()
                        Sp = F.conv2d(pk, map_kernel, padding=pad)
                        a = acc[(thr, kern)]
                        a["alone"].append(Sp.sum().item() / KMP - g)
                        a["soft"].append((m * D + (1 - m) * Sp).sum().item() / KMP - g)
                        for c in CELLS:
                            s_ = to_cells(Sp, c)
                            best = torch.where((dc[c] - gc_[c]).abs() < (s_ - gc_[c]).abs(), dc[c], s_)
                            a[f"upper{c}"].append(best.sum().item() - g)
                        pr = pk[0, 0].nonzero(as_tuple=False).cpu().numpy()[:, ::-1].astype(np.float64)
                        dd = det[(thr, kern)]
                        dd["tp"] += T.count_matches_np(pr, gt_xy, T.SPARSE_MATCH_PX)
                        dd["pred"] += len(pr)
                        dd["gt"] += len(gt_xy)

        n = len(dense_err)
        dmae = float(np.mean(np.abs(dense_err)))
        pv = np.concatenate(peak_values)
        print(
            f"\n{'=' * 104}\n{name} -- {args.split}, {n} patches | dense alone MAE {dmae:.1f} | current setting = thr 0.064, kernel 7"
        )
        print(
            f"sparse local-max values (kernel 3): n/patch {len(pv) / n:.0f} | quantiles 10/25/50/75/90% "
            + " / ".join(f"{q:.3f}" for q in np.quantile(pv, [0.1, 0.25, 0.5, 0.75, 0.9]))
            + f"\n{'=' * 104}"
        )
        print(
            f"{'thr':>6}{'kern':>5}{'prec':>7}{'rec':>6}{'F1':>6}{'pk/pp':>7}{'alone':>8}{'bias':>8}"
            + "".join(f"{'up' + str(c):>8}" for c in CELLS)
            + f"{'router':>8}"
        )
        rows = []
        for thr, kern in combos:
            a, dd = acc[(thr, kern)], det[(thr, kern)]
            prec = dd["tp"] / max(dd["pred"], 1)
            rec = dd["tp"] / max(dd["gt"], 1)
            row = dict(
                threshold=thr,
                kernel=kern,
                precision=prec,
                recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-9),
                peaks_per_person=dd["pred"] / max(dd["gt"], 1),
                alone_mae=float(np.mean(np.abs(a["alone"]))),
                alone_bias=float(np.mean(a["alone"])),
                router_soft_mae=float(np.mean(np.abs(a["soft"]))),
                **{f"upper{c}_mae": float(np.mean(np.abs(a[f"upper{c}"]))) for c in CELLS},
            )
            rows.append(row)
            print(
                f"{thr:>6.3f}{kern:>5}{prec:>7.2f}{rec:>6.2f}{row['f1']:>6.2f}{row['peaks_per_person']:>7.2f}"
                f"{row['alone_mae']:>8.1f}{row['alone_bias']:>+8.1f}"
                + "".join(f"{row[f'upper{c}_mae']:>8.1f}" for c in CELLS)
                + f"{row['router_soft_mae']:>8.1f}"
            )
        for key in ("f1", "alone_mae", "upper25_mae", "upper50_mae"):
            best = (max if key == "f1" else min)(rows, key=lambda r: r[key])
            print(
                f"best {key:<12}: thr {best['threshold']:.3f}, kernel {best['kernel']} -> {best[key]:.3f} "
                f"(F1 {best['f1']:.2f}, recall {best['recall']:.2f}, upper25 {best['upper25_mae']:.1f})"
            )
        all_results[name] = dict(
            n_patches=n,
            dense_alone_mae=dmae,
            rows=rows,
            peak_value_quantiles=dict(
                zip(["10", "25", "50", "75", "90"], map(float, np.quantile(pv, [0.1, 0.25, 0.5, 0.75, 0.9])))
            ),
        )
        del model
        torch.cuda.empty_cache()

    out = os.path.join(args.out_dir, f"nms_sweep_{args.split}.json")
    with open(out, "w") as f:
        json.dump(
            {"args": vars(args), "thresholds": THRESHOLDS, "kernels": KERNELS, "results": all_results}, f, indent=1
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
