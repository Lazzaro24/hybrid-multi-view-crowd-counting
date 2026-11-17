"""
Data-only analysis of the oracle dense-zone mask across scenes (see utils/oracle_mask.py for the criteria).
No model is loaded -- the only model-derived number is the dense expert's theoretical receptive field, read from
HybridRoutingHead's layer config.

Two sweeps (--sweep):
  extent  pointwise threshold tau (true single-person peaks) x morphological opening min_extent k. Finding
          (2026-10-04, 38 scenes): ~half of all dense components are bare pairs, and NO component survives k >= 17,
          so "the dense expert's window must fit inside the mask" deletes the mask entirely.
  window  window-averaged density W (odd, BEV px) x D (persons/m^2), with and without the occupied-only restriction,
          against the current mask as reference. Measures crowding at the expert's scale instead of the 0.45 m
          target kernel's.

Run on TRAIN + VALIDATE scenes only: the mask is a design choice and must not be tuned on the test split.
Masks are computed over each frame's full scene BEV (as evaluate_paper_protocol.oracle_zone does), not on random
200 x 200 patches, so the numbers do not depend on the crop draw.

Per config and frame:
  cov_scene     mask pixels / scene pixels
  cov_occupied  mask pixels / occupied pixels (Gaussian density > EMPTY_FRACTION single-person peaks)
  people_dense  share of GT people whose pixel lies inside the mask (= routed to dense_expert)
  n_comp        connected components (8-connectivity)
  rf_empty      mean, over mask pixels, of the share of the dense expert's RF x RF window that is empty --
                the "dense expert looks at background" quantity
  rf_people     mean, over mask pixels, of GT people inside that RF x RF window
Per component: area (px) and people; in the extent sweep also max_extent = side (px) of the largest square that
fits inside it (largest odd k whose opening still keeps part of it).

Usage:
    uv run scripts/diagnosis/analyse_oracle_mask_extent.py --sweep window --frames_per_scene 10
    uv run scripts/diagnosis/analyse_oracle_mask_extent.py --sweep extent --frames_per_scene 10
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

EMPTY_FRACTION = (
    0.1  # pixel is "empty" if its Gaussian density is below this many single-person peaks (> ~1 m from anyone)
)
CURRENT_TAU = 1.273  # the training mask's threshold in true single-person peaks (T.oracle_density_threshold())


def dense_expert_receptive_field():
    """Theoretical RF of HybridRoutingHead.dense_expert: 1 + sum((kernel - 1) * dilation) over its stride-1 convs."""
    import torch.nn as nn
    from models.head.proposed.routing import HybridRoutingHead

    rf = 1
    for m in HybridRoutingHead().dense_expert:
        if isinstance(m, nn.Conv2d):
            assert m.stride == (1, 1)
            rf += (m.kernel_size[0] - 1) * m.dilation[0]
    return rf


def build_configs(args, peak, ksum):
    """List of (name, group, kwargs for compute_oracle_mask, opening k)."""
    common = dict(kernel_sum=ksum, px_per_m=5.0)
    if args.sweep == "extent":
        return [
            (f"tau{tau:g}_k{k}", f"tau {tau:g}", dict(threshold=tau * peak), k)
            for tau in [float(x) for x in args.taus.split(",")]
            for k in [int(x) for x in args.extents.split(",")]
        ]
    cfgs = [
        ("current", "reference", dict(threshold=CURRENT_TAU * peak), 1),
        ("current_k5", "reference", dict(threshold=CURRENT_TAU * peak), 5),
    ]
    for occ in (False, True):
        for win in [int(x) for x in args.windows.split(",")]:
            for d in [float(x) for x in args.densities.split(",")]:
                kw = dict(threshold=d, density_window=win, **common)
                if occ:
                    kw["occupied_threshold"] = EMPTY_FRACTION * peak
                cfgs.append(
                    (
                        f"W{win}_D{d:g}{'_occ' if occ else ''}",
                        f"W {win} px = {win / 5:g} m{', occupied only' if occ else ''}",
                        kw,
                        1,
                    )
                )
    return cfgs


def main():
    import torch
    import torch.nn.functional as F
    import cv2
    import train_proposed_multiscene as T
    from utils.oracle_mask import compute_oracle_mask, morphological_opening

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="window", choices=["window", "extent"])
    ap.add_argument("--splits", default="train,validate")
    ap.add_argument("--frames_per_scene", type=int, default=10, help="evenly spaced over each scene's frames")
    ap.add_argument(
        "--taus",
        default="1.273,1.5,2,2.5,3",
        help="[extent] thresholds in TRUE single-person peaks (one isolated person = 1.0 in map_gt_conv)",
    )
    ap.add_argument(
        "--extents", default="1,3,5,7,9,11,13,17,25", help="[extent] odd min_extent values (px, 5 px = 1 m)"
    )
    ap.add_argument(
        "--max_extent_probe", type=int, default=41, help="[extent] largest k tried when measuring components"
    )
    ap.add_argument("--windows", default="5,9,13,25", help="[window] odd density windows (px, 5 px = 1 m)")
    ap.add_argument("--densities", default="0.25,0.5,1,2,3,5", help="[window] thresholds in persons/m^2")
    ap.add_argument("--max_scenes", type=int, default=None, help="smoke test on the first N scenes")
    ap.add_argument("--cpu", action="store_true", help="do not touch the GPU")
    ap.add_argument("--out_dir", default=os.path.join(T.src_dir, "diagnostics", "oracle_mask_extent"))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    probe = list(range(1, args.max_extent_probe + 1, 2))
    rf = dense_expert_receptive_field()
    kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(dev)
    # get_map_kernel uses GAUSSIAN_SIGMA as the covariance and normalises the peak to 1: one isolated person peaks at
    # kernel.max() = 1.0 in map_gt_conv, and a uniform crowd of D persons/px^2 gives D * kernel.sum().
    peak, ksum = float(kernel.max()), float(kernel.sum())
    configs = build_configs(args, peak, ksum)
    print(
        f"[config] sweep {args.sweep} ({len(configs)} masks) | dense expert RF {rf} px = {rf / 5:.1f} m | Gaussian "
        f"std {np.sqrt(T.GAUSSIAN_SIGMA):.2f} px | single-person peak {peak:.3f} | current threshold "
        f"{T.oracle_density_threshold():.3f} peaks (~{CURRENT_TAU / ksum * 25:.2f} persons/m^2 uniform-crowd "
        f"equivalent) | smoothing {T.MASK_SMOOTHING_KERNEL} | empty < {EMPTY_FRACTION} peaks | device {dev}"
    )

    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams = json.load(f)
    registry = []
    for split in args.splits.split(","):
        registry += T.build_scene_registry(split.strip(), 5, cams, {}, validate_setup=False)
    registry = registry[: args.max_scenes] if args.max_scenes else registry

    frame_rows = []  # one dict per (scene, frame, config)
    components = defaultdict(list)  # config name -> [(scene, area, people, max_extent or None)]
    for si, entry in enumerate(registry):
        ds = entry["dataset"]
        n = min(args.frames_per_scene, len(ds))
        idxs = np.linspace(0, len(ds) - 1, n).round().astype(int)
        for idx in idxs:
            s = ds[int(idx)]
            wld = s["wld_map_paras"].unsqueeze(0)
            h, w = int(wld[0, 3].item()), int(wld[0, 4].item())
            dots = T.create_ground_truth_heatmap(
                s["gt_world_xy"], wld, torch.zeros(2, dtype=torch.int32), bev_shape=(h, w)
            ).to(dev)
            n_people = float(dots.sum())
            if n_people == 0:
                continue
            conv = F.conv2d(dots, kernel, padding=kernel.shape[-1] // 2)
            occupied = (conv > EMPTY_FRACTION * peak).float()
            empty_share = 1.0 - F.avg_pool2d(occupied, rf, stride=1, padding=rf // 2, count_include_pad=False)
            people_in_rf = F.avg_pool2d(dots, rf, stride=1, padding=rf // 2, count_include_pad=False) * rf * rf
            n_occ = float(occupied.sum())
            dots_np = dots[0, 0].cpu().numpy()

            base_cache = {}
            for name, group, kw, k in configs:
                key = json.dumps({a: b for a, b in kw.items()}, sort_keys=True)
                if key not in base_cache:
                    base_cache[key] = compute_oracle_mask(conv, smoothing_kernel=T.MASK_SMOOTHING_KERNEL, **kw)
                base = base_cache[key]
                m = morphological_opening(base, k)
                area = float(m.sum())
                m_np = (m[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
                n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(m_np, connectivity=8)
                frame_rows.append(
                    {
                        "scene": entry["scene_name"],
                        "frame": int(idx),
                        "config": name,
                        "group": group,
                        "k": k,
                        "people": n_people,
                        "cov_scene": area / (h * w),
                        "cov_occupied": area / max(n_occ, 1.0),
                        "people_dense": float((dots * m).sum()) / n_people,
                        "n_comp": n_lab - 1,
                        "rf_empty": float((empty_share * m).sum()) / area if area else float("nan"),
                        "rf_people": float((people_in_rf * m).sum()) / area if area else float("nan"),
                    }
                )
                if n_lab <= 1:
                    continue
                max_ext = None
                if args.sweep == "extent" and k == 1:
                    max_ext = np.ones(n_lab, dtype=int)
                    for kp in probe[1:]:
                        kept = np.unique(labels[morphological_opening(base, kp)[0, 0].cpu().numpy() > 0.5])
                        kept = kept[kept > 0]
                        if len(kept) == 0:
                            break
                        max_ext[kept] = kp
                people = np.bincount(labels.ravel(), weights=dots_np.ravel(), minlength=n_lab)
                for c in range(1, n_lab):
                    components[name].append(
                        (
                            entry["scene_name"],
                            int(stats[c, cv2.CC_STAT_AREA]),
                            float(people[c]),
                            None if max_ext is None else int(max_ext[c]),
                        )
                    )
        print(f"[{si + 1}/{len(registry)}] {entry['scene_name']}: {n} frames", flush=True)

    # ---------------------------------------------------------------- aggregate (scene-balanced: mean over scenes)
    metrics = ["cov_scene", "cov_occupied", "people_dense", "n_comp", "rf_empty", "rf_people"]
    per_scene = defaultdict(lambda: defaultdict(list))
    for r in frame_rows:
        per_scene[r["config"]][r["scene"]].append(r)
    summary = []
    for name, group, kw, k in configs:
        scenes = per_scene[name]
        row = {"config": name, "group": group, "n_scenes": len(scenes)}
        for mtr in metrics:
            scene_means = [
                np.nanmean(v) if not np.all(np.isnan(v)) else np.nan
                for v in ([r[mtr] for r in rs] for rs in scenes.values())
            ]
            ok = [x for x in scene_means if not np.isnan(x)]
            row[mtr] = float(np.mean(ok)) if ok else float("nan")
            row[mtr + "_scene_std"] = float(np.std(ok)) if ok else float("nan")
        row["scenes_with_no_mask"] = int(sum(all(r["cov_scene"] == 0 for r in rs) for rs in scenes.values()))
        c = components[name]
        ppl = np.array([x[2] for x in c]) if c else np.zeros(0)
        row["n_components"] = len(c)
        row["comp_people_q"] = np.percentile(ppl, [10, 50, 90]).tolist() if len(c) else None
        row["share_comps_le2_people"] = float((ppl <= 2).mean()) if len(c) else None
        summary.append(row)

    print(
        f"\nScene-balanced means over {len({r['scene'] for r in frame_rows})} scenes (+- = std across scenes). "
        f"BEV 5 px = 1 m; dense expert RF = {rf} px."
    )
    hdr = (
        f"{'config':>18} {'cov scene%':>10} {'cov occ%':>9} {'people dense%':>16} {'comps/frame':>11} "
        f"{'ppl/comp p10/50/90':>18} {'<=2 ppl':>7} {'RF empty%':>9} {'people/RF':>9} {'no-mask scenes':>14}"
    )
    for group in dict.fromkeys(r["group"] for r in summary):
        print(f"\n[{group}]\n{hdr}")
        for row in [r for r in summary if r["group"] == group]:
            q = "/".join(f"{v:.0f}" for v in row["comp_people_q"]) if row["comp_people_q"] else "-"
            le2 = f"{100 * row['share_comps_le2_people']:.0f}%" if row["share_comps_le2_people"] is not None else "-"
            print(
                f"{row['config']:>18} {100 * row['cov_scene']:>10.2f} {100 * row['cov_occupied']:>9.1f} "
                f"{100 * row['people_dense']:>8.1f} +-{100 * row['people_dense_scene_std']:>5.1f} "
                f"{row['n_comp']:>11.1f} {q:>18} {le2:>7} {100 * row['rf_empty']:>9.1f} {row['rf_people']:>9.1f} "
                f"{row['scenes_with_no_mask']:>14}"
            )

    if args.sweep == "extent":
        print("\nk = 1 components: max_extent quantiles [10, 50, 90] (side, px, of the largest fitting square)")
        for name, group, kw, k in configs:
            ext = [x[3] for x in components[name] if x[3] is not None]
            if ext:
                print(f"{name:>18} {'/'.join(f'{v:.0f}' for v in np.percentile(ext, [10, 50, 90]))}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"oracle_mask_{args.sweep}.json")
    with open(out, "w") as f:
        json.dump(
            {
                "config": {
                    "sweep": args.sweep,
                    "rf": rf,
                    "gaussian_cov": T.GAUSSIAN_SIGMA,
                    "kernel_sum": ksum,
                    "smoothing_kernel": T.MASK_SMOOTHING_KERNEL,
                    "empty_fraction": EMPTY_FRACTION,
                    "splits": args.splits,
                    "frames_per_scene": args.frames_per_scene,
                    "masks": [{"name": n, "group": g, "kwargs": kw, "k": k} for n, g, kw, k in configs],
                },
                "summary": summary,
                "frames": frame_rows,
                "component_rows": dict(components),
            },
            f,
        )
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
