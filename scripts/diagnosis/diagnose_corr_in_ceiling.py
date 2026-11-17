"""
diagnose_corr_in_ceiling.py

No model involved. For the true density T (same kernel/threshold as training), computes:
  1. per-patch pixel-level spread of T INSIDE the oracle zone (std/mean over zone pixels, not the per-patch-mean CV
     from diagnose_zone_threshold.py -- this is finer-grained: is T flat pixel-to-pixel within a zone, not just
     zone-to-zone?).
  2. a "ceiling" corr_in: blur T itself with box filters of increasing size, correlate the blurred version against
     the ORIGINAL T, restricted to zone pixels. This is the best corr_in ANY prediction with that receptive field
     could achieve, since it starts from the perfect answer and only loses information to blur.
If ceiling corr_in is already near the model's measured corr_in (0.05-0.08) at a blur radius <= the dense expert's
receptive field, the flatness is intrinsic to the zone/kernel/threshold, not a training deficiency.

Usage:
    uv run scripts/diagnosis/diagnose_corr_in_ceiling.py --blur_sizes 1,3,5,7,9,13,17,21,25
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
from visualize_experts import pearson


def main():
    import torch
    import torch.nn.functional as F
    import train_proposed_multiscene as T

    ap = argparse.ArgumentParser()
    ap.add_argument("--val_scenes", default="all", help="comma-separated or 'all'")
    ap.add_argument("--frames_per_scene", type=int, default=10)
    ap.add_argument("--val_seed", type=int, default=1234)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--blur_sizes",
        default="1,3,5,7,9,13,17,21,25",
        help="odd box-filter sizes; 1 = no blur (sanity check, corr must be 1.0)",
    )
    ap.add_argument("--min_zone_pixels", type=int, default=50)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizes = [int(x) for x in args.blur_sizes.split(",")]
    assert all(s == 1 or s % 2 == 1 for s in sizes), "blur sizes must be odd (or 1 for no blur)"

    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams = json.load(f)
    reg = T.build_scene_registry("validate", args.k, cams, {}, validate_setup=False)
    names = [n.strip() for n in args.val_scenes.split(",")]
    sel = list(reg) if names[0].lower() == "all" else [e for e in reg if e["scene_name"] in names]
    samples, _ = T.build_cached_val_samples(sel, args.frames_per_scene, val_seed=args.val_seed)

    kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(dev)
    pad = kernel.shape[-1] // 2
    thr = T.oracle_density_threshold()
    print(
        f"[units] zone threshold {thr:.4f}   [receptive-field note] this script does not know the dense expert's "
        f"actual dilation/kernel config -- compare the numbers below against it yourself; if unsure, treat blur "
        f"size ~9-21 as the relevant range for dilation-2/4, kernel-3 stacks."
    )

    per_scene = {}
    pixel_cv_all, corr_by_size = [], {s: [] for s in sizes}
    n_zones = 0
    for s in samples:
        wld = s["wld_map_paras"].to(dev)
        for p in range(s["hw_random"].shape[-1]):
            hw = s["hw_random"].to(dev)[:, :, p]
            tp = T.create_ground_truth_heatmap(s["gt_world_xy"], wld, hw, bev_shape=T.BEV_SHAPE).to(dev)
            conv = F.conv2d(tp, kernel, padding=pad)
            gm = (T.oracle_mask_from_conv(conv) > 0.5)[0, 0]
            Tn = conv[0, 0]
            if int(gm.sum()) < args.min_zone_pixels:
                continue
            n_zones += 1
            zvals = Tn[gm].cpu().numpy()
            cv = float(zvals.std() / max(zvals.mean(), 1e-6))
            pixel_cv_all.append(cv)
            per_scene.setdefault(s["scene_name"], {"cv": [], **{sz: [] for sz in sizes}})
            per_scene[s["scene_name"]]["cv"].append(cv)
            Tn_np = Tn.cpu().numpy()
            gm_np = gm.cpu().numpy()
            for sz in sizes:
                if sz == 1:
                    blurred = Tn
                else:
                    blurred = F.avg_pool2d(conv, kernel_size=sz, stride=1, padding=sz // 2)[0, 0]
                c = pearson(blurred.cpu().numpy()[gm_np], Tn_np[gm_np])
                corr_by_size[sz].append(c)
                per_scene[s["scene_name"]][sz].append(c)

    print(f"\n[data] {n_zones} patches had >= {args.min_zone_pixels} zone pixels\n")
    print("[1] per-pixel spread of T INSIDE the zone (std/mean over zone pixels, per patch, then averaged)")
    print(
        f"    mean pixel CV = {np.nanmean(pixel_cv_all):.3f}   (compare to the earlier PER-PATCH-MEAN CV of ~0.07 --"
        f" this is a different, finer statistic)"
    )

    print("\n[2] CEILING corr_in: corr(box-blurred T, original T), zone pixels only")
    print("    blur size 1x1 must read 1.00 -- sanity check that the pipeline is correct.")
    print(f"    {'blur size':>10} {'n':>6} {'ceiling corr_in':>16}")
    for sz in sizes:
        v = np.array(corr_by_size[sz], dtype=np.float64)
        print(f"    {sz:>10} {np.isfinite(v).sum():>6} {np.nanmean(v):>16.3f}")
    print("\n    for reference, MEASURED model corr_in from the tier0 run: dense ~0.05-0.08, unified ~0.06-0.07")

    print("\n[3] per-scene breakdown (ceiling corr_in at a few sizes)")
    show_sizes = [s for s in (1, 5, 9, 17, 25) if s in sizes] or sizes[:3]
    hdr = f"    {'scene':<10}{'cv':>8}" + "".join(f"{'b' + str(s):>8}" for s in show_sizes)
    print(hdr)
    for scene, d in sorted(per_scene.items()):
        line = f"    {scene:<10}{np.nanmean(d['cv']):>8.3f}"
        for sz in show_sizes:
            line += f"{np.nanmean(d[sz]):>8.3f}"
        print(line)


if __name__ == "__main__":
    main()
