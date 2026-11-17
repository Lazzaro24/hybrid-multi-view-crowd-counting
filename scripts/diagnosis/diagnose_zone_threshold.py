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


def main():
    import torch
    import torch.nn.functional as F
    import train_proposed_multiscene as T
    from utils.oracle_mask import compute_oracle_mask

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene_14,scene_3,scene_36,scene_42,scene_18,scene_49")
    ap.add_argument("--frames_per_scene", type=int, default=3)
    ap.add_argument("--thresholds", default="1.273,1.6,2.0,2.5,3.0")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams = json.load(f)
    names = [n.strip() for n in args.scenes.split(",")]
    regs = T.build_scene_registry("train", 5, cams, {}, validate_setup=False) + T.build_scene_registry(
        "validate", 5, cams, {}, validate_setup=False
    )
    sel = [e for e in regs if e["scene_name"] in names]
    samples, _ = T.build_cached_val_samples(sel, args.frames_per_scene, val_seed=1234)
    k = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(dev)
    KMP, pad, sk = float(k.sum()), k.shape[-1] // 2, T.MASK_SMOOTHING_KERNEL
    ths = [float(x) for x in args.thresholds.split(",")]
    stats = {t: {"cov": [], "frac": [], "mean_in": [], "area": [], "mass_in": []} for t in ths}
    for s in samples:
        wld = s["wld_map_paras"].to(dev)
        for p in range(s["hw_random"].shape[-1]):
            hw = s["hw_random"].to(dev)[:, :, p]
            tp = T.create_ground_truth_heatmap(s["gt_world_xy"], wld, hw, bev_shape=T.BEV_SHAPE).to(dev)
            conv = F.conv2d(tp, k, padding=pad)
            tot = float(conv.sum() / KMP)
            for t in ths:
                gm = compute_oracle_mask(
                    conv, t, smoothing_kernel=sk, min_extent=T.ORACLE_MIN_EXTENT, min_area=T.ORACLE_MIN_AREA
                )
                area = float(gm.sum())
                if area < 1:
                    continue
                mass_in = float((conv * gm).sum() / KMP)
                st = stats[t]
                st["cov"].append(100 * float(gm.mean()))
                st["frac"].append(mass_in / max(tot, 1e-6))
                st["mean_in"].append(float((conv * gm).sum() / area))
                st["area"].append(area)
                st["mass_in"].append(mass_in)
    print(
        f"{'thr':>6} {'n':>4} {'cov%':>6} {'mass in zone %':>15} {'mean T in zone':>15} {'CV of mean T':>13} {'corr(area,mass)':>16}"
    )
    for t in ths:
        st = stats[t]
        if len(st["area"]) < 3:
            print(f"{t:>6.3f} {len(st['area']):>4}  (too few zones)")
            continue
        m = np.array(st["mean_in"])
        c = np.corrcoef(st["area"], st["mass_in"])[0, 1]
        print(
            f"{t:>6.3f} {len(m):>4} {np.mean(st['cov']):>6.1f} {100 * np.mean(st['frac']):>15.1f} "
            f"{m.mean():>15.2f} {m.std() / m.mean():>13.2f} {c:>16.2f}"
        )


if __name__ == "__main__":
    main()
