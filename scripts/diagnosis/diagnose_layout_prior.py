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
from visualize_experts import build_filtered_samples, pearson


def main():
    import torch
    import torch.nn.functional as F
    import train_proposed_multiscene as T

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene_14")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--skip_multiples_of", type=int, default=10)
    ap.add_argument("--val_seed", type=int, default=1234)
    ap.add_argument("--max_rounds", type=int, default=60)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams = json.load(f)
    reg = T.build_scene_registry("train", args.k, cams, {}, validate_setup=False)
    sel = [e for e in reg if e["scene_name"] == args.scene]
    samples = build_filtered_samples(T, sel, args.frames, args.val_seed, args.skip_multiples_of, 0, args.max_rounds)
    kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(dev)
    pad = kernel.shape[-1] // 2

    def dens(src, wld, hw):
        tp = T.create_ground_truth_heatmap(src["gt_world_xy"], wld, hw, bev_shape=T.BEV_SHAPE).to(dev)
        return F.conv2d(tp, kernel, padding=pad)[0, 0].cpu().numpy()

    corrs = []
    for i, s in enumerate(samples):
        wld = s["wld_map_paras"].to(dev)  # assumes the same world map for all frames of a scene
        for p in range(s["hw_random"].shape[-1]):
            hw = s["hw_random"].to(dev)[:, :, p]
            Ti = dens(s, wld, hw)
            prior = np.mean([dens(o, wld, hw) for j, o in enumerate(samples) if j != i], axis=0)
            corrs.append(pearson(prior, Ti))
    c = np.array(corrs)
    print(
        f"[layout prior, leave-one-out over {len(samples)} unseen frames] corr(prior, true density): "
        f"mean {np.nanmean(c):.2f}, median {np.nanmedian(c):.2f}, n={len(c)}"
    )
    print("compare with the model on the same frames: dense 0.24, sparse 0.36")


if __name__ == "__main__":
    main()
