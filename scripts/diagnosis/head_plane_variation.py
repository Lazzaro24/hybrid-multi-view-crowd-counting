"""
head_plane_variation.py

How much the head plane d (d_mean in wld_map_paras: the mean world height of the annotated heads of a frame) varies
between the frames of a scene, relative to the first frame. The attention necks' reference-point cache uses the
first frame's d (utils/cache_reference.py), the homography each frame's own d. Reads one camera JSON of every
--step-th frame per scene; no images, no model. Quoted in Thesis Chapter 4 (sec:queries-anchors).

Usage (from src):
    uv run python scripts/diagnosis/head_plane_variation.py            # every 10th frame
    uv run python scripts/diagnosis/head_plane_variation.py --step 1   # every frame
"""

import os
import glob
import json
import argparse
import statistics

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(src_dir, "datasets", "SynMVCrowd", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=10, help="use every step-th frame of a scene")
    args = ap.parse_args()

    rows = []
    for split in ("train", "validate", "test"):
        split_dir = os.path.join(DATA, split)
        for scene in sorted(os.listdir(split_dir)):
            scene_dir = os.path.join(split_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            snaps = sorted((s for s in os.listdir(scene_dir) if s.isdigit()), key=int)
            heads = []
            for snap in snaps[:: args.step]:
                jsons = glob.glob(os.path.join(scene_dir, snap, "jsons", "*.json"))
                if jsons:
                    with open(jsons[0]) as f:
                        heads.append(json.load(f)["wld_map_paras"][7])  # d_mean
            if heads:
                rows.append((max(abs(h - heads[0]) for h in heads), split, scene))

    rows.sort()
    devs = [r[0] for r in rows]
    print(f"{len(rows)} scenes, every {args.step}th frame")
    print(f"max |d - d(first frame)| per scene: median {statistics.median(devs):.2f} m")
    for thr in (0.5, 1.0):
        print(f"  scenes above {thr} m: {sum(d > thr for d in devs)}")
    print("largest:", ", ".join(f"{split}/{scene} {dev:.2f} m" for dev, split, scene in rows[-5:]))


if __name__ == "__main__":
    main()
