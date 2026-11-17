"""
build_weather_index.py

Scans every scene/snapshot of every split, reads ONE camera JSON per snapshot and stores all its
top-level scalar fields (weather, and anything time-of-day-like that exists) in weather_index.json.
Then prints a summary that answers:
  1. Is weather constant within each scene, or does it vary per frame?
  2. How many scenes / frames per weather value, per split?
  3. Which other scalar fields exist and vary (candidates for time of day / night)?
  4. (--verify_cams N) Do the first N cameras of a snapshot agree?

Usage (from src/):
    python utils/build_weather_index.py
    python utils/build_weather_index.py --verify_cams 3 --splits train validate test
"""

import os
import json
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)

TIME_HINTS = ("time", "hour", "day", "night", "light", "clock", "sun", "daytime")


def scalars_of(data):
    return {k: v for k, v in data.items() if isinstance(v, (str, int, float, bool)) or v is None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(src_dir, "datasets", "SynMVCrowd", "data"))
    parser.add_argument("--splits", nargs="+", default=["train", "validate", "test"])
    parser.add_argument("--verify_cams", type=int, default=1)
    parser.add_argument("--out", default=os.path.join(current_dir, "weather_index.json"))
    args = parser.parse_args()

    index, scene_split = {}, {}
    cam_disagreements = []
    for split in args.splits:
        split_dir = os.path.join(args.data_root, split)
        if not os.path.isdir(split_dir):
            print(f"[skip] {split_dir} not found")
            continue
        for scene in sorted(os.listdir(split_dir)):
            scene_dir = os.path.join(split_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            scene_split[scene] = split
            index[scene] = {}
            snaps = sorted([d for d in os.listdir(scene_dir) if d.isdigit()], key=int)
            for snap in tqdm(snaps, desc=f"{split}/{scene}", leave=False):
                jdir = os.path.join(scene_dir, snap, "jsons")
                if not os.path.isdir(jdir):
                    continue
                cams = sorted([f for f in os.listdir(jdir) if f.endswith(".json")], key=lambda x: int(x.split(".")[0]))[
                    : max(args.verify_cams, 1)
                ]
                first = None
                for i, cam in enumerate(cams):
                    with open(os.path.join(jdir, cam), "r") as f:
                        sc = scalars_of(json.load(f))
                    if i == 0:
                        first = sc
                    elif sc.get("weather") != first.get("weather"):
                        cam_disagreements.append((scene, snap, cams[0], cam))
                if first is not None:
                    index[scene][snap] = first

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(index, f)
    print(f"\nWrote {args.out} ({sum(len(v) for v in index.values())} snapshots, {len(index)} scenes)")

    # ---- 1. field overview
    values = defaultdict(Counter)
    for scene in index.values():
        for rec in scene.values():
            for k, v in rec.items():
                values[k][v] += 1
    print("\n=== Top-level scalar fields (across all snapshots) ===")
    for k, c in sorted(values.items()):
        hint = "   <-- time-of-day candidate?" if any(h in k.lower() for h in TIME_HINTS) else ""
        sample = ", ".join(f"{v!r}:{n}" for v, n in c.most_common(6))
        print(f"  {k}: {len(c)} distinct{hint}   [{sample}{' ...' if len(c) > 6 else ''}]")

    # ---- 2. weather within scenes
    print("\n=== Weather per scene ===")
    varying, constant = [], []
    per_scene = {}
    for scene, snaps in index.items():
        c = Counter(rec.get("weather", "UNKNOWN") for rec in snaps.values())
        per_scene[scene] = c
        (constant if len(c) == 1 else varying).append(scene)
        desc = "constant" if len(c) == 1 else "VARIES"
        print(f"  {scene:<10}{scene_split[scene]:<10}{sum(c.values()):>4} frames  {desc:<9}{dict(c.most_common())}")
    print(f"\nVERDICT: weather is constant within {len(constant)} scenes and varies within {len(varying)} scenes.")

    # ---- 3. counts per split
    print("\n=== Scenes / frames per weather value, per split ===")
    for split in args.splits:
        scenes = [s for s in index if scene_split.get(s) == split]
        if not scenes:
            continue
        frames = Counter()
        for s in scenes:
            frames.update(per_scene[s])
        scene_by_w = Counter()
        for s in scenes:
            for w in per_scene[s]:
                scene_by_w[w] += 1
        print(
            f"  {split}: frames {dict(frames.most_common())} | scenes containing each: {dict(scene_by_w.most_common())}"
        )

    # ---- 4. other fields varying within scenes (time-of-day candidates)
    print("\n=== Other scalar fields that vary WITHIN a scene ===")
    any_found = False
    for k in values:
        if k == "weather":
            continue
        n_var = sum(1 for snaps in index.values() if len({rec.get(k) for rec in snaps.values()}) > 1)
        if n_var:
            any_found = True
            print(f"  {k}: varies within {n_var}/{len(index)} scenes")
    if not any_found:
        print("  none")

    if args.verify_cams > 1:
        print(f"\n=== Camera agreement (first {args.verify_cams} cameras per snapshot) ===")
        print(
            f"  {len(cam_disagreements)} disagreement(s)"
            + (f", e.g. {cam_disagreements[:3]}" if cam_disagreements else "")
        )


if __name__ == "__main__":
    main()
