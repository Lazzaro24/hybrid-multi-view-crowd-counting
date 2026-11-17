"""
verify_exclusions.py

Independent check of excluded_frames.json against the raw data, across ALL splits
(train / validate / test). Answers four questions:

  1. ID SEMANTICS: do the IDs in excluded_frames.json really correspond to snapshot FOLDER
     names, and are those frames really corrupted (r0 < MIN_VALID_R0)?
  2. COMPLETENESS: does any frame NOT in the list have r0 < MIN_VALID_R0? (This is also the
     test-split sweep -- the original log only contained train/validate entries.)
  3. Are there scenes that are corrupted wholesale (like scene_71) rather than per frame?
  4. Frame-level assumption: with --cams_per_frame N, the first N cameras of each snapshot are
     compared, so a disagreement between cameras (per-camera rather than per-frame corruption)
     shows up.

Usage (from src/):
    python utils/verify_exclusions.py
    python utils/verify_exclusions.py --cams_per_frame 3 --splits test
Exit code is 1 if any inconsistency is found.
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(src_dir)

from utils.exclusions import load_excluded_scenes, load_excluded_frames

MIN_VALID_R0 = 1.0  # keep in sync with synmvcrowd.py


def read_r0(scene_dir, snap, cam_file):
    with open(os.path.join(scene_dir, snap, "jsons", cam_file), "r") as f:
        return json.load(f)["wld_map_paras"][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "validate", "test"])
    parser.add_argument("--cams_per_frame", type=int, default=1)
    args = parser.parse_args()

    data_root = os.path.join(src_dir, "datasets", "SynMVCrowd", "data")
    excluded_frames = load_excluded_frames()
    excluded_scenes = load_excluded_scenes()

    problems = []
    seen_scenes = set()

    for split in args.splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            print(f"[skip] {split_dir} not found")
            continue
        for scene_name in sorted(os.listdir(split_dir)):
            scene_dir = os.path.join(split_dir, scene_name)
            if not os.path.isdir(scene_dir):
                continue
            seen_scenes.add(scene_name)
            snaps = sorted(
                [d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d)) and d.isdigit()], key=int
            )
            bad_listed = excluded_frames.get(scene_name, set())
            listed_missing = bad_listed - {int(s) for s in snaps}
            if listed_missing:
                problems.append(f"{split}/{scene_name}: excluded IDs not on disk: {sorted(listed_missing)}")

            found_bad, disagree = set(), []
            for snap in tqdm(snaps, desc=f"{split}/{scene_name}", leave=False):
                cam_dir = os.path.join(scene_dir, snap, "jsons")
                if not os.path.isdir(cam_dir):
                    continue
                cams = sorted(
                    [f for f in os.listdir(cam_dir) if f.endswith(".json")], key=lambda x: int(x.split(".")[0])
                )[: args.cams_per_frame]
                flags = [read_r0(scene_dir, snap, c) < MIN_VALID_R0 for c in cams]
                if any(flags):
                    found_bad.add(int(snap))
                if any(flags) and not all(flags):
                    disagree.append(int(snap))

            frac = len(found_bad) / max(len(snaps), 1)
            tag = ""
            if scene_name in excluded_scenes:
                tag = "  [in excluded_scenes.json]"
            elif frac > 0.5:
                tag = "  <-- WHOLE-SCENE corruption? consider adding to excluded_scenes.json"
                problems.append(f"{split}/{scene_name}: {frac:.0%} of frames corrupted -- scene-level issue")
            print(
                f"{split}/{scene_name}: {len(snaps)} snapshots | corrupted on disk: {len(found_bad)} "
                f"| listed: {len(bad_listed)}{tag}"
            )

            if scene_name in excluded_scenes:
                continue  # scene-level exclusion supersedes per-frame list
            unlisted = found_bad - bad_listed
            healthy_listed = bad_listed - found_bad
            if unlisted:
                problems.append(f"{split}/{scene_name}: corrupted but NOT excluded: {sorted(unlisted)}")
            if healthy_listed:
                problems.append(
                    f"{split}/{scene_name}: excluded but looks HEALTHY (wrong ID semantics?): {sorted(healthy_listed)}"
                )
            if disagree:
                problems.append(
                    f"{split}/{scene_name}: cameras disagree on corruption at snapshots "
                    f"{sorted(disagree)} (per-camera, not per-frame, corruption)"
                )

    unknown_keys = [k for k in excluded_frames if k not in seen_scenes]
    if unknown_keys:
        problems.append(f"excluded_frames.json keys matching no scene on disk: {unknown_keys}")

    print("\n" + "=" * 70)
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("OK: excluded_frames.json matches the raw data exactly for all checked scenes.")


if __name__ == "__main__":
    main()
