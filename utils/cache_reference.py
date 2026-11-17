"""
cache_reference.py

Caches master_refs.pt for every scene, DENSELY over all 50 raw cameras
(indexed by camera_id, 0-49) -- NOT restricted to whatever subset
optimal_cameras_filtered.json picks for a given k. That filtering happens
at load/training time (master_refs[cam_id_list]), not here -- caching a
compact subset instead of the full 50 caused an IndexError once already,
since SynMVCrowdDataset indexes camera_paras directly by camera_id.

Per camera, compute_reference_points (utils/reference_points.py) gets that camera's z_sign from
z_sign_corrections_full.json (every camera on disk per scene, since all 50 are cached); the off-axis threshold stays
at its default (3.0) for every camera. Splits: train, validate, test.
"""

import os
import sys
import json
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
from reference_points import compute_reference_points
from utils.exclusions import load_excluded_scenes


def load_z_sign_corrections_full(utils_dir, filename="z_sign_corrections_full.json"):
    path = os.path.join(utils_dir, filename)
    if not os.path.exists(path):
        print(
            f"[warning] {path} not found -- proceeding with NO z-sign corrections "
            f"(every camera defaults to z_sign=1.0, likely wrong for some)."
        )
        return {}
    with open(path, "r") as f:
        return json.load(f)


def anchor_offsets(d_delta, num_anchors, below, above):
    """Anchor heights relative to d_mean (= mean HEAD height of the frame): from the feet of people standing `below`
    metres lower than average (-d_delta - below) to the heads of people standing `above` metres higher (+above)."""
    return torch.linspace(-d_delta - below, above, num_anchors)


def cache_all_scenes(
    name="master_refs.pt",
    z_anchors=None,
    below=2.5,
    above=2.5,
    fp16=False,
    scenes=None,
    z_sign_file="z_sign_corrections_full.json",
    overwrite=False,
):
    """z_anchors=None reproduces the original cache (4 anchors at linspace(-d_delta, 1.8 + d_delta) + d_mean).
    Scenes in excluded_scenes.json are skipped, and so are scenes that already have `name` unless overwrite=True."""
    data_root = os.path.join(src_dir, "datasets", "SynMVCrowd", "data")
    utils_dir = os.path.join(src_dir, "utils")
    splits = ["train", "validate", "test"]
    excluded = load_excluded_scenes()

    z_sign_corrections = load_z_sign_corrections_full(utils_dir, z_sign_file)

    for split in splits:
        split_dir = os.path.join(data_root, split)

        if not os.path.exists(split_dir):
            print(f"Skipping {split} directory (not found): {split_dir}")
            continue

        scene_folders = [
            os.path.join(split_dir, d)
            for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d)) and (scenes is None or d in scenes)
        ]
        print(f"\nFound {len(scene_folders)} scenes in {split.upper()}. Starting offline caching...")

        for scene_dir in scene_folders:
            scene_name = os.path.basename(scene_dir)
            if scene_name in excluded:
                print(f"Skipping {split}/{scene_name}: excluded ({excluded[scene_name][:60]}...)")
                continue
            if not overwrite and os.path.exists(os.path.join(scene_dir, name)):
                print(f"Skipping {split}/{scene_name}: {name} exists (--overwrite to rebuild)")
                continue
            print(f"Processing: {split}/{scene_name}")

            # camera_ids=None: ALL 50 cameras, dense, indexed by raw camera_id.
            # Do not restrict this to optimal_cameras_filtered.json's picks --
            # that selection happens later, by indexing this dense tensor.
            temp_dataset = SynMVCrowdDataset(scene_dir=scene_dir, camera_ids=None, is_caching=True)

            sample = temp_dataset[0]
            wld_paras = sample["wld_map_paras"]
            cam_paras = sample["camera_paras"]

            scene_corrections = z_sign_corrections.get(scene_name, {})
            num_cameras = cam_paras.shape[0]
            # Assumes camera_paras row i corresponds to raw camera_id i (the same
            # assumption the dense-by-camera-id caching convention already relies
            # on). Worth a one-time sanity check if this hasn't been explicitly
            # verified: confirm SynMVCrowdDataset(camera_ids=None) really returns
            # cameras in numeric id order (0, 1, 2, ..., 49), not e.g. a string
            # sort of filenames ("0", "1", "10", "11", ..., "2", ...) which would
            # silently misalign every z_sign lookup below.
            z_sign_list = [scene_corrections.get(str(i), 1.0) for i in range(num_cameras)]

            # NOTE: the anchors are placed relative to frame 0's d_mean; the homography uses every frame's own d_mean
            z_offsets = None if z_anchors is None else anchor_offsets(float(wld_paras[10]), z_anchors, below, above)
            master_refs = compute_reference_points(
                wld_map_paras=wld_paras,
                camera_paras=cam_paras,
                img_size=(720, 1280),
                num_z_anchors=4,
                z_sign=z_sign_list,
                z_offsets=z_offsets,
                # offaxis_threshold: reference_points.py's default (3.0)
            )
            if fp16:  # normalised image coords: fp16 error ~5e-4 (~0.3 px at 1280 px); the -10 sentinel is exact
                master_refs = master_refs.half()

            save_path = os.path.join(scene_dir, name)
            torch.save(master_refs, save_path)
            print(f"Saved master tensor {master_refs.shape} {master_refs.dtype} to {save_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Cache per-scene reference points for all 50 cameras.")
    ap.add_argument("--name", default="master_refs.pt", help="output file name inside each scene folder")
    ap.add_argument(
        "--z_anchors",
        type=int,
        default=None,
        help="number of anchors from d_mean - d_delta - below to d_mean + above (default: original 4)",
    )
    ap.add_argument("--below", type=float, default=2.5, help="terrain margin below the average head plane [m]")
    ap.add_argument("--above", type=float, default=2.5, help="terrain margin above the average head plane [m]")
    ap.add_argument("--fp16", action="store_true", help="store the cache in float16 (halves disk and RAM)")
    ap.add_argument("--scenes", default=None, help="comma-separated scene names (default: all)")
    ap.add_argument("--z_sign_file", default="z_sign_corrections_full.json")
    ap.add_argument("--overwrite", action="store_true", help="rebuild scenes that already have --name")
    a = ap.parse_args()
    cache_all_scenes(
        a.name,
        a.z_anchors,
        a.below,
        a.above,
        a.fp16,
        None if a.scenes is None else set(a.scenes.split(",")),
        a.z_sign_file,
        a.overwrite,
    )
