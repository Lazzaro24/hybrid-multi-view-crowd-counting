"""
regenerate_optimal_cameras_filtered.py

Rebuilds optimal_cameras.json with bad cameras (foldback OR uncertain,
under check_offaxis_combined.py's classification) excluded from the K-means
candidate pool BEFORE clustering -- not swapped out after the fact, and not
via any iterative re-check loop. Foldback status is a per-camera geometric
property, independent of which other cameras end up selected, so a single
pass (exclude, then cluster once per scene per k) is sufficient: there is
no scenario where a camera's status changes based on what else gets picked.

PREREQUISITE: run the full-pool sweep first, so every raw camera (not just
the ones a prior K-means run happened to select) has a known status:

    uv run python .\\utils\\run_offaxis_sweep.py --script check_offaxis_combined.py \\
        --all_cameras --threshold 2.5 --closing_iterations 1

That produces offaxis_combined_allcams.json, which this script reads.

Output: optimal_cameras_filtered.json (does NOT overwrite the original --
compare the two before deciding to replace it, especially checking whether
any scene's clean-candidate count is too thin for k=10).

Usage:
    python regenerate_optimal_cameras_filtered.py
"""

import os
import json
import numpy as np
from sklearn.cluster import KMeans

BAD_STATUSES = {"foldback", "uncertain"}


def load_exclusions(allcams_path):
    """scene_name -> set of excluded camera ids (int)"""
    with open(allcams_path, "r") as f:
        data = json.load(f)
    exclusions = {}
    for scene_name, cam_map in data.items():
        excluded = {int(cam_id) for cam_id, entry in cam_map.items() if entry.get("status") in BAD_STATUSES}
        exclusions[scene_name] = excluded
    return exclusions


def get_optimal_cameras(scene_dir, num_cameras, excluded_cam_ids=None):
    excluded_cam_ids = excluded_cam_ids or set()

    snapshot_folders = [d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d)) and d.isdigit()]
    if not snapshot_folders:
        print(f"No valid snapshot folders found in: {scene_dir}")
        return []
    first_snap = sorted(snapshot_folders, key=lambda x: int(x))[0]
    snapshot_0_json_dir = os.path.join(scene_dir, str(first_snap), "jsons")
    if not os.path.exists(snapshot_0_json_dir):
        snapshot_0_json_dir = os.path.join(scene_dir, str(first_snap), "json")
    if not os.path.exists(snapshot_0_json_dir):
        print(f"Directory not found: {snapshot_0_json_dir}")
        return []

    camera_positions = {}
    for cam_id in range(51):  # cameras 0-50 inclusive, 51 total
        if cam_id in excluded_cam_ids:
            continue
        json_path = os.path.join(snapshot_0_json_dir, f"{cam_id}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                tvec = np.array(data["tvec"]).flatten()
                camera_positions[cam_id] = tvec

    if len(camera_positions) < num_cameras:
        print(
            f"  !! WARNING: only {len(camera_positions)} clean candidates available "
            f"for k={num_cameras} in {scene_dir} (needed {num_cameras}). "
            f"Falling back to using ALL clean candidates -- spatial coverage may suffer."
        )
        return sorted(camera_positions.keys())

    cam_ids = list(camera_positions.keys())
    positions = np.array(list(camera_positions.values()))

    kmeans = KMeans(n_clusters=num_cameras, random_state=42, n_init=10)
    kmeans.fit(positions)
    centroids = kmeans.cluster_centers_

    optimal_cam_ids = []
    for centroid in centroids:
        distances = np.linalg.norm(positions - centroid, axis=1)
        closest_idx = np.argmin(distances)
        while cam_ids[closest_idx] in optimal_cam_ids:
            distances[closest_idx] = np.inf
            closest_idx = np.argmin(distances)
        optimal_cam_ids.append(cam_ids[closest_idx])

    return sorted(optimal_cam_ids)


if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(src_dir, "datasets", "SynMVCrowd", "data")
    allcams_path = os.path.join(src_dir, "offaxis_combined_allcams.json")

    if not os.path.exists(allcams_path):
        print(f"Missing {allcams_path} -- run the --all_cameras sweep first (see docstring).")
        raise SystemExit(1)

    exclusions = load_exclusions(allcams_path)

    k_values = [3, 4, 5, 6, 7, 8, 9, 10]
    master_config = {}
    thin_scenes = []  # scenes where exclusion left too few candidates for some k

    splits = ["train", "validate", "test"]
    for split in splits:
        base_data_path = os.path.join(data_root, split)
        if not os.path.exists(base_data_path):
            print(f"Skipping {split} (not found): {base_data_path}")
            continue

        scenes = sorted(d for d in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, d)))
        print(f"\n--- {split.upper()}: {len(scenes)} scenes ---")

        for scene in scenes:
            scene_path = os.path.join(base_data_path, scene)
            excluded = exclusions.get(scene, set())
            n_clean = 50 - len(excluded)
            print(f"Processing {split}/{scene}... ({len(excluded)} excluded, {n_clean} clean candidates)")

            master_config[scene] = {}
            for k in k_values:
                if n_clean < k:
                    thin_scenes.append((scene, k, n_clean))
                best_cams = get_optimal_cameras(scene_path, num_cameras=k, excluded_cam_ids=excluded)
                master_config[scene][str(k)] = best_cams

    out_path = os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")
    with open(out_path, "w") as f:
        json.dump(master_config, f, indent=4)

    print("\n" + "=" * 70)
    print(f"Saved filtered config to: {out_path}")
    if thin_scenes:
        print(
            f"\n!! {len(thin_scenes)} (scene, k, clean_candidates) cases had FEWER clean "
            f"candidates than k -- these fell back to using every clean camera available, "
            f"which may not give good spatial spread. Review before using for the ablation study:"
        )
        for scene, k, n_clean in thin_scenes:
            print(f"    {scene}: k={k} needed, only {n_clean} clean candidates")
    else:
        print("Every scene had enough clean candidates for every k value tested (3-10).")
    print("=" * 70)
