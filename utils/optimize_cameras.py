import os
import json
import numpy as np
from sklearn.cluster import KMeans


def get_optimal_cameras(scene_dir, num_cameras):
    """
    Simulates optimal real-world camera placement by analyzing the 3D coordinates
    of all 50 synthetic cameras and selecting the most spatially distributed set.
    """
    # >>> NEW LOGIC: Dynamically find the first valid frame folder <<<
    snapshot_folders = [d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d)) and d.isdigit()]

    if not snapshot_folders:
        print(f"No valid snapshot folders found in: {scene_dir}")
        return []

    first_snap = sorted(snapshot_folders, key=lambda x: int(x))[0]

    snapshot_0_json_dir = os.path.join(scene_dir, str(first_snap), "jsons")

    # Fallback in case the folder is named 'json' instead of 'jsons'
    if not os.path.exists(snapshot_0_json_dir):
        snapshot_0_json_dir = os.path.join(scene_dir, str(first_snap), "json")

    if not os.path.exists(snapshot_0_json_dir):
        print(f"Directory not found: {snapshot_0_json_dir}")
        return []

    camera_positions = {}

    # 1. Read the 3D location (tvec) of all cameras
    for cam_id in range(51):
        json_path = os.path.join(snapshot_0_json_dir, f"{cam_id}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                tvec = np.array(data["tvec"]).flatten()
                camera_positions[cam_id] = tvec

    if len(camera_positions) < num_cameras:
        return list(camera_positions.keys())

    cam_ids = list(camera_positions.keys())
    positions = np.array(list(camera_positions.values()))

    # 2. Cluster the 3D space into 'num_cameras' distinct physical zones
    kmeans = KMeans(n_clusters=num_cameras, random_state=42, n_init=10)
    kmeans.fit(positions)
    centroids = kmeans.cluster_centers_

    optimal_cam_ids = []

    # 3. Find the real camera closest to the center of each zone
    for centroid in centroids:
        distances = np.linalg.norm(positions - centroid, axis=1)
        closest_idx = np.argmin(distances)

        # Ensure we don't pick the same camera twice
        while cam_ids[closest_idx] in optimal_cam_ids:
            distances[closest_idx] = np.inf
            closest_idx = np.argmin(distances)

        optimal_cam_ids.append(cam_ids[closest_idx])

    return sorted(optimal_cam_ids)


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_root = os.path.join(project_root, "src", "datasets", "SynMVCrowd", "data")

    # We want to pre-calculate setups for models using 3 to 10 cameras
    k_values = [3, 4, 5, 6, 7, 8, 9, 10]
    master_config = {}

    # Loop through all available splits
    splits = ["train", "validate", "test"]

    for split in splits:
        base_data_path = os.path.join(data_root, split)

        if not os.path.exists(base_data_path):
            print(f"Skipping {split} directory (not found): {base_data_path}")
            continue

        # Dynamically fetch all directories inside the split folder
        scenes = [d for d in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, d))]
        scenes.sort()  # Sort alphabetically to keep processing consistent

        print(f"\n--- Calculating Optimal Camera Distributions for {len(scenes)} {split.upper()} scenes ---")

        for scene in scenes:
            scene_path = os.path.join(base_data_path, scene)
            master_config[scene] = {}
            print(f"Processing {split}/{scene}...")

            for k in k_values:
                best_cams = get_optimal_cameras(scene_path, num_cameras=k)
                master_config[scene][str(k)] = best_cams

    config_path = os.path.join(project_root, "src", "datasets", "SynMVCrowd", "optimal_cameras.json")

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(master_config, f, indent=4)

    print("\n====================================================")
    print("OPTIMIZATION COMPLETE!")
    print(f"Saved master configuration to: {config_path}")
    print("====================================================")
