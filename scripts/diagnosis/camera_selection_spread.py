"""
camera_selection_spread.py

How evenly the K-means view selection spreads the cameras around each SynMVCrowd scene (thesis Section 5.3).

utils/optimize_cameras.py clusters the extrinsic translation vectors tvec, which are the world origin in camera
coordinates, not the camera centres C = -R^T tvec. This script compares the stored selection
(datasets/SynMVCrowd/optimal_cameras_filtered.json, k = 5) with a K-means clustering of the camera centres over the
same candidate pool (fold-back cameras of utils/offaxis_combined_allcams.json removed, same seed and initializations).

Per scene and selection: the largest angular gap between the selected cameras around the scene centre (mean xy of
the pool's camera centres; 360 deg = all cameras on one spot) and the smallest xy distance between two selected
cameras. Calibration of the first snapshot of each scene.

Usage:
    uv run python scripts/diagnosis/camera_selection_spread.py [--k 5]
"""

import argparse
import json
import os

import cv2
import numpy as np
from sklearn.cluster import KMeans

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(src_dir, "datasets", "SynMVCrowd", "data")


def scene_dir(scene):
    for split in ("train", "validate", "test"):
        path = os.path.join(DATA, split, scene)
        if os.path.isdir(path):
            return path
    return None


def camera_centres(path):
    """cam_id -> camera centre C = -R^T tvec, from the first snapshot's calibration."""
    first = sorted((d for d in os.listdir(path) if d.isdigit()), key=int)[0]
    json_dir = os.path.join(path, first, "jsons")
    centres = {}
    for cam_id in range(51):
        f = os.path.join(json_dir, f"{cam_id}.json")
        if os.path.exists(f):
            with open(f) as fh:
                d = json.load(fh)
            tvec = np.array(d["tvec"], float).flatten()
            R, _ = cv2.Rodrigues(np.array(d["rvec"], float).flatten())
            centres[cam_id] = -R.T @ tvec
    return centres


def kmeans_select(points, ids, k):
    """Same rule as utils/optimize_cameras.py: the camera closest to each cluster centre, no duplicates."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(points)
    chosen = []
    for centre in km.cluster_centers_:
        for j in np.argsort(np.linalg.norm(points - centre, axis=1)):
            if ids[j] not in chosen:
                chosen.append(ids[j])
                break
    return chosen


def max_gap_deg(C, ids, centre):
    a = np.sort([np.degrees(np.arctan2(C[i][1] - centre[1], C[i][0] - centre[0])) % 360 for i in ids])
    return float(np.diff(np.r_[a, a[0] + 360]).max())


def min_dist(C, ids):
    P = np.array([C[i][:2] for i in ids])
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    D[D == 0] = np.inf
    return float(D.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        selection = json.load(f)
    with open(os.path.join(src_dir, "utils", "offaxis_combined_allcams.json")) as f:
        offaxis = json.load(f)

    rows = []
    for scene, sets in selection.items():
        path = scene_dir(scene)
        if path is None or str(args.k) not in sets:
            continue
        C = camera_centres(path)
        bad = {int(c) for c, v in offaxis.get(scene, {}).items() if v["status"] != "clean"}
        pool = [c for c in C if c not in bad]
        centre = np.mean([C[c][:2] for c in pool], axis=0)
        stored = [int(c) for c in sets[str(args.k)]]
        alt = kmeans_select(np.array([C[c] for c in pool]), pool, args.k)
        rows.append((max_gap_deg(C, stored, centre), max_gap_deg(C, alt, centre), min_dist(C, stored), min_dist(C, alt)))

    r = np.array(rows)
    print(f"scenes: {len(rows)}, k = {args.k}")
    print(f"largest angular gap [deg], median: tvec selection {np.median(r[:, 0]):.1f}, camera centres {np.median(r[:, 1]):.1f}")
    print(f"smallest distance between two selected cameras [m], median: tvec {np.median(r[:, 2]):.1f}, centres {np.median(r[:, 3]):.1f}")
    print(f"scenes with a gap > 180 deg: tvec {int((r[:, 0] > 180).sum())}, centres {int((r[:, 1] > 180).sum())}")


if __name__ == "__main__":
    main()
