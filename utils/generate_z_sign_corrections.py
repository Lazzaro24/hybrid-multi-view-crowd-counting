"""
generate_z_sign_corrections.py

Writes the per-camera z_sign corrections (scene_name -> {camera_id: z_sign}; --output, default z_sign_corrections.json;
the file in use is utils/z_sign_corrections_full.json) for EVERY camera found on disk (every
*.json in a scene's first snapshot's jsons/ folder), not only the K-means subsets: the reference-point cache
(cache_reference.py) covers all 50 cameras, and an uncalibrated camera would silently get z_sign = +1.0 even if its
extrinsics are flipped (doc/camera_bug.md).

Usage:
    python generate_z_sign_corrections.py
    python generate_z_sign_corrections.py --dry_run
"""

import os
import json
import argparse

import numpy as np
import cv2

src_dir = os.path.dirname(os.path.abspath(__file__))


def calibrate_z_sign_for_camera(scene_folder, cam_id, snapshot_ids, min_visible_people=5, max_snapshots_to_try=3):
    all_world_points = []
    rvec, tvec = None, None

    for snapshot_id in snapshot_ids[:max_snapshots_to_try]:
        json_path = os.path.join(scene_folder, str(snapshot_id), "jsons", f"{cam_id}.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r") as f:
            cam_data = json.load(f)

        if rvec is None:
            rvec = np.array(cam_data["rvec"], dtype=np.float32)
            tvec = np.array(cam_data["tvec"], dtype=np.float32)

        world_points = [
            p["world"] for p in cam_data["image_info"] if p.get("pixel") is not None and p.get("world") is not None
        ]
        all_world_points.extend(world_points)

        if len(all_world_points) >= min_visible_people:
            break

    if rvec is None or len(all_world_points) == 0:
        return None, None, 0, None, None

    world_points_np = np.array(all_world_points, dtype=np.float32)
    R, _ = cv2.Rodrigues(rvec)

    z_cam = ((R @ world_points_np.T).T + tvec)[:, 2]
    n_points = len(z_cam)
    frac_negative = float((z_cam < 0).mean())
    z_sign = -1.0 if frac_negative > 0.5 else 1.0

    camera_position = -R.T @ tvec
    look_direction = R[2, :]
    look_direction = look_direction / (np.linalg.norm(look_direction) + 1e-12)
    downward_ness = float(-look_direction[2])

    return z_sign, frac_negative, n_points, float(camera_position[2]), downward_ness


def discover_all_cameras(scene_folder, snapshot_id):
    """Every camera that actually has a json file for this snapshot --
    i.e. the full set camera_ids=None would load."""
    jsons_dir = os.path.join(scene_folder, str(snapshot_id), "jsons")
    if not os.path.isdir(jsons_dir):
        return []
    cam_ids = []
    for fname in os.listdir(jsons_dir):
        if fname.endswith(".json"):
            try:
                cam_ids.append(int(fname[:-5]))
            except ValueError:
                continue
    return sorted(cam_ids)


def find_scene_folders(dataset_root):
    data_dir = os.path.join(dataset_root, "data")
    result = []
    for split_name in sorted(os.listdir(data_dir)):
        split_dir = os.path.join(data_dir, split_name)
        if not os.path.isdir(split_dir):
            continue
        for scene_name in sorted(os.listdir(split_dir)):
            scene_folder = os.path.join(split_dir, scene_name)
            if os.path.isdir(scene_folder):
                result.append((split_name, scene_name, scene_folder))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_visible_people", type=int, default=5)
    parser.add_argument("--max_snapshots_to_try", type=int, default=3)
    parser.add_argument("--ambiguous_low", type=float, default=0.1)
    parser.add_argument("--ambiguous_high", type=float, default=0.9)
    parser.add_argument("--output", type=str, default="z_sign_corrections.json")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    dataset_root = os.path.join(src_dir, "datasets", "SynMVCrowd")
    scenes = find_scene_folders(dataset_root)
    print(f"[info] {len(scenes)} scenes found across all splits\n")

    corrections = {}
    n_flipped_total = 0
    n_ambiguous_total = 0

    for split_name, scene_name, scene_folder in scenes:
        snapshot_ids = sorted(
            [d for d in os.listdir(scene_folder) if os.path.isdir(os.path.join(scene_folder, d))],
            key=lambda x: int(x) if x.isdigit() else x,
        )
        if not snapshot_ids:
            print(f"[warning] no snapshots for {split_name}/{scene_name}, skipping")
            continue

        all_cam_ids = discover_all_cameras(scene_folder, snapshot_ids[0])
        if not all_cam_ids:
            print(f"[warning] no camera json files found for {split_name}/{scene_name}, skipping")
            continue

        print(f"[info] {split_name}/{scene_name}: {len(all_cam_ids)} cameras discovered on disk")

        scene_corrections = {}
        for cam_id in all_cam_ids:
            z_sign, frac_neg, n_points, height, downward_ness = calibrate_z_sign_for_camera(
                scene_folder,
                cam_id,
                snapshot_ids,
                min_visible_people=args.min_visible_people,
                max_snapshots_to_try=args.max_snapshots_to_try,
            )

            if z_sign is None:
                scene_corrections[str(cam_id)] = 1.0
                continue

            is_ambiguous = args.ambiguous_low < frac_neg < args.ambiguous_high
            scene_corrections[str(cam_id)] = z_sign if not is_ambiguous else 1.0

            if is_ambiguous:
                n_ambiguous_total += 1
                print(f"  [AMBIGUOUS] camera {cam_id}: {frac_neg * 100:.1f}% negative over {n_points} points")
            elif z_sign < 0:
                n_flipped_total += 1
                print(
                    f"  [FLIPPED]   camera {cam_id}: {frac_neg * 100:.1f}% negative over {n_points} points "
                    f"| height={height:.1f} downward={downward_ness:.2f}"
                )

        corrections[scene_name] = scene_corrections

    print("\n" + "=" * 70)
    print(f"Total flipped cameras (full universe): {n_flipped_total}")
    print(f"Total ambiguous cameras (full universe): {n_ambiguous_total}")

    if args.dry_run:
        print(f"[dry run] would write {len(corrections)} scenes' corrections to {args.output} -- skipped.")
    else:
        with open(args.output, "w") as f:
            json.dump(corrections, f, indent=2)
        print(f"[info] wrote corrections for {len(corrections)} scenes to {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
