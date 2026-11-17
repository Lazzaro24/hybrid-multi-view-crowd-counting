"""
check_offaxis_combined.py

Per-camera off-axis fold-back classifier used to build optimal_cameras_filtered.json (via run_offaxis_sweep.py and
regenerate_optimal_cameras_filtered.py). A large off-axis distance r alone is not enough: scene_2/camera 33 has a
single coverage component with max r = 5.11, a wide but legitimate field of view.

The criterion is the intersection of two signals:
  1. STRUCTURAL: is there a disconnected secondary component at all
     (after closing, to consolidate fragmented slivers)?
  2. MAGNITUDE: does that secondary component's OWN r (median, not ratio-
     to-main -- avoids the dilution/occlusion-confound problem from the
     ratio-based script) exceed an absolute cutoff tied to the documented
     r_offaxis > 3.0 production threshold?

A camera is flagged 'foldback' only if BOTH hold. Single-component cameras
(like scene_2/cam33) are never flagged regardless of how high their r
reaches, because there's nothing disconnected to blame on model breakdown.

Residual risk, not yet resolved: if closing_iterations merges a genuinely
disconnected fold-back sliver INTO the main component (geometrically
adjacent with only a small gap), it becomes invisible to this check too,
since only non-main components are evaluated. Spot-check known cases with
--closing_iterations 0 vs 1 if this is a concern for a given scene.

--threshold here applies to the SECONDARY component's median r (not a
ratio) -- default 2.5, giving margin below camera 26's confirmed 4.98
while still comfortably above typical in-FOV values. Not yet validated
against a confirmed LEGITIMATE secondary-component case (e.g. real
occlusion-caused disconnection with moderate r) -- we don't have one yet.
Spot-check newly-flagged secondaries visually before trusting this fully.

Usage:
    python check_offaxis_combined.py --only_scene scene_2 --all_cameras  # for exclusion-set building
    python check_offaxis_combined.py --only_scene scene_2
    python check_offaxis_combined.py --only_scene scene_2 --threshold 2.5 --closing_iterations 1
"""

import os
import sys
import json
import gc
import argparse

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from check_offaxis_scene import (
    build_grid_points,
    compute_offaxis_and_frame,
    get_labeled_components,
    load_z_sign_corrections,
    load_scene_locations,
    NUM_Z_ANCHORS,
)
from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset

THRESHOLD_DEFAULT = 2.5
MIN_COMPONENT_SIZE_DEFAULT = 4
CLOSING_ITERATIONS_DEFAULT = 1


def classify_camera(
    z_sign_i, cam_row, points_3d_np, h_actual, w_actual, min_component_size, threshold, closing_iterations
):
    """
    Returns (status, detail) where status is one of:
      'clean'     - single component (however wide/oblique), or only
                    noise-sized / below-threshold secondaries
      'foldback'  - a real-sized secondary component's own median r
                    exceeds --threshold
      'uncertain' - no coverage at all (likely still-broken z_sign)
    detail is a dict with diagnostic numbers for logging.
    """
    r_map, z_map, x_norm_map, y_norm_map = compute_offaxis_and_frame(
        cam_row, points_3d_np, h_actual, w_actual, NUM_Z_ANCHORS
    )
    z_corrected = z_map * z_sign_i
    in_frame = (x_norm_map >= -1.0) & (x_norm_map <= 1.0) & (y_norm_map >= -1.0) & (y_norm_map <= 1.0)
    current_valid = in_frame & (z_corrected >= 1e-2)

    any_coverage = current_valid.any(axis=-1)
    if not any_coverage.any():
        return "uncertain", {"reason": "no coverage at all (likely still-broken z_sign)"}

    overall_max_r = float(r_map[current_valid].max())

    _, labeled, n_components, component_sizes = get_labeled_components(any_coverage, closing_iterations)
    if n_components <= 1:
        # Single continuous component -- however high its own r gets, there's
        # no disconnection to blame on model breakdown. See scene_2/cam33.
        return "clean", {"n_components": 1, "overall_max_r": overall_max_r}

    main_id = int(np.argmax(component_sizes)) + 1

    worst_secondary_median_r = None
    flagged_component = None
    for comp_id in range(1, n_components + 1):
        if comp_id == main_id:
            continue
        comp_size = component_sizes[comp_id - 1]
        if comp_size < min_component_size:
            continue
        comp_mask_3d = np.broadcast_to((labeled == comp_id)[..., None], r_map.shape)
        r_comp = r_map[comp_mask_3d & current_valid]
        if r_comp.size == 0:
            continue
        median_r = float(np.median(r_comp))
        if worst_secondary_median_r is None or median_r > worst_secondary_median_r:
            worst_secondary_median_r = median_r
            flagged_component = {
                "comp_id": comp_id,
                "size": int(comp_size),
                "r_range": [float(r_comp.min()), float(r_comp.max())],
                "median_r": median_r,
            }

    detail = {"n_components": n_components, "overall_max_r": overall_max_r, "worst_secondary": flagged_component}

    if worst_secondary_median_r is not None and worst_secondary_median_r > threshold:
        return "foldback", detail
    return "clean", detail


def run_calibration_check(
    scene_locations,
    z_sign_corrections,
    optimal_cameras_config,
    calibrate_scene,
    calibrate_cam,
    num_cams,
    min_component_size,
    threshold,
    closing_iterations,
):
    if calibrate_scene not in scene_locations:
        print(f"[calibration] scene {calibrate_scene} not found -- skipping")
        return
    split_name, scene_folder = scene_locations[calibrate_scene]
    cam_ids = optimal_cameras_config.get(calibrate_scene, {}).get(str(num_cams))
    if cam_ids is None or calibrate_cam not in cam_ids:
        cam_ids = [calibrate_cam]

    dataset = SynMVCrowdDataset(scene_dir=scene_folder, camera_ids=cam_ids, is_caching=True)
    sample = dataset[0]
    points_3d_np, h_actual, w_actual = build_grid_points(sample["wld_map_paras"])
    idx = cam_ids.index(calibrate_cam)
    z_sign_i = z_sign_corrections.get(calibrate_scene, {}).get(str(calibrate_cam), 1.0)
    status, detail = classify_camera(
        z_sign_i,
        sample["camera_paras"][idx],
        points_3d_np,
        h_actual,
        w_actual,
        min_component_size,
        threshold,
        closing_iterations,
    )

    del dataset, sample
    gc.collect()

    print("\n" + "=" * 70)
    print(f"[CALIBRATION CHECK] {calibrate_scene}/camera {calibrate_cam} (known real fold-back case)")
    print(f"  -> classified as: {status}  ({detail})")
    if status != "foldback":
        print(
            f"  !! WARNING: known-bad camera NOT flagged. Threshold={threshold} may be too high, "
            f"or closing_iterations={closing_iterations} merged the artifact into the main component."
        )
    print("=" * 70 + "\n")


def process_one_scene(scene_folder, cam_ids, scene_corrections, min_component_size, threshold, closing_iterations):
    dataset = SynMVCrowdDataset(scene_dir=scene_folder, camera_ids=cam_ids, is_caching=True)
    sample = dataset[0]
    points_3d_np, h_actual, w_actual = build_grid_points(sample["wld_map_paras"])

    cam_results = {}
    for i, cam_id in enumerate(cam_ids):
        z_sign_i = scene_corrections.get(str(cam_id), 1.0)
        status, detail = classify_camera(
            z_sign_i,
            sample["camera_paras"][i],
            points_3d_np,
            h_actual,
            w_actual,
            min_component_size,
            threshold,
            closing_iterations,
        )
        cam_results[cam_id] = (status, detail)

    del dataset, sample, points_3d_np
    gc.collect()
    return cam_results


def load_existing_output(out_path):
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            return json.load(f)
    return {}


def save_output(out_path, data):
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cams", type=int, default=5)
    parser.add_argument("--min_component_size", type=int, default=MIN_COMPONENT_SIZE_DEFAULT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD_DEFAULT,
        help="Absolute median-r cutoff for a SECONDARY component (not a ratio to main).",
    )
    parser.add_argument("--closing_iterations", type=int, default=CLOSING_ITERATIONS_DEFAULT)
    parser.add_argument("--calibrate_scene", type=str, default="scene_2")
    parser.add_argument("--calibrate_cam", type=int, default=26)
    parser.add_argument("--skip_calibration", action="store_true")
    parser.add_argument("--only_scene", type=str, default=None)
    parser.add_argument(
        "--all_cameras",
        action="store_true",
        help="Classify all 50 raw cameras per scene instead of just the "
        "ones already selected for --num_cams. Needed to build a full "
        "per-scene exclusion set BEFORE running K-means, rather than "
        "checking cameras after the fact.",
    )
    args = parser.parse_args()

    dataset_root = os.path.join(src_dir, "datasets", "SynMVCrowd")
    with open(os.path.join(dataset_root, "optimal_cameras.json"), "r") as f:
        optimal_cameras_config = json.load(f)

    z_sign_corrections = load_z_sign_corrections(src_dir)
    scene_locations = load_scene_locations(dataset_root)

    if not args.skip_calibration:
        run_calibration_check(
            scene_locations,
            z_sign_corrections,
            optimal_cameras_config,
            args.calibrate_scene,
            args.calibrate_cam,
            args.num_cams,
            args.min_component_size,
            args.threshold,
            args.closing_iterations,
        )

    out_path = os.path.join(src_dir, "offaxis_combined_allcams.json" if args.all_cameras else "offaxis_combined.json")
    existing_output = load_existing_output(out_path)

    scenes_to_process = [args.only_scene] if args.only_scene else sorted(optimal_cameras_config.keys())

    for scene_name in scenes_to_process:
        if scene_name not in scene_locations or scene_name not in optimal_cameras_config:
            print(f"[skip] {scene_name}: not found")
            continue
        split_name, scene_folder = scene_locations[scene_name]
        if args.all_cameras:
            cam_ids = list(range(51))  # cameras 0-50 inclusive, 51 total
        else:
            cam_ids = optimal_cameras_config[scene_name].get(str(args.num_cams))
            if not cam_ids:
                continue

        scene_corrections = z_sign_corrections.get(scene_name, {})
        cam_results = process_one_scene(
            scene_folder, cam_ids, scene_corrections, args.min_component_size, args.threshold, args.closing_iterations
        )

        existing_output[scene_name] = {str(cid): {"status": s, "detail": d} for cid, (s, d) in cam_results.items()}
        save_output(out_path, existing_output)

        statuses = [s for s, _ in cam_results.values()]
        tag = (
            "CLEAN"
            if all(s == "clean" for s in statuses)
            else "FOLDBACK"
            if any(s == "foldback" for s in statuses)
            else "UNCERTAIN"
        )
        print(f"[{tag}] {split_name}/{scene_name}: { {cid: s for cid, (s, _) in cam_results.items()} }")

    n_total = len(existing_output)
    n_clean = sum(1 for r in existing_output.values() if all(v["status"] == "clean" for v in r.values()))
    n_foldback = sum(1 for r in existing_output.values() if any(v["status"] == "foldback" for v in r.values()))
    print("\n" + "=" * 70)
    print(f"Scenes recorded so far: {n_total}  clean: {n_clean}  have foldback camera: {n_foldback}")
    print(f"Output: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
