"""
check_offaxis_scene.py

Per-camera off-axis fold-back check, ratio-based: a camera is flagged when a disconnected secondary coverage component
reaches a median off-axis distance r of at least --flag_median_ratio times the main component's. All secondary
components are scanned; 'foldback' wins over 'uncertain'. --closing_iterations (default 1, 3x3 structure) merges thin
fragments before labelling; the size floor and the ratio are still measured on the unclosed mask.

The geometry helpers here (build_grid_points, compute_offaxis_and_frame, get_labeled_components, ...) are shared with
check_offaxis_combined.py, the classifier the camera selection uses, and with visualize_camera_fixes.py.

Usage:
    python check_offaxis_scene.py --only_scene scene_14
    python check_offaxis_scene.py --only_scene scene_14 --skip_calibration
    python check_offaxis_scene.py --only_scene scene_14 --flag_median_ratio 1.6 --closing_iterations 1
    python check_offaxis_scene.py   # full corpus in one process -- NOT recommended,
                                     # use run_offaxis_sweep.py instead
"""

import os
import sys
import json
import gc
import argparse

import numpy as np
import torch
import cv2
from scipy import ndimage

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset

NUM_Z_ANCHORS = 4
IMG_SIZE = (720, 1280)
MIN_COMPONENT_SIZE_DEFAULT = 4
FLAG_MEDIAN_RATIO_DEFAULT = 1.6
CLOSING_ITERATIONS_DEFAULT = 1


def build_grid_points(wld_map_paras, num_z_anchors=NUM_Z_ANCHORS):
    r, a, b, h_actual, w_actual, d_mean, w_min, h_min, w_max, h_max, d_delta = wld_map_paras
    xs = torch.arange(w_actual) / r + w_min - a
    ys = torch.arange(h_actual) / r + h_min - b
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    z_heights = torch.linspace(-d_delta, 1.8 + d_delta, steps=num_z_anchors) + d_mean
    points_3d = []
    for z in z_heights:
        Z_plane = torch.full_like(grid_x, z.item())
        plane_3d = torch.stack([grid_x, grid_y, Z_plane], dim=-1)
        points_3d.append(plane_3d)

    points_3d = torch.stack(points_3d, dim=-2).reshape(-1, 3)
    return points_3d.numpy().astype(np.float32), int(h_actual), int(w_actual)


def compute_offaxis_and_frame(cam_row, points_3d_np, h_actual, w_actual, num_z_anchors, img_size=IMG_SIZE):
    H, W = img_size
    cam = cam_row.cpu().numpy() if torch.is_tensor(cam_row) else cam_row
    fx, fy = cam[0] * W, cam[1] * H
    cx, cy = cam[2] * W, cam[3] * H
    dist_coeffs = cam[4:9]
    rvec = cam[9:12]
    tvec = cam[12:15]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    R_matrix, _ = cv2.Rodrigues(rvec)
    points_cam = (R_matrix @ points_3d_np.T).T + tvec
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    eps = 1e-6
    safe_Z = np.where(np.abs(Z) < eps, eps, Z)
    r = np.sqrt((X / safe_Z) ** 2 + (Y / safe_Z) ** 2)

    image_points, _ = cv2.projectPoints(points_3d_np, rvec, tvec, K, dist_coeffs)
    image_points = image_points.reshape(-1, 2)
    x_norm = (image_points[:, 0] / W) * 2.0 - 1.0
    y_norm = (image_points[:, 1] / H) * 2.0 - 1.0

    def reshape(arr):
        return arr.reshape(h_actual, w_actual, num_z_anchors)

    return reshape(r), reshape(Z), reshape(x_norm), reshape(y_norm)


def get_labeled_components(any_coverage, closing_iterations):
    """Coverage mask (optionally closed) and its connected components; shared by both classifiers."""
    coverage_for_labeling = any_coverage
    if closing_iterations > 0:
        coverage_for_labeling = ndimage.binary_closing(
            any_coverage, structure=np.ones((3, 3)), iterations=closing_iterations
        )
    labeled, n_components = ndimage.label(coverage_for_labeling, structure=np.ones((3, 3)))
    component_sizes = (
        ndimage.sum(coverage_for_labeling, labeled, index=range(1, n_components + 1)) if n_components else np.array([])
    )
    return coverage_for_labeling, labeled, n_components, component_sizes


def classify_camera(
    z_sign_i, cam_row, points_3d_np, h_actual, w_actual, min_component_size, flag_median_ratio, closing_iterations
):
    """
    Returns (status, note) where status is one of:
      'clean'      - single component, or only noise-sized secondaries
      'foldback'   - at least one real-sized secondary clears the ratio test
      'uncertain'  - no coverage at all (likely still-broken z_sign), or at
                     least one real-sized secondary whose ratio doesn't
                     clear the bar and none of the others do either
    """
    r_map, z_map, x_norm_map, y_norm_map = compute_offaxis_and_frame(
        cam_row, points_3d_np, h_actual, w_actual, NUM_Z_ANCHORS
    )
    z_corrected = z_map * z_sign_i
    in_frame = (x_norm_map >= -1.0) & (x_norm_map <= 1.0) & (y_norm_map >= -1.0) & (y_norm_map <= 1.0)
    current_valid = in_frame & (z_corrected >= 1e-2)

    any_coverage = current_valid.any(axis=-1)
    if not any_coverage.any():
        return "uncertain", "no coverage at all (likely still-broken z_sign)"

    _, labeled, n_components, component_sizes = get_labeled_components(any_coverage, closing_iterations)
    if n_components <= 1:
        return "clean", None

    main_id = int(np.argmax(component_sizes)) + 1
    main_mask_3d = np.broadcast_to((labeled == main_id)[..., None], r_map.shape)
    r_main = r_map[main_mask_3d & current_valid]
    if r_main.size == 0:
        return "uncertain", "main component has no valid anchors (edge case)"
    main_max_r = float(r_main.max())

    noise_filtered = 0
    best_status = "clean"
    best_note = None
    for comp_id in range(1, n_components + 1):
        if comp_id == main_id:
            continue
        comp_size = component_sizes[comp_id - 1]
        if comp_size < min_component_size:
            noise_filtered += 1
            continue

        other_mask_3d = np.broadcast_to((labeled == comp_id)[..., None], r_map.shape)
        r_other = r_map[other_mask_3d & current_valid]
        if r_other.size == 0:
            continue
        median_r_other = float(np.median(r_other))
        ratio = median_r_other / main_max_r if main_max_r > 0 else float("inf")

        if ratio > flag_median_ratio:
            return "foldback", (
                f"secondary component (size={int(comp_size)}) median_r={median_r_other:.2f} "
                f"vs main max_r={main_max_r:.2f} (ratio={ratio:.2f})"
            )
        else:
            best_status = "uncertain"
            best_note = (
                f"secondary component (size={int(comp_size)}) median_r={median_r_other:.2f} "
                f"vs main max_r={main_max_r:.2f} (ratio={ratio:.2f}) -- real-sized but ratio inconclusive"
            )
            # Don't return yet -- a later component might still clear the bar.

    if best_status == "uncertain":
        return "uncertain", best_note
    if noise_filtered:
        return "clean", f"{noise_filtered} noise-sized secondary component(s) ignored"
    return "clean", None


def load_z_sign_corrections(src_dir):
    """The full corrections file (every camera on disk, utils/z_sign_corrections_full.json) if present, else the
    narrow ablation-subset file (utils/, or src/ where it used to live)."""
    candidates = [
        os.path.join(src_dir, "utils", "z_sign_corrections_full.json"),
        os.path.join(src_dir, "utils", "z_sign_corrections.json"),
        os.path.join(src_dir, "z_sign_corrections.json"),
    ]
    for z_sign_path in candidates:
        if os.path.exists(z_sign_path):
            print(f"[z_sign] using {z_sign_path}")
            with open(z_sign_path, "r") as f:
                return json.load(f)
    print(f"[warning] none of {candidates} found -- proceeding with NO z-sign corrections")
    return {}


def load_scene_locations(dataset_root):
    data_dir = os.path.join(dataset_root, "data")
    scene_locations = {}
    for split_name in sorted(os.listdir(data_dir)):
        split_dir = os.path.join(data_dir, split_name)
        if not os.path.isdir(split_dir):
            continue
        for scene_name in os.listdir(split_dir):
            if os.path.isdir(os.path.join(split_dir, scene_name)):
                scene_locations[scene_name] = (split_name, os.path.join(split_dir, scene_name))
    return scene_locations


def run_calibration_check(
    scene_locations,
    z_sign_corrections,
    optimal_cameras_config,
    calibrate_scene,
    calibrate_cam,
    num_cams,
    min_component_size,
    flag_median_ratio,
    closing_iterations,
):
    if calibrate_scene not in scene_locations:
        print(f"[calibration] scene {calibrate_scene} not found on disk -- skipping")
        return

    split_name, scene_folder = scene_locations[calibrate_scene]
    cam_ids = optimal_cameras_config.get(calibrate_scene, {}).get(str(num_cams))
    if cam_ids is None or calibrate_cam not in cam_ids:
        cam_ids = [calibrate_cam]

    dataset = SynMVCrowdDataset(scene_dir=scene_folder, camera_ids=cam_ids, is_caching=True)
    sample = dataset[0]
    points_3d_np, h_actual, w_actual = build_grid_points(sample["wld_map_paras"])
    scene_corrections = z_sign_corrections.get(calibrate_scene, {})

    idx = cam_ids.index(calibrate_cam)
    z_sign_i = scene_corrections.get(str(calibrate_cam), 1.0)
    status, note = classify_camera(
        z_sign_i,
        sample["camera_paras"][idx],
        points_3d_np,
        h_actual,
        w_actual,
        min_component_size,
        flag_median_ratio,
        closing_iterations,
    )

    del dataset, sample
    gc.collect()

    print("\n" + "=" * 70)
    print(f"[CALIBRATION CHECK] {calibrate_scene}/camera {calibrate_cam} (known real fold-back case)")
    print(f"  -> classified as: {status}  ({note})")
    if status != "foldback":
        note_text = note or ""
        if "size=" not in note_text:
            print(
                "  Likely cause: min_component_size may be filtering the real "
                "component as noise. Try lowering --min_component_size."
            )
        elif "ratio inconclusive" in note_text:
            print(
                "  Likely cause: FLAG_MEDIAN_RATIO is still too strict for this case. "
                "Try lowering --flag_median_ratio further."
            )
        else:
            print("  Cause unclear from this single case -- inspect visually.")
    print("=" * 70 + "\n")


def process_one_scene(
    scene_folder, cam_ids, scene_corrections, min_component_size, flag_median_ratio, closing_iterations
):
    dataset = SynMVCrowdDataset(scene_dir=scene_folder, camera_ids=cam_ids, is_caching=True)
    sample = dataset[0]
    points_3d_np, h_actual, w_actual = build_grid_points(sample["wld_map_paras"])

    cam_results = {}
    for i, cam_id in enumerate(cam_ids):
        z_sign_i = scene_corrections.get(str(cam_id), 1.0)
        status, note = classify_camera(
            z_sign_i,
            sample["camera_paras"][i],
            points_3d_np,
            h_actual,
            w_actual,
            min_component_size,
            flag_median_ratio,
            closing_iterations,
        )
        cam_results[cam_id] = (status, note)

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
    parser.add_argument("--flag_median_ratio", type=float, default=FLAG_MEDIAN_RATIO_DEFAULT)
    parser.add_argument(
        "--closing_iterations",
        type=int,
        default=CLOSING_ITERATIONS_DEFAULT,
        help="Morphological closing iterations before labeling, to merge "
        "fragmented thin artifacts. 0 disables closing entirely.",
    )
    parser.add_argument("--calibrate_scene", type=str, default="scene_2")
    parser.add_argument("--calibrate_cam", type=int, default=26)
    parser.add_argument("--skip_calibration", action="store_true")
    parser.add_argument(
        "--only_scene",
        type=str,
        default=None,
        help="Process just this one scene and merge into the existing output file. "
        "Intended to be driven by run_offaxis_sweep.py, one subprocess per scene.",
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
            args.flag_median_ratio,
            args.closing_iterations,
        )

    out_path = os.path.join(src_dir, "offaxis_classification.json")
    existing_output = load_existing_output(out_path)

    scenes_to_process = [args.only_scene] if args.only_scene else sorted(optimal_cameras_config.keys())

    for scene_name in scenes_to_process:
        if scene_name not in scene_locations or scene_name not in optimal_cameras_config:
            print(f"[skip] {scene_name}: not found")
            continue
        split_name, scene_folder = scene_locations[scene_name]
        cam_ids = optimal_cameras_config[scene_name].get(str(args.num_cams))
        if not cam_ids:
            continue

        scene_corrections = z_sign_corrections.get(scene_name, {})
        cam_results = process_one_scene(
            scene_folder,
            cam_ids,
            scene_corrections,
            args.min_component_size,
            args.flag_median_ratio,
            args.closing_iterations,
        )

        existing_output[scene_name] = {str(cid): status for cid, (status, _) in cam_results.items()}
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
    n_clean = sum(1 for r in existing_output.values() if all(s == "clean" for s in r.values()))
    n_foldback = sum(1 for r in existing_output.values() if any(s == "foldback" for s in r.values()))
    n_uncertain = n_total - n_clean - n_foldback
    print("\n" + "=" * 70)
    print(f"Scenes recorded so far: {n_total}")
    print(f"  fully clean: {n_clean}   have a foldback camera: {n_foldback}   uncertain only: {n_uncertain}")
    print(f"Per-camera classification file: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
