"""
check_calibration.py

Calibration check, step 1 of doc/calibration_check_plan.md: how far does the projection of every annotated 3D head
(`image_info[*].world`) with the json camera (cameraMatrix scaled to 1920x1080, distCoeffs, rvec, tvec; the same math
as the homography projector and the reference points) land from the dataset's own head pixel (`pixel`, [row, col])?

Per scene and camera (the K-means 5-camera set the models use) over --snapshots evenly spaced snapshots:
  - median / p90 miss [px at 1920x1080] over all people with a `pixel`,
  - the same split by radial position (centre / middle / border of the image) to separate distortion from pose,
  - the per-snapshot median miss (min / median / max over snapshots): the cameras are static, but the json gives
    different intrinsics, distortion and pose per snapshot, so the miss can change from frame to frame,
  - the spread of the json parameters over snapshots (fx, k1, camera centre).

Output: diagnostics/calibration/calibration.json (per scene, per camera) and a per-scene table on stdout.

Usage (from src):
    uv run python utils/check_calibration.py                    # every split, 25 snapshots per scene
    uv run python utils/check_calibration.py --scenes validate/scene_49 --snapshots 200
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from utils.occluder_masks import project, IMG_W, IMG_H

DATA = os.path.join(src_dir, "datasets", "SynMVCrowd", "data")
RADIAL_BINS = (0.0, 0.4, 0.7, 10.0)  # normalised distance from the image centre (1 = half the diagonal)
RADIAL_NAMES = ("centre", "middle", "border")


def miss_one(meta):
    """Per person with a `pixel`: miss [px] and normalised radial position of the dataset pixel."""
    people = [p for p in meta["image_info"] if p.get("pixel") is not None]
    if not people:
        return np.zeros(0), np.zeros(0)
    world = np.array([p["world"] for p in people], np.float64)
    pix = np.array([[p["pixel"][1], p["pixel"][0]] for p in people], np.float64)  # -> [x, y]
    uv, depth, _ = project(meta, world)
    miss = np.linalg.norm(uv - pix, axis=1)
    miss[depth <= 0] = np.inf  # behind the camera: a total miss
    r = np.linalg.norm((pix - [IMG_W / 2, IMG_H / 2]) / (np.hypot(IMG_W, IMG_H) / 2), axis=1)
    return miss, r


def stats(x):
    x = x[np.isfinite(x)] if len(x) else x
    if not len(x):
        return None
    return {
        "n": int(len(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "mean": float(x.mean()),
    }


def check_scene(split, scene, cams, n_snap):
    scene_dir = os.path.join(DATA, split, scene)
    snaps = sorted([s for s in os.listdir(scene_dir) if s.isdigit()], key=int)
    snaps = [snaps[i] for i in np.unique(np.linspace(0, len(snaps) - 1, min(n_snap, len(snaps))).round().astype(int))]
    out = {}
    for cam in cams:
        misses, rads, per_snap, params, behind = [], [], [], [], 0
        for s in snaps:
            jp = os.path.join(scene_dir, s, "jsons", f"{cam}.json")
            if not os.path.exists(jp):
                continue
            with open(jp) as f:
                meta = json.load(f)
            m, r = miss_one(meta)
            behind += int(np.isinf(m).sum())
            misses.append(m)
            rads.append(r)
            if np.isfinite(m).any():
                per_snap.append(float(np.median(m[np.isfinite(m)])))
            R, _ = cv2.Rodrigues(np.array(meta["rvec"], np.float64))
            centre = -R.T @ np.array(meta["tvec"], np.float64)
            params.append([meta["cameraMatrix"][0][0], meta["distCoeffs"][0], *centre])
        if not misses:
            continue
        m, r = np.concatenate(misses), np.concatenate(rads)
        params = np.array(params)
        out[str(cam)] = {
            "all": stats(m),
            "radial": {
                name: stats(m[(r >= lo) & (r < hi)])
                for name, lo, hi in zip(RADIAL_NAMES, RADIAL_BINS[:-1], RADIAL_BINS[1:])
            },
            "per_snapshot_median": {
                "min": float(np.min(per_snap)),
                "median": float(np.median(per_snap)),
                "max": float(np.max(per_snap)),
            }
            if per_snap
            else None,
            "behind_camera": behind,
            "param_spread": {
                "fx_rel_std": float(params[:, 0].std() / abs(params[:, 0].mean())),
                "k1_min": float(params[:, 1].min()),
                "k1_max": float(params[:, 1].max()),
                "centre_std": float(np.linalg.norm(params[:, 2:].std(0))),
            },
            "snapshots": len(misses),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Reprojection miss of the json calibration against the dataset pixels.")
    ap.add_argument("--scenes", default="all", help="comma-separated split/scene, or 'all'")
    ap.add_argument("--snapshots", type=int, default=25, help="evenly spaced snapshots per scene")
    ap.add_argument("--out", default=os.path.join(src_dir, "diagnostics", "calibration", "calibration.json"))
    args = ap.parse_args()

    with open(os.path.join(src_dir, "utils", "excluded_scenes.json")) as f:
        excluded = json.load(f)
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cam_sets = json.load(f)
    if args.scenes == "all":
        specs = [
            f"{split}/{s}"
            for split in ("train", "validate", "test")
            for s in sorted(os.listdir(os.path.join(DATA, split)), key=lambda x: int(x.split("_")[1]))
        ]
    else:
        specs = args.scenes.split(",")

    result = {"args": vars(args), "scenes": {}}
    print(f"{'scene':<22} {'median':>7} {'p90':>7} {'worst cam p90':>14} {'border/centre':>14} {'snap med range':>16}")
    for spec in specs:
        split, scene = spec.split("/")
        if scene in excluded or scene not in cam_sets:
            continue
        cams = cam_sets[scene]["5"]
        per_cam = check_scene(split, scene, cams, args.snapshots)
        meds = [c["all"]["median"] for c in per_cam.values() if c["all"]]
        p90s = [c["all"]["p90"] for c in per_cam.values() if c["all"]]
        worst = max(per_cam, key=lambda c: per_cam[c]["all"]["p90"] if per_cam[c]["all"] else -1)
        ratios = [
            c["radial"]["border"]["median"] / c["radial"]["centre"]["median"]
            for c in per_cam.values()
            if c["radial"]["border"] and c["radial"]["centre"] and c["radial"]["centre"]["median"] > 0
        ]
        snap_rng = [
            (c["per_snapshot_median"]["min"], c["per_snapshot_median"]["max"])
            for c in per_cam.values()
            if c["per_snapshot_median"]
        ]
        result["scenes"][scene] = {
            "split": split,
            "cams": cams,
            "median": float(np.mean(meds)),
            "p90": float(np.mean(p90s)),
            "worst_cam": int(worst),
            "worst_cam_p90": per_cam[worst]["all"]["p90"],
            "border_centre_ratio": float(np.median(ratios)) if ratios else None,
            "per_cam": per_cam,
        }
        lo, hi = min(r[0] for r in snap_rng), max(r[1] for r in snap_rng)
        print(
            f"{spec:<22} {np.mean(meds):7.1f} {np.mean(p90s):7.1f} {per_cam[worst]['all']['p90']:9.1f} (c{worst:<2})"
            f" {np.median(ratios) if ratios else float('nan'):14.2f} {lo:7.1f}-{hi:<8.1f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
