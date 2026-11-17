"""
Camera viewing geometry of SynMVCrowd, WILDTRACK and MultiviewX, read from the calibration only (no model, no images).
Feeds the camera-geometry paragraph of Thesis Ch. 3 (sec:datasets) and the flat-plane error Delta_Z = (Z - d) cot(theta)
of Ch. 2.

SynMVCrowd (first snapshot of every scene; all cameras and the five selected views of optimal_cameras_filtered.json):
  pitch         depression of the optical axis below the horizon, from the json's Unreal camera rotation
  ray_elev      elevation of the ray from the camera to every visible annotated head (pixel is not None) = theta
  cam_height    per camera, median height of the camera above its visible heads (m)
WILDTRACK / MultiviewX (calibrations/extrinsic/*.xml): camera height and optical-axis depression from rvec / tvec.

Finding (2026-11-03): SynMVCrowd views are oblique, not near-overhead. Selected views: pitch median 30.9 deg
(p10-p90 24.8-39.0); ray_elev median 23.0 deg (p10-p90 12.1-41.3); cameras ~17.6 m above the heads. WILDTRACK:
1.7-3.4 m high, depression 8.7-20.1 deg. At theta = 23 deg a head 0.2 m off the head plane moves 0.47 m.

Usage:
    uv run scripts/diagnosis/camera_elevation.py
"""

import ast
import glob
import json
import math
import os
import re

import cv2
import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYN_ROOT = os.path.join(SRC_DIR, "datasets", "SynMVCrowd", "data")
SYN_SELECTION = os.path.join(SRC_DIR, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")
XML_DIRS = {
    "WILDTRACK": os.path.join(SRC_DIR, "datasets", "Wildtrack", "data", "calibrations", "extrinsic"),
    "MultiviewX": os.path.join(SRC_DIR, "datasets", "MultiviewX", "data", "calibrations", "extrinsic"),
}


def summarize(name, values):
    a = np.asarray(values)
    p10, p50, p90 = np.percentile(a, [10, 50, 90])
    print(f"{name:<34} n={len(a):>7}  median={p50:6.1f}  p10={p10:6.1f}  p90={p90:6.1f}  min={a.min():6.1f}  max={a.max():6.1f}")


def synmvcrowd():
    selection = json.load(open(SYN_SELECTION))
    pitch_all, pitch_sel, elev_all, elev_sel, cam_height = [], [], [], [], []
    for split in ("train", "validate", "test"):
        for scene in sorted(os.listdir(os.path.join(SYN_ROOT, split))):
            scene_dir = os.path.join(SYN_ROOT, split, scene)
            snapshot = min((d for d in os.listdir(scene_dir) if d.isdigit()), key=int)
            json_dir = os.path.join(scene_dir, snapshot, "jsons")
            chosen = set(selection.get(scene, {}).get("5", []))
            for f in os.listdir(json_dir):
                cam = json.load(open(os.path.join(json_dir, f)))
                selected = int(f.split(".")[0]) in chosen
                loc = np.asarray(cam["camera"]["location"])
                pitch = -cam["camera"]["rotation"][0]
                pitch_all.append(pitch)
                if selected:
                    pitch_sel.append(pitch)
                info = cam["image_info"]
                info = ast.literal_eval(info) if isinstance(info, str) else info
                heads = np.asarray([p["world"] for p in info if p["pixel"] is not None])
                if not len(heads):
                    continue
                dz = loc[2] - heads[:, 2]
                elev = np.degrees(np.arctan2(dz, np.hypot(*(heads[:, :2] - loc[:2]).T)))
                elev_all += elev.tolist()
                cam_height.append(np.median(dz))
                if selected:
                    elev_sel += elev.tolist()
    print("SynMVCrowd")
    summarize("pitch, all cameras (deg)", pitch_all)
    summarize("pitch, selected 5 (deg)", pitch_sel)
    summarize("ray elevation, all cameras (deg)", elev_all)
    summarize("ray elevation, selected 5 (deg)", elev_sel)
    summarize("camera height above heads (m)", cam_height)
    for theta in np.percentile(elev_sel, [10, 50, 90]):
        cot = 1 / math.tan(math.radians(theta))
        print(f"  theta={theta:5.1f} deg: 0.2 m off the plane -> {0.2 * cot:.2f} m, feet (1.75 m) -> {1.75 * cot:.2f} m")


def read_xml_vector(text, tag):
    body = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.S).group(1)
    data = re.search(r"<data>(.*?)</data>", body, re.S)
    return np.asarray([float(x) for x in (data.group(1) if data else body).split()])


def xml_calibrations():
    # Heights are in the dataset's world unit (WILDTRACK: cm). The MultiviewX world z axis points the other way, so
    # only the depression magnitude is comparable there.
    for name, xml_dir in XML_DIRS.items():
        print(name)
        for f in sorted(glob.glob(os.path.join(xml_dir, "*.xml"))):
            text = open(f).read()
            R, _ = cv2.Rodrigues(read_xml_vector(text, "rvec"))
            center = -R.T @ read_xml_vector(text, "tvec")
            axis = R.T @ np.array([0.0, 0.0, 1.0])
            depression = np.degrees(np.arcsin(abs(axis[2])))
            print(f"  {os.path.basename(f):<20} height={center[2]:8.2f}  depression={depression:5.1f} deg")


if __name__ == "__main__":
    synmvcrowd()
    xml_calibrations()
