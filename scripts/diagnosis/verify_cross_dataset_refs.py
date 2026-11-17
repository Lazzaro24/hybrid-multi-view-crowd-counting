"""
Checks the reference points of the Wildtrack / MultiviewX loaders (datasets/multiview_detection.py, ref_points=True)
before the proposed models use them (doc/cross_dataset.md, WP2). No model is loaded.

For every annotated person and every camera with a box for them, the map cell under the person is looked up in
'ref_points_full' and its feet anchor (z = d_mean - 1.75 m = 0 at --head_height 1.75) and head anchor (z = d_mean)
are converted to 1280 x 720 image pixels and compared with the box (scaled to 1280 x 720):
  feet     distance to the box's bottom centre, head  distance to its top centre;
  rel      the same divided by the box height (comparable between near and far people);
  invalid  share of these person cells marked invalid (-10): behind the camera (dataset Z_SIGN), or rejected by
           the off-axis threshold. Must be ~0 -- every one of them is visible.
The cell's coordinate is its corner (as in SynMVCrowd), up to 0.28 m from the person, so a few pixels of error
come from the grid alone.
Per camera over the whole map: share of cells whose feet anchor lies inside the image but is rejected by the
off-axis check (offaxis_threshold), which must not drop valid cells on wide-angle cameras.
With --images, one frame per camera is drawn: box bottom / top (green), feet / head anchors (red / blue).

Usage (from src/):
    uv run scripts/diagnosis/verify_cross_dataset_refs.py
    uv run scripts/diagnosis/verify_cross_dataset_refs.py --data_root ..\\..\\hybrid-multi-view-crowd-counting\\src\\datasets --images
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

from evaluate_cross_dataset import DATASETS
from utils.reference_points import compute_reference_points

W_IMG, H_IMG = 1280, 720


def to_pixels(uv):
    """grid_sample coordinates [..., 2] in [-1, 1] -> 1280 x 720 pixels."""
    return np.stack([(uv[..., 0] + 1) / 2 * W_IMG, (uv[..., 1] + 1) / 2 * H_IMG], axis=-1)


def check(name, args):
    cls, root = DATASETS[name]
    if args.data_root:
        root = os.path.join(args.data_root, os.path.relpath(root, os.path.join(src_dir, "datasets")))
    ds = cls(root, split=args.split, head_height=args.head_height, min_map_px=(200, 200), ref_points=True)
    refs = ds.ref_points.numpy()  # [C, H, W, 4, 2]
    paras = ds.wld_map_paras()
    r, a, b = (paras[i].item() for i in range(3))
    w_min, h_min = paras[6].item(), paras[7].item()
    sx, sy = W_IMG / ds.IMG_SIZE[1], H_IMG / ds.IMG_SIZE[0]
    frames = ds.frames[:: max(1, len(ds.frames) // args.frames)][: args.frames]
    print(
        f"\n=== {name} ({args.split}, {len(frames)} frames) | map {tuple(refs.shape[1:3])} | margins a={a} m, "
        f"b={b} m | eval_window {ds.eval_window().tolist()} | anchors z = "
        f"{[round(paras[5].item() + z, 2) for z in np.linspace(-paras[10].item(), 1.8 + paras[10].item(), 4)]} m"
    )

    per_cam = {
        c: {"feet": [], "head": [], "rel_feet": [], "rel_head": [], "invalid": 0, "n": 0} for c in range(ds.NUM_CAMS)
    }
    for f in frames:
        for x, y, boxes in ds.people[f]:
            row, col = int((y - h_min + b) * r), int((x - w_min + a) * r)
            for c, (x0, y0, x1, y1) in boxes.items():
                s = per_cam[c]
                s["n"] += 1
                if refs[c, row, col, 0, 0] == -10 or refs[c, row, col, 1, 0] == -10:
                    s["invalid"] += 1
                    continue
                feet, head = to_pixels(refs[c, row, col, :2])
                bottom = np.array([(x0 + x1) / 2 * sx, y1 * sy])
                top = np.array([(x0 + x1) / 2 * sx, y0 * sy])
                height = max((y1 - y0) * sy, 1.0)
                s["feet"].append(np.linalg.norm(feet - bottom))
                s["head"].append(np.linalg.norm(head - top))
                s["rel_feet"].append(s["feet"][-1] / height)
                s["rel_head"].append(s["head"][-1] / height)

    # Off-axis rejection: cells inside the image without the check but rejected with it (feet anchor)
    no_axis = compute_reference_points(
        paras,
        torch.stack(ds.camera_paras),
        img_size=(H_IMG, W_IMG),
        z_sign=[ds.Z_SIGN] * ds.NUM_CAMS,
        offaxis_threshold=1e9,
    ).numpy()[:, :, :, 0]
    inside = (np.abs(no_axis[..., 0]) <= 1) & (np.abs(no_axis[..., 1]) <= 1)
    rejected = inside & (refs[:, :, :, 0, 0] == -10)

    print(
        f"{'cam':>4} {'boxes':>6} {'invalid':>8} {'feet med':>9} {'p90':>6} {'rel':>6} {'head med':>9} {'p90':>6} "
        f"{'rel':>6} {'visible cells':>14} {'offaxis-rejected':>17}"
    )
    result = {"map": list(refs.shape[1:3]), "margins_m": [a, b], "eval_window": ds.eval_window().tolist(), "cams": {}}
    for c, s in per_cam.items():
        q = lambda v, p: float(np.percentile(v, p)) if v else float("nan")  # noqa: E731
        row = {
            "boxes": s["n"],
            "invalid": s["invalid"] / max(s["n"], 1),
            "feet_median_px": q(s["feet"], 50),
            "feet_p90_px": q(s["feet"], 90),
            "feet_rel_median": q(s["rel_feet"], 50),
            "head_median_px": q(s["head"], 50),
            "head_p90_px": q(s["head"], 90),
            "head_rel_median": q(s["rel_head"], 50),
            "visible_cells": float(inside[c].mean()),
            "offaxis_rejected_cells": int(rejected[c].sum()),
        }
        result["cams"][c] = row
        print(
            f"{c:>4} {row['boxes']:>6} {100 * row['invalid']:>7.1f}% {row['feet_median_px']:>9.1f} "
            f"{row['feet_p90_px']:>6.1f} {row['feet_rel_median']:>6.2f} {row['head_median_px']:>9.1f} "
            f"{row['head_p90_px']:>6.1f} {row['head_rel_median']:>6.2f} {100 * row['visible_cells']:>13.1f}% "
            f"{row['offaxis_rejected_cells']:>17}"
        )

    if args.images:
        f = frames[0]
        for c in range(ds.NUM_CAMS):
            img = Image.open(ds.image_path(f, c)).convert("RGB").resize((W_IMG, H_IMG))
            draw = ImageDraw.Draw(img)
            for x, y, boxes in ds.people[f]:
                if c not in boxes:
                    continue
                x0, y0, x1, y1 = boxes[c]
                for px, py in ((x0 + x1) / 2 * sx, y1 * sy), ((x0 + x1) / 2 * sx, y0 * sy):
                    draw.ellipse([px - 4, py - 4, px + 4, py + 4], outline=(0, 255, 0), width=2)
                row, col = int((y - h_min + b) * r), int((x - w_min + a) * r)
                for k, colour in ((0, (255, 0, 0)), (1, (0, 128, 255))):
                    if refs[c, row, col, k, 0] != -10:
                        px, py = to_pixels(refs[c, row, col, k])
                        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=colour)
            img.save(os.path.join(args.out_dir, f"{name}_frame{f}_cam{c}.jpg"), quality=85)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reprojection check of the cross-dataset reference points")
    parser.add_argument("--datasets", type=str, default="wildtrack,multiviewx")
    parser.add_argument(
        "--data_root", type=str, default=None, help="Folder holding Wildtrack/ and MultiviewX/ (default src/datasets)."
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--frames", type=int, default=20, help="Frames per dataset, evenly spaced.")
    parser.add_argument("--head_height", type=float, default=1.75)
    parser.add_argument("--images", action="store_true", help="Draw one frame per camera.")
    parser.add_argument("--out_dir", type=str, default=os.path.join(src_dir, "runs", "cross_dataset", "refs_check"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    results = {name: check(name, args) for name in args.datasets.split(",")}
    out = os.path.join(args.out_dir, "refs_check.json")
    with open(out, "w") as f:
        json.dump({"head_height": args.head_height, "split": args.split, "frames": args.frames, **results}, f, indent=2)
    print(f"\nSaved {out}")
