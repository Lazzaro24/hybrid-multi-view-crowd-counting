"""
occluder_masks.py

Static occluder masks per camera (trees, umbrellas, roofs, building parts) from the data itself, for the occlusion
evaluation (evaluation/eval_occlusion.py). SynMVCrowd has no visibility labels: a person's `pixel` is present
whenever the head is inside the image, also when it is hidden.

Idea: occluders do not move, people do. A grid cell of the image is an occluder if annotated heads project into it
regularly but the image there looks no different with a head underneath than without one.

  1. Snapshots are grouped by lighting (k-means on 32x18 grey thumbnails; time of day and weather vary a lot). Each
     group gets a per-pixel median background and the pixels' own temporal variation (median absolute deviation +
     --noise_floor grey levels). Groups darker than --min_brightness (night) are not used in steps 2-3.
  2. Per snapshot, a change map: |image - background| / variation, averaged over each grid cell (--cell px at half
     resolution). Normalising by the variation makes swaying palm fronds and moving shadows comparable to still ground.
  3. Per cell, contrast = mean change in the windows of the heads that project into it (window of the head's projected
     size, --head_radius m) / mean change of the cell in snapshots where no person's projected body (0.25 m-radius
     cylinder, feet to head) covers it. A visible person raises the change, a hidden one does not. Cells with
     >= --min_hits heads, >= --min_empty truly empty snapshots and contrast < --contrast_thr are occluder cells; cells
     that are never empty (dense crowds) stay unlabelled instead of being guessed. "Empty" must exclude bodies, not
     just heads: in crowds the bodies below the heads would otherwise make every baseline look busy. The mask is closed morphologically and patches smaller than
     --min_component cells (isolated noise cells in busy areas) are removed.

Head positions are the dataset's own `pixel` annotations (exact), not reprojections of the 3D head points: the
reprojection with the json calibration is off by tens of pixels in some scenes (see pixel_offsets).

This is a label for static occluders only. People hidden by other people come from the cylinder z-buffer in
eval_occlusion.py.

Outputs in diagnostics/occluders/<scene>/: cam<id>_mask.png (255 = occluder, half resolution 960x540),
cam<id>_overlay.jpg (red = occluder, green = cell with enough data and visible people), spotcheck_{hidden,visible}.jpg
(random daylight heads per label, for checking by eye), cam<id>_contrast.npz
(per-cell contrast, per-pixel empty change, counts) and occluders.json (parameters and per-camera statistics).

Usage (from src):
    uv run python utils/occluder_masks.py --scenes validate/scene_1
    uv run python utils/occluder_masks.py --scenes heldout          # every validate and test scene, skips finished ones
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

IMG_W, IMG_H = 1920, 1080
DS = 2  # work at half resolution


def project(cam, pts):
    K = np.array(cam["cameraMatrix"], np.float64)
    K[0] *= IMG_W
    K[1] *= IMG_H
    rvec, tvec = np.array(cam["rvec"], np.float64), np.array(cam["tvec"], np.float64)
    ip, _ = cv2.projectPoints(pts.reshape(-1, 3), rvec, tvec, K, np.array(cam["distCoeffs"], np.float64))
    R, _ = cv2.Rodrigues(rvec)
    depth = (R @ pts.reshape(-1, 3).T).T[:, 2] + tvec[2]
    return ip.reshape(-1, 2), depth, K[0, 0]


def pixel_offsets(meta):
    """Per person: the dataset's own head pixel (exact, 1920x1080; `pixel` is [row, col]) minus our reprojection of
    the head point, or 0 when the person has no `pixel` (head outside the image). The reprojection with the json's
    calibration is off by up to tens of pixels in some scenes/cameras (scene_49, scene_25, scene_16 cam 11, ...), so
    head positions come from `pixel` and every projected body point is shifted by the same offset."""
    heads = np.array([p["world"] for p in meta["image_info"]], np.float64)
    uv, depth, fx = project(meta, heads)
    off = np.zeros_like(uv)
    for i, p in enumerate(meta["image_info"]):
        if p.get("pixel") is not None:
            off[i] = [p["pixel"][1] - uv[i, 0], p["pixel"][0] - uv[i, 1]]
    return heads, uv, depth, fx, off


def body_cover(meta, h, w):
    """Half-resolution mask of every annotated person's projected body: a 0.25 m-radius cylinder from the feet
    (head point - 1.75 m) to 0.10 m above the head point, shifted onto the dataset's head pixel."""
    heads, _, _, fx, off = pixel_offsets(meta)
    top, dt, _ = project(meta, heads + [0, 0, 0.10])
    feet, _, _ = project(meta, heads - [0, 0, 1.75])
    top, feet = top + off, feet + off
    mask = np.zeros((h, w), np.uint8)
    for a, b, d in zip(top / DS, feet / DS, dt):
        if d <= 0.5 or np.abs(np.concatenate([a, b])).max() > 1e5:
            continue
        ax = a - b
        n = np.array([-ax[1], ax[0]]) / max(np.linalg.norm(ax), 1e-6) * fx * 0.25 / d / DS
        cv2.fillConvexPoly(mask, np.round(np.array([a + n, a - n, b - n, b + n])).astype(np.int32), 1)
    return mask


def heads_in_view(meta, head_radius):
    """Half-resolution centre and window radius of every annotated head inside the image (centre = the dataset's
    head pixel; only people that have one)."""
    _, uv, depth, fx, off = pixel_offsets(meta)
    has = np.array([p.get("pixel") is not None for p in meta["image_info"]], bool)
    uv = uv + off
    ok = has & (depth > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)
    r = np.maximum(1, np.round(fx * head_radius / depth[ok] / DS)).astype(int)
    return uv[ok] / DS, r


def kmeans(x, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    return lab


def lighting_groups(imgs, k, min_size):
    thumbs = np.stack([cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (32, 18)).ravel() for im in imgs])
    thumbs = thumbs.astype(np.float32) / 255.0
    lab = kmeans(thumbs, min(k, len(imgs)))
    for _ in range(k):  # merge groups that are too small
        sizes = np.bincount(lab, minlength=lab.max() + 1)
        small = [g for g in range(len(sizes)) if 0 < sizes[g] < min_size]
        if not small:
            break
        g = small[0]
        cent = {j: thumbs[lab == j].mean(0) for j in range(len(sizes)) if sizes[j] and j != g}
        lab[lab == g] = min(cent, key=lambda j: ((cent[j] - thumbs[lab == g].mean(0)) ** 2).sum())
    return lab


def main():
    ap = argparse.ArgumentParser(description="Static occluder masks per camera from moving people.")
    ap.add_argument(
        "--scenes",
        default="validate/scene_1",
        help="comma-separated split/scene, or 'heldout' for every validate and test scene",
    )
    ap.add_argument("--overwrite", action="store_true", help="rebuild scenes that already have all masks")
    ap.add_argument("--spotcheck", type=int, default=30, help="heads per label in the spot-check montage")
    ap.add_argument("--cams", default=None, help="comma-separated camera ids (default: the K-means 5-camera set)")
    ap.add_argument("--groups", type=int, default=8, help="lighting groups")
    ap.add_argument("--min_group", type=int, default=8, help="smaller groups are merged into the closest one")
    ap.add_argument("--noise_floor", type=float, default=4.0, help="added to the per-pixel MAD (grey levels)")
    ap.add_argument("--cell", type=int, default=8, help="grid cell size in half-resolution px")
    ap.add_argument("--head_radius", type=float, default=0.15, help="half-size of a head window [m]")
    ap.add_argument("--min_hits", type=int, default=5)
    ap.add_argument("--min_empty", type=int, default=5)
    ap.add_argument("--contrast_thr", type=float, default=1.15, help="head / empty change below this: occluder")
    ap.add_argument(
        "--min_brightness",
        type=float,
        default=60.0,
        help="skip lighting groups darker than this (mean grey level): at night the change of a visible "
        "head is small next to the noise floor, which pushes every contrast towards 1",
    )
    ap.add_argument("--min_component", type=int, default=6, help="remove occluder patches smaller than this [cells]")
    args = ap.parse_args()
    if args.scenes == "heldout":
        specs = []
        for split in ("validate", "test"):
            d = os.path.join(src_dir, "datasets", "SynMVCrowd", "data", split)
            specs += [
                f"{split}/{x}"
                for x in sorted(os.listdir(d), key=lambda x: int(x.split("_")[1]) if "_" in x else 0)
                if os.path.isdir(os.path.join(d, x))
            ]
    else:
        specs = args.scenes.split(",")
    with open(os.path.join(src_dir, "utils", "excluded_scenes.json")) as f:
        excluded = json.load(f)
    for spec in specs:
        scene = spec.split("/")[1]
        if scene in excluded:
            print(f"[skip] {spec}: excluded")
            continue
        out_dir = os.path.join(src_dir, "diagnostics", "occluders", scene)
        with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
            cams = [int(c) for c in args.cams.split(",")] if args.cams else json.load(f)[scene]["5"]
        if not args.overwrite and all(os.path.exists(os.path.join(out_dir, f"cam{c}_mask.png")) for c in cams):
            print(f"[skip] {spec}: masks exist (--overwrite to rebuild)")
            continue
        build_scene(spec, args)


def build_scene(spec, args):
    split, scene = spec.split("/")
    scene_dir = os.path.join(src_dir, "datasets", "SynMVCrowd", "data", split, scene)
    if args.cams:
        cams = [int(c) for c in args.cams.split(",")]
    else:
        with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
            cams = json.load(f)[scene]["5"]
    snaps = sorted([s for s in os.listdir(scene_dir) if s.isdigit()], key=int)
    out_dir = os.path.join(src_dir, "diagnostics", "occluders", scene)
    os.makedirs(out_dir, exist_ok=True)
    w, h = IMG_W // DS, IMG_H // DS
    gh, gw = h // args.cell, w // args.cell
    summary_path = os.path.join(out_dir, "occluders.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {"per_cam": {}}
    summary.update({"scene": spec, "snapshots": len(snaps), "params": vars(args), "mask_resolution": [w, h]})

    for cam in cams:
        imgs, metas = [], []
        for s in tqdm(snaps, desc=f"cam {cam} load", mininterval=10):
            ip = os.path.join(scene_dir, s, "jpgs", f"{cam}.jpg")
            jp = os.path.join(scene_dir, s, "jsons", f"{cam}.json")
            if not (os.path.exists(ip) and os.path.exists(jp)):
                continue
            imgs.append(cv2.resize(cv2.imread(ip), (w, h), interpolation=cv2.INTER_AREA))
            with open(jp) as f:
                metas.append(json.load(f))
        imgs = np.stack(imgs)  # [T, h, w, 3] uint8
        T = len(imgs)

        # 1. lighting groups: background and temporal variation
        lab = lighting_groups(imgs, args.groups, args.min_group)
        bgs, mads = {}, {}
        for g in np.unique(lab):
            stack = imgs[lab == g].astype(np.float32)
            bgs[g] = np.median(stack, axis=0)
            mads[g] = np.median(np.abs(stack - bgs[g]), axis=0).mean(-1) + args.noise_floor
            del stack

        bright = {g: float(bgs[g].mean()) for g in bgs}
        use = np.array([bright[lab[t]] >= args.min_brightness for t in range(T)])

        def change_map(t):
            return np.abs(imgs[t].astype(np.float32) - bgs[lab[t]]).mean(-1) / mads[lab[t]]  # noqa: F821  (imgs is deleted only after the last call)

        # 2. per cell: mean change in snapshots where no person's projected body covers the cell ("empty")
        heads = [heads_in_view(metas[t], args.head_radius) for t in range(T)]
        e_sum = np.zeros((gh, gw))
        n_empty = np.zeros((gh, gw), np.int32)
        for t in np.where(use)[0]:
            cells = cv2.resize(
                change_map(t)[: gh * args.cell, : gw * args.cell], (gw, gh), interpolation=cv2.INTER_AREA
            )
            cover = (
                cv2.resize(
                    body_cover(metas[t], h, w)[: gh * args.cell, : gw * args.cell].astype(np.float32),
                    (gw, gh),
                    interpolation=cv2.INTER_AREA,
                )
                > 0
            )
            e_sum += cells * ~cover
            n_empty += ~cover
        e_mean = e_sum / np.maximum(n_empty, 1)

        # 3. per cell: mean change in the head windows of the heads that project into it, relative to "empty"
        h_sum = np.zeros((gh, gw))
        n_head = np.zeros((gh, gw), np.int32)
        for t in np.where(use)[0]:
            ch = change_map(t)
            for (x, y), r in zip(*heads[t]):
                x0, x1 = max(int(x) - r, 0), min(int(x) + r + 1, w)
                y0, y1 = max(int(y) - r, 0), min(int(y) + r + 1, h)
                cy, cx = min(int(y) // args.cell, gh - 1), min(int(x) // args.cell, gw - 1)
                h_sum[cy, cx] += ch[y0:y1, x0:x1].mean()
                n_head[cy, cx] += 1
        contrast = (h_sum / np.maximum(n_head, 1)) / np.maximum(e_mean, 1e-3)
        enough = (n_head >= args.min_hits) & (n_empty >= args.min_empty)
        occ = (enough & (contrast < args.contrast_thr)).astype(np.uint8)
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
        for j in range(1, n_cc):
            if stats[j, cv2.CC_STAT_AREA] < args.min_component:
                occ[cc == j] = 0
        np.savez(
            os.path.join(out_dir, f"cam{cam}_contrast.npz"),
            contrast=contrast,
            empty_change=e_mean,
            n_head=n_head,
            n_empty=n_empty,
            occ=occ,
        )

        def up(a):
            u = cv2.resize(a.astype(np.uint8), (gw * args.cell, gh * args.cell), interpolation=cv2.INTER_NEAREST)
            return cv2.copyMakeBorder(u, 0, h - u.shape[0], 0, w - u.shape[1], cv2.BORDER_CONSTANT, value=0)

        mask = up(occ)
        cv2.imwrite(os.path.join(out_dir, f"cam{cam}_mask.png"), mask * 255)
        ov = imgs[T // 2].copy()
        tint = np.zeros_like(ov)
        tint[up(enough & (occ == 0)) > 0] = (0, 160, 0)
        tint[mask > 0] = (0, 0, 255)
        cv2.imwrite(os.path.join(out_dir, f"cam{cam}_overlay.jpg"), cv2.addWeighted(ov, 1.0, tint, 0.45, 0))

        cvals = contrast[enough] if enough.any() else np.array([np.nan])
        summary["per_cam"][str(cam)] = {
            "frames": T,
            "frames_used": int(use.sum()),
            "lighting_groups": {int(g): [int((lab == g).sum()), round(bright[g], 1)] for g in np.unique(lab)},
            "cells_with_data": int(enough.sum()),
            "occluder_cells": int(occ.sum()),
            "occluder_share_of_image": float(mask.mean()),
            "contrast_percentiles_5_25_50_75_95": np.percentile(cvals, [5, 25, 50, 75, 95]).round(2).tolist(),
        }
        print(
            f"cam {cam}: {T} frames, lighting groups {summary['per_cam'][str(cam)]['lighting_groups']}, "
            f"{int(enough.sum())} cells with data, contrast p5/25/50/75/95 "
            f"{summary['per_cam'][str(cam)]['contrast_percentiles_5_25_50_75_95']}, "
            f"occluder {mask.mean():.1%} of the image"
        )
        del imgs

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    spot_check(scene_dir, cams, snaps, out_dir, args.spotcheck, args.min_brightness)
    print(f"saved {out_dir}")


def spot_check(scene_dir, cams, snaps, out_dir, n, min_brightness, seed=0):
    """Montages of random heads (daylight snapshots) labelled hidden / visible by the masks, crop centred on the
    projected head (magenta cross), for checking the labels by eye: spotcheck_hidden.jpg, spotcheck_visible.jpg."""
    import random

    masks = {c: cv2.imread(os.path.join(out_dir, f"cam{c}_mask.png"), 0) > 0 for c in cams}
    rng = random.Random(seed)
    picks = {True: [], False: []}
    for _ in range(50 * n):
        if all(len(v) >= n for v in picks.values()):
            break
        c, sn = rng.choice(cams), rng.choice(snaps)
        jp = os.path.join(scene_dir, sn, "jsons", f"{c}.json")
        if not os.path.exists(jp):
            continue
        with open(jp) as f:
            meta = json.load(f)
        if not meta["image_info"]:
            continue
        person = rng.choice(meta["image_info"])
        if person.get("pixel") is None:
            continue
        _, depth, fx = project(meta, np.array([person["world"]], np.float64))
        (u, v), d = (person["pixel"][1], person["pixel"][0]), depth[0]
        if d <= 0.5 or not (40 <= u < IMG_W - 40 and 40 <= v < IMG_H - 40):
            continue
        lab = bool(masks[c][int(v / DS), int(u / DS)])
        if len(picks[lab]) >= n:
            continue
        im = cv2.imread(os.path.join(scene_dir, sn, "jpgs", f"{c}.jpg"))
        if im is None or im.mean() < min_brightness:
            continue
        r = int(max(24, fx / d * 0.18))
        crop = cv2.resize(im[max(int(v) - r, 0) : int(v) + r, max(int(u) - r, 0) : int(u) + r], (120, 120))
        cv2.drawMarker(crop, (60, 60), (255, 0, 255), cv2.MARKER_CROSS, 14, 1)
        picks[lab].append(crop)
    for lab, name in ((True, "hidden"), (False, "visible")):
        tiles = picks[lab] + [np.zeros((120, 120, 3), np.uint8)] * (-len(picks[lab]) % 10)
        if tiles:
            rows = [np.hstack(tiles[i : i + 10]) for i in range(0, len(tiles), 10)]
            cv2.imwrite(os.path.join(out_dir, f"spotcheck_{name}.jpg"), np.vstack(rows))


if __name__ == "__main__":
    main()
