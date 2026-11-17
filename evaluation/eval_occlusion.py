"""
eval_occlusion.py

Do the fusion models miss occluded people? Per-person evaluation of neck_ablation.py models on chosen scenes.

Per person (GT head point, world x, y, z; the same people the count evaluation uses) and camera of the sample:
  in_fov    the person has the dataset's head `pixel` in this camera (= head inside the image). Head positions are
            that `pixel`, not reprojections: the json calibration misses by tens of pixels in some scenes.
  vis_geo   fraction of the head not covered by nearer annotated people. Every person is a cylinder (radius
            --radius_body m) from 1.75 m below the head point to 0.10 m above it; per camera the cylinders are painted
            far to near into a full-resolution ID buffer, and the head part (top --head_len m) is checked against it.
            Sees person-person occlusion only: umbrellas, roofs and trees are not annotated.
  static    the head falls on a static occluder (tree, umbrella, roof, building) in this camera, from the masks of
            utils/occluder_masks.py (diagnostics/occluders/<scene>/cam<id>_mask.png). Only frames with mean grey level
            >= --min_brightness are evaluated when masks are used (the masks are built from daylight frames).
  ev_2d     evidence of the 2D-pretrained head model at the projected head (5x5 window of the stride-8 map). NOT a
            visibility label: the 2D model was trained on hidden heads too and predicts heads under umbrellas. Kept
            only as a diagnostic.
Per person in BEV: recovery = predicted density mass / GT density mass within --radius m of the head (GT built with
the training kernel); local density = GT people within 2 m. Both from the whole-scene map, as in the count evaluation.

Reported per model: recovery and miss rate (recovery < 0.5) by the number of cameras that see the head, for each
visibility label, overall and within local-density strata (occlusion grows with density, so both must be separated),
and a regression recovery ~ cameras in view + views blocked by people + views blocked by static occluders + local
density, with 95% intervals from resampling whole frames; for several scenes, the within-scene version (one intercept
per scene; the pooled one is misleading across scenes) with intervals from resampling whole scenes, the paired
difference of the second model to the first, and the coefficients of every scene on its own.
Overlay images of the first frame show the labels on every camera image, to check them by eye.

Usage (from src):
    uv run python evaluation/eval_occlusion.py --scenes train/scene_14 --frames 20
    uv run python evaluation/eval_occlusion.py --scenes train/scene_14 --models ablation_v2:homography_256,ablation_k8:hca
    uv run python evaluation/eval_occlusion.py --models bud200_h0ctrl:homography_256         --proposed h1_hom=<checkpoint>:learned --baselines ot=runs/baseline/ot/best.pth
Writes diagnostics/occlusion/<tag>/{persons.csv,summary.json,*.jpg}.
"""

import os
import sys
import csv
import json
import random
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))


import neck_ablation as NA
import evaluate_paper_protocol as EP
from diagnose_hca import load_model
from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
from losses.gaussian_mse import get_map_kernel

IMG_W, IMG_H = 1920, 1080  # camera intrinsics are normalised to the original image size
PERSON_HEIGHT = 1.75  # = d_delta in every scene
CELLS_PER_M = 5.0  # wld_map_paras r


# ==========================================
# GEOMETRY: projection and person-person occlusion
# ==========================================


def camera_matrices(paras):
    """camera_paras row [fx, fy, u, v, k1, k2, p1, p2, k3, rvec(3), tvec(3)] (normalised intrinsics)."""
    p = paras.cpu().numpy().astype(np.float64)
    K = np.array([[p[0] * IMG_W, 0, p[2] * IMG_W], [0, p[1] * IMG_H, p[3] * IMG_H], [0, 0, 1]])
    return K, p[4:9], p[9:12], p[12:15]


def project(K, dist, rvec, tvec, pts):
    ip, _ = cv2.projectPoints(pts.reshape(-1, 3), rvec, tvec, K, dist)
    R, _ = cv2.Rodrigues(rvec)
    depth = (R @ pts.reshape(-1, 3).T).T[:, 2] + tvec[2]
    return ip.reshape(-1, 2), depth


def quad(top, bottom, half_w):
    """Image-space rectangle around the projected axis top -> bottom, half_w px wide on each side."""
    a = top - bottom
    n = np.array([-a[1], a[0]]) / max(np.linalg.norm(a), 1e-6)
    return np.array([top + n * half_w, top - n * half_w, bottom - n * half_w, bottom + n * half_w])


def visibility_geo(K, dist, rvec, tvec, heads, head_len, radius, pix=None):
    """Per person: (in_fov, visible fraction of the head, head position) in this camera, from cylinders painted far
    to near. pix: [P, 2] the dataset's head pixel (x, y) in this camera, NaN where the person has none. The reprojection
    of the 3D head with the json calibration is off by up to tens of pixels in some scenes, so the head is placed on
    `pixel` and every projected point of that person is shifted by the same offset; in view = has a `pixel`."""
    P = len(heads)
    top = heads + [0, 0, 0.10]
    head_bot = heads - [0, 0, head_len]
    feet = heads - [0, 0, PERSON_HEIGHT]
    t2, d_top = project(K, dist, rvec, tvec, top)
    hb2, _ = project(K, dist, rvec, tvec, head_bot)
    f2, _ = project(K, dist, rvec, tvec, feet)
    h2, d_head = project(K, dist, rvec, tvec, heads)
    if pix is not None:
        has = ~np.isnan(pix[:, 0])
        off = np.where(has[:, None], pix - h2, 0.0)
        t2, hb2, f2, h2 = t2 + off, hb2 + off, f2 + off, h2 + off
        in_fov = has & (d_head > 0.5)
    else:
        in_fov = (d_head > 0.5) & (h2[:, 0] >= 0) & (h2[:, 0] < IMG_W) & (h2[:, 1] >= 0) & (h2[:, 1] < IMG_H)
    half_w = K[0, 0] * radius / np.maximum(d_top, 0.5)
    buf = np.zeros((IMG_H, IMG_W), np.int32)
    front = np.where(d_head > 0.5)[0]
    for i in front[np.argsort(-d_head[front])]:  # far to near: nearer people overwrite
        q = quad(t2[i], f2[i], half_w[i])
        if np.abs(q).max() > 1e5:
            continue
        cv2.fillConvexPoly(buf, np.round(q).astype(np.int32), int(i) + 1)
    vis = np.full(P, np.nan)
    for i in np.where(in_fov)[0]:
        q = np.round(quad(t2[i], hb2[i], half_w[i])).astype(np.int32)
        x0, y0 = np.maximum(q.min(0), 0)
        x1, y1 = np.minimum(q.max(0) + 1, [IMG_W, IMG_H])
        if x1 <= x0 or y1 <= y0:
            continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(m, q - [x0, y0], 1)
        area = m.sum()
        vis[i] = float(((buf[y0:y1, x0:x1] == i + 1) & (m > 0)).sum() / area) if area else np.nan
    return in_fov, vis, h2


# ==========================================
# MODELS
# ==========================================


class Head2D(nn.Module):
    """head_2d of the 2D pretraining (training_scripts/train_2d_pretrain.py), loaded from the backbone checkpoint."""

    def __init__(self, ckpt_path):
        super().__init__()
        self.head_2d = nn.Sequential(nn.Conv2d(512, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
        self.head_2d.load_state_dict({k[len("head_2d.") :]: v for k, v in sd.items() if k.startswith("head_2d.")})

    def forward(self, feats):  # [B, N, 512, h, w] -> [N, h, w] density, count units
        B, N = feats.shape[:2]
        return self.head_2d(feats.flatten(0, 1).float())[:, 0] / NA.SCALE


@torch.no_grad()
def predict_scene(model, feats, ctx, H, W, device):
    pred = torch.zeros((H, W), device=device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for h0 in NA.tile_starts(H, NA.TILE[0]):
            for w0 in NA.tile_starts(W, NA.TILE[1]):
                out = model.predict_tile(feats, ctx, h0, w0)[0, 0].float()
                hh, ww = min(NA.TILE[0], H - h0), min(NA.TILE[1], W - w0)
                pred[h0 : h0 + hh, w0 : w0 + ww] = out[:hh, :ww]
    return pred / NA.SCALE


@torch.no_grad()
def predict_proposed_scene(model, routing, s, device):
    """Whole-scene map (persons per cell) of a proposed checkpoint (train_proposed_multiscene.py) for one unbatched
    sample, with the paper protocol's routing code: 'learned' (the router) | 'dense' | 'sparse' | 'dzone<thr>' (the
    density rule), each but 'learned' with an optional trailing 'p' (sparse expert as NMS peaks; applied at load time)."""
    sample = {k: s[k][None] for k in ("images", "ref_points_full", "camera_paras", "wld_map_paras")}
    shape = (int(s["wld_map_paras"][3].item()), int(s["wld_map_paras"][4].item()))
    base, _ = EP.parse_routing(routing)
    if base == "learned":
        zone = None
    elif base in ("dense", "sparse"):
        zone = torch.full(shape, base == "dense", dtype=torch.bool)
    elif base.startswith("dzone"):
        zone = EP.density_zone(model, sample, device, float(base[len("dzone") :]), shape)
    else:
        raise SystemExit(
            f"--proposed routing {routing!r}: use learned | dense | sparse | dzone<thr>, optional trailing p"
        )
    return EP.predict_proposed(model, sample, device, zone) / EP.SCALE


@torch.no_grad()
def predict_baseline_scene(model, s, device):
    """Whole-scene map (persons per cell) of a SynMVCrowd baseline checkpoint (train_baseline.py) for one
    unbatched sample. It sees the same 5 cameras as the other models; it was trained on random 5-of-all draws."""
    sample = {k: s[k][None] for k in ("images", "camera_paras", "wld_map_paras")}
    return EP.predict_baseline(model, sample, device) / EP.SCALE


def disk_sum(m, radius_cells):
    """Sum of a [H, W] map within radius_cells of every cell."""
    r = radius_cells
    yy, xx = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij")
    k = ((yy**2 + xx**2) <= r * r).float().to(m.device)[None, None]
    return F.conv2d(m[None, None], k, padding=r)[0, 0]


# ==========================================
# MAIN LOOP
# ==========================================


def active_camera_ids(ds, snapshot, camera_paras):
    """The sample's cameras are a shuffled draw; identify them by their extrinsics."""
    ids = []
    for row in camera_paras:
        rv = row[9:12].numpy()
        for c in ds.camera_ids:
            with open(os.path.join(ds.scene_dir, str(snapshot), "jsons", f"{c}.json")) as f:
                if np.allclose(json.load(f)["rvec"], rv, atol=1e-4):
                    ids.append(c)
                    break
        else:
            ids.append(None)
    return ids


def main():
    ap = argparse.ArgumentParser(description="Per-person occlusion evaluation of neck_ablation models.")
    ap.add_argument("--scenes", default="train/scene_14", help="comma-separated split/scene")
    ap.add_argument("--models", default="ablation_v2:homography_256", help="comma-separated run:variant")
    ap.add_argument(
        "--proposed",
        default="",
        help="comma-separated name=checkpoint:routing of proposed checkpoints, scored after --models "
        "(routing dense | sparse | dzone<thr>, optional trailing p, e.g. dzone0.15p); their fusion "
        "must use the same cameras / refs as the first --models entry",
    )
    ap.add_argument(
        "--baselines",
        default="",
        help="comma-separated name=checkpoint of SynMVCrowd baseline checkpoints (train_baseline.py), "
        "scored after --proposed on the same cameras, e.g. ot=runs/baseline/ot/best.pth",
    )
    ap.add_argument("--frames", type=int, default=20, help="frames per scene, evenly spaced")
    ap.add_argument("--radius", type=float, default=1.0, help="BEV disk radius for recovery [m]")
    ap.add_argument("--head_len", type=float, default=0.30, help="head part of the cylinder [m]")
    ap.add_argument("--radius_body", type=float, default=0.25, help="cylinder radius [m]")
    ap.add_argument("--vis_thr", type=float, default=0.5, help="vis_geo >= this: head visible")
    ap.add_argument("--ev_thr", type=float, default=0.4, help="ev_2d >= this: head detected by the 2D model")
    ap.add_argument(
        "--occluders",
        default="auto",
        help="'auto': use diagnostics/occluders/<scene>/cam<id>_mask.png when present; 'none': no masks",
    )
    ap.add_argument("--min_brightness", type=float, default=60.0, help="with masks: skip darker (night) frames")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    specs = [s.split(":") for s in args.models.split(",")]
    variants = [v for _, v in specs]
    for run, variant in specs:
        m, a = load_model(
            os.path.join(src_dir, "runs", "neck_ablation", run),
            variant,
            {"z_anchors": 4, "refs_file": "master_refs.pt"},
            device,
        )  # defaults of runs before these args existed
        # the variant names the model; with two runs of the same variant the run name is added so their columns differ
        models.append((variant if variants.count(variant) == 1 else f"{run}_{variant}", m, a))
    refs_file = next(
        (getattr(a, "refs_file", "master_refs.pt") for _, _, a in models if getattr(a, "z_anchors", 4) != 4),
        "master_refs.pt",
    )
    z_anchors = 8 if refs_file != "master_refs.pt" else 4
    a0 = models[0][2]

    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    excluded = NA.load_excluded_scenes()
    train = NA.load_split("train", cams_cfg, excluded, refs_file, z_anchors)
    for _, m, _ in models:  # precise-BN exactly as before the ablation's evaluation
        NA.precise_bn(m, NA.make_loader(train, 1, a0.val_seed + 7, 0), device)
    del train
    # proposed checkpoints: stored BN statistics, as in eval_map_quality.py / the paper protocol
    proposed = []
    for spec in [p for p in args.proposed.split(",") if p]:
        name, rest = spec.split("=", 1)
        weights, routing = rest.rsplit(":", 1)
        pm, _ = EP.load_model("proposed", os.path.join(src_dir, weights), device)
        if routing.endswith("p"):
            pm.T.apply_sparse_render(pm, {"mode": "peaks", "kernel": 3, "threshold": 0.064})
            pm.head.render_kernel = pm.head.render_kernel.to(device)
        print(f"[load] proposed {name}: {weights} ({routing})")
        proposed.append((name, pm, routing))
    baselines = []
    for spec in [b for b in args.baselines.split(",") if b]:
        name, weights = spec.split("=", 1)
        bm, epoch = EP.load_model("baseline", os.path.join(src_dir, weights), device)
        print(f"[load] baseline {name}: {weights} (epoch {epoch})")
        baselines.append((name, bm))
    head2d = Head2D(a0.backbone).to(device).eval()

    tag = args.tag or (
        "_".join(s.split("/")[-1] for s in args.scenes.split(","))
        + "__"
        + "_".join([v for v, _, _ in models] + [n for n, _, _ in proposed] + [n for n, _ in baselines])
    )
    out_dir = os.path.join(src_dir, "diagnostics", "occlusion", tag)
    os.makedirs(out_dir, exist_ok=True)
    kernel = get_map_kernel(sigma=NA.MAP_SIGMA, kernel_size=NA.MAP_KERNEL_SIZE).to(device)
    kernel = (kernel / kernel.sum()).view(1, 1, *kernel.shape[-2:])
    rc = int(round(args.radius * CELLS_PER_M))

    rows = []
    for spec in args.scenes.split(","):
        split, scene = spec.split("/")
        ds = SynMVCrowdDataset(
            scene_dir=os.path.join(src_dir, "datasets", "SynMVCrowd", "data", split, scene),
            camera_ids=cams_cfg[scene][str(NA.NUM_CAMS)],
            num_sampled_cams=NA.NUM_CAMS,
            load_refs=True,
            refs_file=refs_file,
        )
        occ_dir = os.path.join(src_dir, "diagnostics", "occluders", scene)
        masks = {}
        if args.occluders == "auto":
            for c in ds.camera_ids:
                mp = os.path.join(occ_dir, f"cam{c}_mask.png")
                if os.path.exists(mp):
                    masks[c] = cv2.imread(mp, 0) > 0
            if masks and len(masks) < len(ds.camera_ids):
                raise SystemExit(f"{scene}: occluder masks for only {sorted(masks)} of {ds.camera_ids}")
        print(f"{scene}: static occluder masks {'used' if masks else 'not available'}")
        frame_idx = np.unique(np.linspace(0, len(ds) - 1, min(args.frames, len(ds))).round().astype(int))
        first = True
        n_dark = 0
        for idx in tqdm(frame_idx, desc=scene):
            snap0 = ds.snapshots[int(idx)]
            if masks:
                grey = np.mean(
                    [
                        cv2.imread(os.path.join(ds.scene_dir, str(snap0), "jpgs", f"{c}.jpg"), 0).mean()
                        for c in ds.camera_ids
                    ]
                )
                if grey < args.min_brightness:
                    n_dark += 1
                    continue
            random.seed(args.seed * 100003 + int(idx))
            s = ds[int(idx)]
            snap = s["snapshot_id"]
            cam_ids = active_camera_ids(ds, snap, s["camera_paras"])
            with open(os.path.join(ds.scene_dir, str(snap), "jsons", f"{cam_ids[0]}.json")) as f:
                people = json.load(f)["image_info"]
            seen = set()  # the count evaluation's people
            pixmaps = []  # per camera: idx -> dataset head pixel (x, y)
            for c in cam_ids:
                with open(os.path.join(ds.scene_dir, str(snap), "jsons", f"{c}.json")) as f:
                    pm = {
                        p["idx"]: (p["pixel"][1], p["pixel"][0])
                        for p in json.load(f)["image_info"]
                        if p.get("pixel") is not None
                    }
                pixmaps.append(pm)
                seen |= set(pm)
            people = [p for p in people if p["idx"] in seen]
            heads = np.array([p["world"] for p in people], np.float64)
            Pn, N = len(people), len(cam_ids)

            # --- per camera labels
            in_fov = np.zeros((Pn, N), bool)
            vis = np.full((Pn, N), np.nan)
            ev = np.full((Pn, N), np.nan)
            static = np.zeros((Pn, N), bool)
            wld = s["wld_map_paras"]
            ctx = {
                "ref_points_full": s["ref_points_full"][None].to(device),
                "camera_paras": s["camera_paras"][None].to(device),
                "wld_map_paras": wld[None].to(device),
            }
            with torch.no_grad():
                feats = models[0][1].features(s["images"][None].to(device))
                d2 = head2d(feats)  # [N, h, w]
            d2_win = F.avg_pool2d(d2[:, None], 5, stride=1, padding=2)[:, 0] * 25  # 5x5 window sums
            fh, fw = d2.shape[-2:]
            heads2d = []
            for n in range(N):
                K, dist, rvec, tvec = camera_matrices(s["camera_paras"][n])
                pix = np.array([pixmaps[n].get(p["idx"], (np.nan, np.nan)) for p in people], np.float64)
                in_fov[:, n], vis[:, n], h2 = visibility_geo(
                    K, dist, rvec, tvec, heads, args.head_len, args.radius_body, pix
                )
                heads2d.append(h2)
                cx = np.clip((h2[:, 0] * fw / IMG_W).astype(int), 0, fw - 1)
                cy = np.clip((h2[:, 1] * fh / IMG_H).astype(int), 0, fh - 1)
                e = d2_win[n].cpu().numpy()[cy, cx]
                ev[:, n] = np.where(in_fov[:, n], e, np.nan)
                if masks:
                    mk = masks[cam_ids[n]]
                    mx = np.clip((h2[:, 0] * mk.shape[1] / IMG_W).astype(int), 0, mk.shape[1] - 1)
                    my = np.clip((h2[:, 1] * mk.shape[0] / IMG_H).astype(int), 0, mk.shape[0] - 1)
                    static[:, n] = in_fov[:, n] & mk[my, mx]

            # --- BEV: GT map, per-person disk sums
            H, W = int(wld[3]), int(wld[4])
            r, a, b, w_min, h_min = (wld[i].item() for i in (0, 1, 2, 6, 7))
            cols = np.round((heads[:, 0] - w_min + a) * r).astype(int)
            rws = np.round((heads[:, 1] - h_min + b) * r).astype(int)
            ok = (rws >= 0) & (rws < H) & (cols >= 0) & (cols < W)
            pts = torch.zeros((H, W), device=device)
            for y, x in zip(rws[ok], cols[ok]):
                pts[y, x] += 1
            gt = F.conv2d(pts[None, None], kernel, padding=kernel.shape[-1] // 2)[0, 0][:H, :W]
            gt_disk = disk_sum(gt, rc)
            local = disk_sum(pts, int(round(2.0 * CELLS_PER_M)))
            preds = {}
            for name, m, _ in models:
                pred = predict_scene(m, feats, ctx, H, W, device)
                preds[name] = (pred, disk_sum(pred, rc))
            for name, pm, routing in proposed:
                pred = predict_proposed_scene(pm, routing, s, device)
                assert pred.shape == (H, W), f"{name}: map {tuple(pred.shape)} vs grid {(H, W)}"
                preds[name] = (pred, disk_sum(pred, rc))
            for name, bm in baselines:
                pred = predict_baseline_scene(bm, s, device)
                assert pred.shape == (H, W), f"{name}: map {tuple(pred.shape)} vs grid {(H, W)}"
                preds[name] = (pred, disk_sum(pred, rc))

            for i, p in enumerate(people):
                if not ok[i]:
                    continue
                y, x = rws[i], cols[i]
                fov = in_fov[i]
                by_people = fov & ~(vis[i] >= args.vis_thr)
                by_static = static[i]
                row = {
                    "scene": scene,
                    "snapshot": snap,
                    "idx": p["idx"],
                    "z_rel": float(heads[i, 2] - wld[5].item()),
                    "n_fov": int(fov.sum()),
                    "n_vis_geo": int(np.nansum(vis[i] >= args.vis_thr)),
                    "n_ev_2d": int(np.nansum(ev[i] >= args.ev_thr)),
                    "n_both": int(np.nansum((vis[i] >= args.vis_thr) & (ev[i] >= args.ev_thr))),
                    "vis_geo_mean": float(np.nanmean(vis[i])) if fov.any() else float("nan"),
                    "n_block_people": int(by_people.sum()),
                    "n_block_static": int(by_static.sum()),
                    "n_vis_all": int((fov & ~by_people & ~by_static).sum()),
                    "local_2m": float(local[y, x]),
                    "gt_disk": float(gt_disk[y, x]),
                }
                for name, (pred, pd) in preds.items():
                    row[f"pred_disk_{name}"] = float(pd[y, x])
                    row[f"recovery_{name}"] = float(pd[y, x] / max(float(gt_disk[y, x]), 1e-6))
                rows.append(row)

            if first:
                save_overlays(out_dir, scene, snap, s["images"], cam_ids, heads2d, in_fov, vis, ev, static, args)
                first = False
        if masks:
            print(f"{scene}: skipped {n_dark} dark frames of {len(frame_idx)}")

    with open(os.path.join(out_dir, "persons.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, [n for n, _, _ in models] + [n for n, _, _ in proposed] + [n for n, _ in baselines], args)
    summary["args"] = vars(args)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nsaved {out_dir}")


# ==========================================
# REPORT
# ==========================================


def summarize(rows, names, args):
    out = {"n_people": len(rows)}
    loc = np.array([r["local_2m"] for r in rows])
    t1, t2 = np.percentile(loc, [33.3, 66.7])
    strata = {
        "all": np.ones(len(rows), bool),
        f"sparse (<= {t1:.0f} within 2 m)": loc <= t1,
        "mid": (loc > t1) & (loc <= t2),
        f"dense (> {t2:.0f} within 2 m)": loc > t2,
    }
    print(f"\n{len(rows)} person-frames | local density (GT people within 2 m) tertiles {t1:.1f} / {t2:.1f}")
    nf = np.array([r["n_fov"] for r in rows])
    print("cameras with the head in view (n_fov): " + "  ".join(f"{k}: {(nf == k).sum()}" for k in range(6)))
    labels = [("geometric: head not covered by nearer people", "n_vis_geo")]
    if any(r["n_block_static"] for r in rows):
        labels.append(("not covered by people AND not behind a static occluder", "n_vis_all"))
    for label, key in labels:
        nv = np.array([r[key] for r in rows])
        print(f"\n=== cameras that see the head -- {label} ===")
        out[key] = {}
        for sname, smask in strata.items():
            print(f"[{sname}]")
            print(
                f"  {'n cams':<8}{'people':>8}"
                + "".join(f"{'recovery ' + n:>26}{'miss<0.5':>10}" for n in names)
                + f"{'local 2m':>10}{'n_fov':>7}"
            )
            out[key][sname] = {}
            for k in range(6):
                m = smask & (nv == k)
                if m.sum() < 10:
                    continue
                line = f"  {k:<8}{int(m.sum()):>8}"
                rec = {}
                for n in names:
                    rv = np.array([r[f"recovery_{n}"] for r in rows])[m]
                    rec[n] = {
                        "mean": float(rv.mean()),
                        "median": float(np.median(rv)),
                        "miss": float((rv < 0.5).mean()),
                    }
                    line += f"{rec[n]['mean']:>18.3f} ({rec[n]['median']:.2f}){rec[n]['miss']:>10.1%}"
                line += f"{loc[m].mean():>10.1f}{nf[m].mean():>7.1f}"
                print(line)
                out[key][sname][k] = {
                    "people": int(m.sum()),
                    "recovery": rec,
                    "local_2m": float(loc[m].mean()),
                    "n_fov": float(nf[m].mean()),
                }

    # --- blocked views at fixed coverage, and the regression
    bp = np.array([r["n_block_people"] for r in rows])
    bs = np.array([r["n_block_static"] for r in rows])
    snap = np.array([f"{r['scene']}/{r['snapshot']}" for r in rows])
    print(
        "\n=== recovery by blocked views, same number of cameras in view (recovery / miss rate"
        + (" per model: " + ", ".join(names) if len(names) > 1 else "")
        + ") ==="
    )
    out["blocked_tables"] = {}
    for what, b in (("blocked by people", bp), ("blocked by static occluders", bs)):
        print(f"[{what}]")
        for f in range(1, 6):
            line, cells = f"  {f} in view:", {}
            for k in range(0, f + 1):
                m = (nf == f) & (b == k)
                if m.sum() < 20:
                    continue
                parts = []
                for n in names:
                    rv = np.array([r[f"recovery_{n}"] for r in rows])[m]
                    parts.append(f"{rv.mean():.2f}/{(rv < 0.5).mean():.0%}")
                    cells.setdefault(n, {})[k] = {
                        "people": int(m.sum()),
                        "recovery": float(rv.mean()),
                        "miss": float((rv < 0.5).mean()),
                    }
                line += f"   {k} blocked (n={m.sum()}): " + " | ".join(parts)
            if cells:
                print(line)
                out["blocked_tables"].setdefault(what, {})[f] = cells

    X = np.column_stack([np.ones(len(rows)), nf, bp, bs, loc])
    cols = ["const", "n_in_view", "n_blocked_people", "n_blocked_static", "local_2m"]
    use_cols = [j for j in range(X.shape[1]) if j == 0 or X[:, j].std() > 0]
    us = np.unique(snap)
    rng = np.random.default_rng(0)
    groups = {u: np.where(snap == u)[0] for u in us}
    print(
        "\n=== regression: recovery ~ "
        + " + ".join(cols[j] for j in use_cols[1:])
        + f"  (95% interval: {args.bootstrap} resamples of whole frames, {len(us)} frames) ==="
    )
    out["regression"] = {}
    for n in names:
        y = np.array([r[f"recovery_{n}"] for r in rows])
        coef = np.linalg.lstsq(X[:, use_cols], y, rcond=None)[0]
        boot = []
        for _ in range(args.bootstrap):
            idx = np.concatenate([groups[u] for u in rng.choice(us, len(us))])
            boot.append(np.linalg.lstsq(X[idx][:, use_cols], y[idx], rcond=None)[0])
        lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
        res = {
            cols[j]: {"coef": float(c), "ci95": [float(a), float(b_)]} for j, c, a, b_ in zip(use_cols, coef, lo, hi)
        }
        out["regression"][n] = res
        print(
            f"  {n}: "
            + "  ".join(f"{k} {v['coef']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for k, v in res.items())
        )

    # --- across scenes: intervals from resampling whole scenes, and the coefficients of every scene on its own
    scn = np.array([r["scene"] for r in rows])
    scenes = list(dict.fromkeys(scn))
    if len(scenes) > 1:
        sgroups = {u: np.where(scn == u)[0] for u in scenes}
        # Within-scene effects: one intercept per scene (fixed effects). Pooling scenes without it mixes "which scene"
        # into the occlusion effect (scenes with many static-blocked views can have a higher overall recovery).
        slopes = [j for j in use_cols if j != 0]
        D = np.column_stack([(scn == u).astype(float) for u in scenes])
        XF = np.column_stack([D, X[:, slopes]])
        nd = len(scenes)

        def fe_fit(idx, y):
            dcols = [j for j in range(nd) if XF[idx, j].any()]
            c = np.linalg.lstsq(XF[idx][:, dcols + list(range(nd, XF.shape[1]))], y[idx], rcond=None)[0]
            return c[len(dcols) :]

        print(
            f"\n=== within-scene regression (one intercept per scene), 95% interval from {args.bootstrap} resamples "
            f"of whole scenes ({len(scenes)} scenes) -- the main result for several scenes ==="
        )
        out["regression_within_scene"] = {}
        for n in names + (["difference " + names[1] + " - " + names[0]] if len(names) > 1 else []):
            if n.startswith("difference "):
                y = np.array([r[f"recovery_{names[1]}"] - r[f"recovery_{names[0]}"] for r in rows])
            else:
                y = np.array([r[f"recovery_{n}"] for r in rows])
            coef = fe_fit(np.arange(len(y)), y)
            boot = [
                fe_fit(np.concatenate([sgroups[u] for u in rng.choice(scenes, len(scenes))]), y)
                for _ in range(args.bootstrap)
            ]
            lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
            res = {
                cols[j]: {"coef": float(c), "ci95": [float(a_), float(b_)]}
                for j, c, a_, b_ in zip(slopes, coef, lo, hi)
            }
            out["regression_within_scene"][n] = res
            print(
                f"  {n}: "
                + "  ".join(f"{k} {v['coef']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for k, v in res.items())
            )
        print(
            "\n=== per scene: coefficient per view blocked by static occluders / by people "
            "(same regression on that scene alone) ==="
        )
        print(f"  {'scene':<10}{'people':>8}{'static share':>14}" + "".join(f"{n[:22]:>26}" for n in names))
        out["per_scene"] = {}
        for u in scenes:
            m = sgroups[u]
            cols_u = [j for j in use_cols if j == 0 or X[m, j].std() > 0]
            line = f"  {u:<10}{len(m):>8}{(bs[m] > 0).mean():>14.0%}"
            out["per_scene"][u] = {"people": int(len(m)), "share_with_static_block": float((bs[m] > 0).mean())}
            for n in names:
                y = np.array([r[f"recovery_{n}"] for r in rows])[m]
                cf = dict(zip([cols[j] for j in cols_u], np.linalg.lstsq(X[m][:, cols_u], y, rcond=None)[0]))
                st, pe = cf.get("n_blocked_static", np.nan), cf.get("n_blocked_people", np.nan)
                out["per_scene"][u][n] = {
                    "blocked_static": float(st),
                    "blocked_people": float(pe),
                    "mean_recovery": float(y.mean()),
                }
                line += f"{st:>+13.3f} /{pe:>+7.3f}    "
            print(line)
    return out


def save_overlays(out_dir, scene, snap, images, cam_ids, heads2d, in_fov, vis, ev, static, args):
    """Camera images of one frame with every in-view head marked: green = visible, red = covered by nearer people,
    blue = behind a static occluder (mask), magenta = both."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    for n, c in enumerate(cam_ids):
        im = ((images[n] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()[:, :, ::-1] * 255).astype(np.uint8)
        im = cv2.resize(im, (IMG_W, IMG_H))
        for i in np.where(in_fov[:, n])[0]:
            x, y = np.round(heads2d[n][i]).astype(int)
            covered = not (vis[i, n] >= args.vis_thr)
            col = {
                (False, False): (0, 255, 0),
                (True, False): (0, 0, 255),
                (False, True): (255, 0, 0),
                (True, True): (255, 0, 255),
            }[(covered, bool(static[i, n]))]
            cv2.circle(im, (x, y), 7, col, -1)
        cv2.imwrite(os.path.join(out_dir, f"{scene}_snap{snap}_cam{c}.jpg"), im)


if __name__ == "__main__":
    main()
