"""
eval_ray_ghosts.py

Where does the attention necks' extra far mass come from? (doc/neck_ablation.md section 10: ca, ca_sa and the
8-anchor ca_sa all put ~10 points more of their density > 1 m from every person than homography_256; the
self-attention and the anchor set do not change it.) Two checks on the existing checkpoints, same frames and
evaluation as eval_map_quality.py:

0b  Ray ghosts. A BEV cell whose height-anchor point z in camera c projects onto a person's head pixel lies on the
    camera ray through that head, at height z. Per frame, per active camera and per person whose head is in that
    camera's image, the ground trace of that ray is drawn as a corridor (half-width --corridor_m) in bins of height
    relative to the head plane (dz). Of the mass > 1 m from every person ("far mass") we measure
      - the share on any corridor, per dz bin, against the share of far AREA there (chance level);
      - the share by support = number of cameras whose corridor covers the cell (a ghost from one camera has
        support 1; agreement between cameras would suppress it).
    The excess far mass of an attention model over the homography, split by dz bin, shows whether it sits at the
    model's anchor heights (depth ambiguity) or elsewhere.

0c  What the cross-attention learned, and test-time switches. In ca / ca_sa the cross-attention's offsets and
    weights come from the learned per-cell queries only, so they are the same for every image: dumped directly
    (offset length in image pixels, weight per anchor). Then the same frames are predicted with
      zero_off   sampling offsets set to 0 (pure reference points)
      head       all weight on the anchor nearest the head plane
      head_zero  both
    No retraining: the head was trained on the full mixture, so counts shift; far mass and GAME are what to read.

Writes diagnostics/ray_ghosts/<run>__<variant>__<mode>.csv (per frame), static_<run>__<variant>.json and
summary.json; prints the tables.

Usage (from src):
    uv run python evaluation/eval_ray_ghosts.py
    uv run python evaluation/eval_ray_ghosts.py --analyze_only
"""

import os
import sys
import csv
import json
import types
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

import neck_ablation as NA
import diagnose_hca as DH
import eval_per_frame as PF
import eval_map_quality as MQ
import evaluate_paper_protocol as EP
from utils.exclusions import load_excluded_scenes
from losses.gaussian_mse import get_map_kernel

OUT = os.path.join(src_dir, "diagnostics", "ray_ghosts")
IMG_W, IMG_H = 1920, 1080
DZ_EDGES = np.arange(-4.5, 4.01, 0.5)  # height along the ray relative to the head plane, metres
DZ_BINS = [f"{a:+.1f}..{b:+.1f}" for a, b in zip(DZ_EDGES[:-1], DZ_EDGES[1:])]
SUPPORT = ("0", "1", "2", "3+")
MODES = ("full", "zero_off", "head", "head_zero")
FIELDS = (
    ["split", "scene", "snapshot", "gt", "pred", "tp", "fp", "fn", "game3", "far1m", "far2m", "corridor_area"]
    + [f"ray_mass_{b}" for b in DZ_BINS]
    + [f"ray_area_{b}" for b in DZ_BINS]
    + [f"sup_mass_{s}" for s in SUPPORT]
    + [f"sup_area_{s}" for s in SUPPORT]
)


# ==========================================
# 0b: ray corridors
# ==========================================


def to_map(xy, wld):
    """World (x, y) -> continuous map (col, row), same convention as evaluate_paper_protocol.gt_pixels."""
    r, a, b, w_min, h_min = (wld[i] for i in (0, 1, 2, 6, 7))
    return np.stack([(xy[..., 0] - w_min + a) * r, (xy[..., 1] - h_min + b) * r], axis=-1)


def corridors(cam_paras, wld, gt_xy, shape, corridor_m):
    """cov[c, k]: cells within corridor_m of the ground trace of camera c's rays through every head it images, for
    heights in dz bin k. Heads are put at the frame's head plane d_mean."""
    H, W = shape
    r, d_mean = wld[0], wld[5]
    th = max(1, int(round(2 * corridor_m * r)))
    cov = np.zeros((len(cam_paras), len(DZ_BINS), H, W), np.uint8)
    if len(gt_xy) == 0:
        return cov.astype(bool)
    heads = np.concatenate([gt_xy, np.full((len(gt_xy), 1), d_mean)], axis=1)
    for c, cp in enumerate(cam_paras):
        fx, fy, u, v = cp[:4]
        K = np.array([[fx * IMG_W, 0, u * IMG_W], [0, fy * IMG_H, v * IMG_H], [0, 0, 1]], np.float64)
        rvec, tvec = cp[9:12].astype(np.float64), cp[12:15].astype(np.float64)
        R = cv2.Rodrigues(rvec)[0]
        C = -R.T @ tvec
        depth = (R @ heads.T).T[:, 2] + tvec[2]
        uv = cv2.projectPoints(heads, rvec, tvec, K, cp[4:9].astype(np.float64))[0].reshape(-1, 2)
        seen = (depth > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)
        hs = heads[seen]
        if len(hs) == 0 or C[2] <= d_mean + 0.5:
            continue
        for k, (z0, z1) in enumerate(zip(DZ_EDGES[:-1], DZ_EDGES[1:])):
            z1 = min(z1, C[2] - d_mean - 0.5)  # the ray never reaches heights above the camera
            if z1 <= z0:
                continue
            t0 = (d_mean + z0 - C[2]) / (d_mean - C[2])  # P(t) = C + t (head - C); t = 1 at the head
            t1 = (d_mean + z1 - C[2]) / (d_mean - C[2])
            p0 = to_map(C[:2] + t0 * (hs[:, :2] - C[:2]), wld)
            p1 = to_map(C[:2] + t1 * (hs[:, :2] - C[:2]), wld)
            segs = np.stack([p0, p1], axis=1)
            ok = np.isfinite(segs).all((1, 2)) & (np.abs(segs).max((1, 2)) < 1e5)
            cv2.polylines(cov[c, k], np.round(segs[ok]).astype(np.int32), False, 1, thickness=th)
    return cov.astype(bool)


def ray_metrics(p, gts, wld, cam_paras, gt_xy, corridor_m):
    """p: predicted map in people [H, W] (numpy). Shares of the TOTAL predicted mass that is far (> 1 m) and on a
    corridor of dz bin k / with support s; and the matching shares of the far AREA (chance level)."""
    r = wld[0]
    dots = np.zeros(p.shape, bool)
    if len(gts):
        dots[gts[:, 0].astype(int), gts[:, 1].astype(int)] = True
    far = (distance_transform_edt(~dots) / r > 1.0) if len(gts) else np.ones(p.shape, bool)
    pos = np.clip(p, 0, None)
    tot, far_area = max(pos.sum(), 1e-8), max(far.sum(), 1)
    cov = corridors(cam_paras, wld, gt_xy, p.shape, corridor_m)
    out = {}
    anyc = cov.any(0)  # [K, H, W]
    for k, b in enumerate(DZ_BINS):
        m = far & anyc[k]
        out[f"ray_mass_{b}"] = float(pos[m].sum() / tot)
        out[f"ray_area_{b}"] = float(m.sum() / far_area)
    sup = cov.any(1).sum(0)  # cameras whose corridor covers the cell
    for s, m in zip(SUPPORT, (sup == 0, sup == 1, sup == 2, sup >= 3)):
        out[f"sup_mass_{s}"] = float(pos[far & m].sum() / tot)
        out[f"sup_area_{s}"] = float((far & m).sum() / far_area)
    out["corridor_area"] = float((far & (sup > 0)).sum() / far_area)
    return out


# ==========================================
# 0c: static dump and test-time switches of the cross-attention
# ==========================================


def anchor_heights(a):
    """Anchor heights relative to the head plane for the cache the model was trained with."""
    if a.refs_file == "master_refs.pt":
        return torch.linspace(-1.75, 1.8 + 1.75, a.z_anchors).numpy()  # utils/reference_points.py (legacy)
    return torch.linspace(-1.75 - 2.5, 2.5, a.z_anchors).numpy()  # cache_reference.anchor_offsets defaults


@torch.no_grad()
def static_dump(model, a):
    """Offsets / weights of the cross-attention from the learned queries (image-independent in ca / ca_sa)."""
    ca = model.fusion.cross_attention
    q = model.fusion.bev_queries.float()
    N, K = ca.num_cameras, ca.num_points
    off = (ca.offset_predictor(q).view(-1, N, K, 2) * 0.1).cpu()
    off_px = torch.stack([off[..., 0] * IMG_W / 2, off[..., 1] * IMG_H / 2], -1).norm(dim=-1)  # [L, N, K]
    w = F.softmax(ca.weight_predictor(q).view(-1, N, K), dim=-1).cpu()
    ent = -(w * w.clamp_min(1e-12).log()).sum(-1) / np.log(K)
    z = anchor_heights(a)
    return {
        "anchor_dz_m": [round(float(v), 3) for v in z],
        "weight_mean_per_anchor": w.mean((0, 1)).tolist(),
        "weight_std_over_cells_per_anchor": w.std(0).mean(0).tolist(),
        "share_cells_argmax_per_anchor": torch.bincount(w.argmax(-1).flatten(), minlength=K)
        .div(w[..., 0].numel())
        .tolist(),
        "weight_entropy_norm_mean": float(ent.mean()),
        "offset_px_median_per_anchor": off_px.median(0).values.median(0).values.tolist(),
        "offset_px_p90_per_anchor": off_px.flatten(0, 1).quantile(0.9, dim=0).tolist(),
        "offset_px_weighted_mean": float((off_px * w).sum(-1).mean()),
        "offset_px_note": "in 1920x1080 image pixels; bilinear sampling on the stride-8 feature map",
    }


def patched_ca_forward(self, bev_queries, features_2d, reference_points):
    """DeformableCrossAttention.forward with self.mode in MODES (see module docstring)."""
    B, N, C, H_img, W_img = features_2d.shape
    L = bev_queries.shape[1]
    offsets = self.offset_predictor(bev_queries).view(B, L, N, self.num_points, 2) * 0.1
    weights = F.softmax(self.weight_predictor(bev_queries).view(B, L, N, self.num_points), dim=-1)
    if self.mode in ("zero_off", "head_zero"):
        offsets = torch.zeros_like(offsets)
    if self.mode in ("head", "head_zero"):
        weights = torch.zeros_like(weights)
        weights[..., self.head_idx] = 1.0
    ref = reference_points.reshape(B, N, L, self.num_points, 2).transpose(1, 2)
    loc = ref + offsets
    fused = torch.zeros_like(bev_queries)
    for c in range(N):
        s = F.grid_sample(features_2d[:, c], loc[:, :, c], mode="bilinear", padding_mode="zeros", align_corners=False)
        fused = fused + (s.permute(0, 2, 3, 1) * weights[:, :, c].unsqueeze(-1)).sum(dim=2)
    return self.output_proj(fused)


# ==========================================
# Run
# ==========================================


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {path} ({len(rows)} frames)")


@torch.no_grad()
def run(args, kernel, device):
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    excluded = load_excluded_scenes()
    splits = {}
    for spec in [s for s in args.models.split(",") if s]:
        run_name, variant = spec.split(":")
        modes = MODES if variant in ("ca", "ca_sa") else ("full",)
        paths = {m: os.path.join(OUT, f"{run_name}__{variant}__{m}.csv") for m in modes}
        if all(os.path.exists(p) for p in paths.values()) and not args.overwrite:
            print(f"[skip] {spec}: all modes exist")
            continue
        model, a = DH.load_model(
            os.path.join(src_dir, "runs", "neck_ablation", run_name),
            variant,
            {"z_anchors": 4, "refs_file": "master_refs.pt"},
            device,
        )
        if len(modes) > 1:
            ca = model.fusion.cross_attention
            ca.head_idx = int(np.abs(anchor_heights(a)).argmin())
            ca.forward = types.MethodType(patched_ca_forward, ca)
            ca.mode = "full"
            st = {**static_dump(model, a), "head_anchor_index": ca.head_idx}
            with open(os.path.join(OUT, f"static_{run_name}__{variant}.json"), "w") as f:
                json.dump(st, f, indent=1)
            print(f"[static] {spec}: {json.dumps(st)}")
        key = (a.refs_file, a.z_anchors)
        if key not in splits:
            splits[key] = {s: NA.load_split(s, cams_cfg, excluded, *key) for s in ("train", "validate", "test")}
        data = splits[key]
        NA.precise_bn(model, NA.make_loader(data["train"], 1, a.val_seed + 7, args.num_workers), device)
        rows = {m: [] for m in modes}
        for split in ("validate", "test"):
            scenes = data[split]
            names, bounds = [n for n, _ in scenes], np.cumsum([0] + [len(ds) for _, ds in scenes])
            loader = NA.make_loader(
                scenes, args.frames_per_scene, args.seed + (0 if split == "validate" else 13), args.num_workers
            )
            for sample, idx in zip(tqdm(loader, desc=f"{spec} {split}", mininterval=30), loader.sampler):
                k = int(np.searchsorted(bounds, idx, side="right") - 1)
                wld = sample["wld_map_paras"][0].numpy().astype(np.float64)
                cam_paras = sample["camera_paras"][0].numpy()
                gt_xy = sample["gt_world_xy"].reshape(-1, 2).numpy().astype(np.float64)
                base = {"split": split, "scene": names[k], "snapshot": scenes[k][1].snapshots[idx - bounds[k]]}
                for m in modes:
                    if len(modes) > 1:
                        model.fusion.cross_attention.mode = m
                    pred = PF.whole_map(model, sample, device)
                    gts = EP.gt_pixels(sample["gt_world_xy"], sample["wld_map_paras"], tuple(pred.shape))
                    mq = MQ.frame_metrics(pred, gts, sample["wld_map_paras"], kernel, args)
                    p = (pred / EP.SCALE).float().cpu().numpy()
                    rows[m].append({**base, **mq, **ray_metrics(p, gts, wld, cam_paras, gt_xy, args.corridor_m)})
        for m in modes:
            write(paths[m], rows[m])
        del model
        torch.cuda.empty_cache()


# ==========================================
# Analysis
# ==========================================


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def mean(rs, k):
    return float(np.mean([float(r[k]) for r in rs]))


def summarize(rs):
    tp, fp = sum(float(r["tp"]) for r in rs), sum(float(r["fp"]) for r in rs)
    gt = sum(float(r["gt"]) for r in rs)
    prec, rec = tp / max(tp + fp, 1), tp / max(gt, 1)
    return {
        "mae": float(np.mean([abs(float(r["pred"]) - float(r["gt"])) for r in rs])),
        "bias": float(np.mean([float(r["pred"]) - float(r["gt"]) for r in rs])),
        "f1": 100 * 2 * prec * rec / max(prec + rec, 1e-8),
        "game3": mean(rs, "game3"),
        "far1m": 100 * mean(rs, "far1m"),
        "corridor_area": 100 * mean(rs, "corridor_area"),
        **{f"ray_mass_{b}": 100 * mean(rs, f"ray_mass_{b}") for b in DZ_BINS},
        **{f"ray_area_{b}": 100 * mean(rs, f"ray_area_{b}") for b in DZ_BINS},
        **{f"sup_mass_{s}": 100 * mean(rs, f"sup_mass_{s}") for s in SUPPORT},
        **{f"sup_area_{s}": 100 * mean(rs, f"sup_area_{s}") for s in SUPPORT},
    }


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile(d[rng.integers(0, len(d), (n, len(d)))].mean(1), [2.5, 97.5])


def analyze(args):
    files = sorted(f for f in os.listdir(OUT) if f.endswith(".csv"))
    data = {f[:-4]: load_rows(os.path.join(OUT, f)) for f in files}
    ref = "ablation_v2__homography_256__full"
    summary = {}
    for split in ("validate", "test"):
        S = {n: summarize([r for r in rs if r["split"] == split]) for n, rs in data.items()}
        summary[split] = S
        print(f"\n===== {split} =====")
        print(
            f"{'model / mode':<42}{'MAE':>7}{'bias':>7}{'F1':>6}{'GAME3':>7}{'far>1m':>8}"
            + "".join(f"{'sup' + s:>7}" for s in SUPPORT)
            + f"{'corr.area':>10}"
        )
        for n, s in S.items():
            print(
                f"{n:<42}{s['mae']:7.1f}{s['bias']:7.1f}{s['f1']:6.1f}{s['game3']:7.1f}{s['far1m']:7.1f}%"
                + "".join(f"{s[f'sup_mass_{x}']:6.1f}%" for x in SUPPORT)
                + f"{s['corridor_area']:9.1f}%"
            )
        a = S[next(iter(S))]
        print(
            f"{'  far AREA share by support (chance)':<42}{'':>35}"
            + "".join(f"{a[f'sup_area_{x}']:6.1f}%" for x in SUPPORT)
        )
        # how the far mass of every model / mode is spread over the dz bins, against the far area there (chance)
        print("\nshare of the FAR mass on ray corridors per dz bin, and enrichment = that share / share of far area")
        cols = list(S)
        print(f"{'dz bin':<12}{'area':>7}" + "".join(f"{n.replace('ablation_', '')[:24]:>26}" for n in cols))
        for b in DZ_BINS:
            cells = []
            for n in cols:
                sh = 100 * S[n][f"ray_mass_{b}"] / max(S[n]["far1m"], 1e-8)
                cells.append(f"{sh:6.1f}% x{sh / max(a[f'ray_area_{b}'], 1e-8):4.2f}")
            print(f"{b:<12}{a[f'ray_area_{b}']:6.1f}%" + "".join(f"{c:>26}" for c in cells))
        print(f"{'support':<12}{'':>7}")
        for x in SUPPORT:
            cells = []
            for n in cols:
                sh = 100 * S[n][f"sup_mass_{x}"] / max(S[n]["far1m"], 1e-8)
                cells.append(f"{sh:6.1f}% x{sh / max(a[f'sup_area_{x}'], 1e-8):4.2f}")
            print(f"{'  ' + x + ' cams':<12}{a[f'sup_area_{x}']:6.1f}%" + "".join(f"{c:>26}" for c in cells))
        # paired per scene vs homography for the full models: far mass on support-1 cells and off any corridor
        if ref in data:
            print("\npaired per scene vs homography_256 (points of total mass, 95% CI over scenes)")

            def per(n, k):
                return {
                    sc: np.mean([float(r[k]) for r in data[n] if r["split"] == split and r["scene"] == sc])
                    for sc in sorted({r["scene"] for r in data[n] if r["split"] == split})
                }

            for n in data:
                if n == ref:
                    continue
                for k in ("far1m", "sup_mass_0", "sup_mass_1", "sup_mass_2", "sup_mass_3+"):
                    x, y = per(n, k), per(ref, k)
                    d = 100 * np.array([x[sc] - y[sc] for sc in x])
                    lo, hi = boot_ci(d)
                    print(
                        f"  {n:<40} {k:<12} {d.mean():+6.2f} [{lo:+.2f}, {hi:+.2f}]  higher in {(d > 0).sum()}/{len(d)}"
                    )
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "dz_bins": DZ_BINS, "splits": summary}, f, indent=1)
    print(f"[saved] {os.path.join(OUT, 'summary.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ablation_v2:homography_256,ablation_v2:ca_sa")
    ap.add_argument(
        "--frames_per_scene", type=int, default=50, help="seeded frames per held-out scene (as map quality)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corridor_m", type=float, default=0.3, help="half-width of a ray corridor")
    ap.add_argument("--cls_thres", type=float, default=0.2)
    ap.add_argument("--nms_radius_px", type=float, default=2.5)
    ap.add_argument("--match_radius_m", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", default="ray_ghosts", help="folder under diagnostics/")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()
    global OUT
    OUT = os.path.join(src_dir, "diagnostics", args.out)
    os.makedirs(OUT, exist_ok=True)
    if not args.analyze_only:
        device = torch.device("cuda")
        kernel = get_map_kernel(sigma=NA.MAP_SIGMA, kernel_size=NA.MAP_KERNEL_SIZE).to(device)
        kernel = (kernel / kernel.sum()).float().reshape(1, 1, *kernel.shape[-2:])
        run(args, kernel, device)
    analyze(args)


if __name__ == "__main__":
    main()
