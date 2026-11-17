"""
sweep_density_rule.py

Threshold sweep of the density rule (doc/head_ablation.md §6i, §8.10): the hybrid uses the dense expert where its OWN
predicted local density (5 x 5 m window) is at least `thr` persons/m^2 and the sparse expert (NMS peaks, kernel 3,
threshold 0.064) elsewhere. The rule combines the two experts' full-scene maps pixel by pixel, so each frame is
predicted once per expert and every threshold is assembled from those two maps: a sweep costs one evaluation.

Same frames, cameras and metrics as eval_map_quality.py (K-means cameras, --frames_per_scene seeded frames per held-out
scene, frame_metrics), so the 0.25 row reproduces `proposed__<name>_dz25p.csv`. thr 0 = the dense expert everywhere,
inf = the sparse expert everywhere.

Writes diagnostics/density_sweep/<name>.csv (one row per frame and threshold) and prints, per split and threshold:
MAE, bias, F1, MODA, GAME3, far > 2 m and the dense-zone share; the threshold with the lowest val MAE is marked.

Usage (from src):
    uv run python scripts/diagnosis/sweep_density_rule.py --ckpt hom_s42=multiscene_final_stage_2_abl_homography_256_dense_S4D2_s42_ema.pth
    uv run python scripts/diagnosis/sweep_density_rule.py --analyze_only --ckpt hom_s42=...
"""

import os
import sys
import csv
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

import neck_ablation as NA
import evaluate_paper_protocol as EP
import eval_map_quality as MQ
from utils.exclusions import load_excluded_scenes
from losses.gaussian_mse import get_map_kernel

OUT = os.path.join(src_dir, "diagnostics", "density_sweep")
THRESHOLDS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, float("inf"))
WIN = 25  # 25 px = 5 m at 5 px/m, as EP.density_zone


def frame_metrics_fast(pred, gts, wld, dist, args):
    """MQ.frame_metrics without the GT-reference metrics, with the distance-to-nearest-person map precomputed once per
    frame (it does not depend on the threshold). Same definitions and keys for every metric it returns."""
    r = wld[0, 0].item()
    tp, fp, fn, modp = EP.match(EP.detect(pred, args.cls_thres, args.nms_radius_px), gts, args.match_radius_m * r)
    p = (pred / EP.SCALE).float().cpu().numpy()
    out = {"gt": len(gts), "pred": float(p.sum()), "tp": tp, "fp": fp, "fn": fn, "modp_sum": modp}
    for L in MQ.GAME_LEVELS:
        out[f"game{L}"] = MQ.game(p, gts, L)
    pos = np.clip(p, 0, None)
    for d in MQ.FAR_M:
        out[f"far{d:g}m"] = float(pos[dist > d].sum() / max(pos.sum(), 1e-8))
    return out


def distance_to_people(gts, shape, r):
    dots = np.zeros(shape, bool)
    if len(gts):
        dots[gts[:, 0].astype(int), gts[:, 1].astype(int)] = True
    return distance_transform_edt(~dots) / r if len(gts) else np.full(shape, np.inf)


@torch.no_grad()
def sweep(name, weights, args, kernel, device, data):
    path = os.path.join(OUT, f"{name}.csv")
    if os.path.exists(path) and not args.overwrite:
        print(f"[skip] {name}: {path} exists")
        return
    model, _ = EP.load_model("proposed", os.path.join(src_dir, weights), device)
    model.T.apply_sparse_render(model, {"mode": "peaks", "kernel": 3, "threshold": 0.064})
    model.head.render_kernel = model.head.render_kernel.to(device)
    print(f"[load] {name}: {weights} (sparse render {model.head.sparse_render})")
    rows = []
    for split in ("validate", "test"):
        scenes = data[split]
        names, bounds = [n for n, _ in scenes], np.cumsum([0] + [len(ds) for _, ds in scenes])
        loader = NA.make_loader(
            scenes, args.frames_per_scene, args.seed + (0 if split == "validate" else 13), args.num_workers
        )
        for sample, idx in zip(tqdm(loader, desc=f"{name} {split}", mininterval=30), loader.sampler):
            k = int(np.searchsorted(bounds, idx, side="right") - 1)
            shape = (int(sample["wld_map_paras"][0, 3].item()), int(sample["wld_map_paras"][0, 4].item()))
            dense = EP.predict_proposed(model, sample, device, torch.ones(shape, dtype=torch.bool))
            sparse = EP.predict_proposed(model, sample, device, torch.zeros(shape, dtype=torch.bool))
            dens = (
                F.avg_pool2d((dense / EP.SCALE)[None, None], WIN, stride=1, padding=WIN // 2, count_include_pad=False)[
                    0, 0
                ]
                * WIN
                * WIN
                / (WIN / 5.0) ** 2
            )
            gts = EP.gt_pixels(sample["gt_world_xy"], sample["wld_map_paras"], shape)
            meta = {"split": split, "scene": names[k], "snapshot": scenes[k][1].snapshots[idx - bounds[k]]}
            dist = distance_to_people(gts, shape, sample["wld_map_paras"][0, 0].item())
            for thr in THRESHOLDS:
                zone = (dens >= thr).float()
                pred = zone * dense + (1 - zone) * sparse
                rows.append(
                    {
                        **meta,
                        "thr": thr,
                        **frame_metrics_fast(pred, gts, sample["wld_map_paras"], dist, args),
                        "dense_zone_frac": float(zone.mean()),
                    }
                )
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {path} ({len(rows)} rows)")
    del model
    torch.cuda.empty_cache()


def analyze(names, select="game3", write_routing=None):
    """Per model and threshold: pooled metrics per split; the threshold with the lowest val <select> (default GAME3, the
    count-scale-robust rule of doc/head_ablation.md §8.10; val MAE is confounded by the val/test count offset) is the
    selection. --write_routing writes the first model's selection as a routing token ('dzone<thr>p') to a file, for
    evaluate_paper_protocol.py --routing @<file>."""
    summary = {}
    for name in names:
        path = os.path.join(OUT, f"{name}.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        res = {}
        for split in ("validate", "test"):
            res[split] = {}
            for thr in THRESHOLDS:
                rs = [r for r in rows if r["split"] == split and float(r["thr"]) == thr]
                p = MQ.pooled(rs, "")
                p["bias"] = float(np.mean([float(r["pred"]) - float(r["gt"]) for r in rs]))
                p["dense_zone_frac"] = float(np.mean([float(r["dense_zone_frac"]) for r in rs]))
                res[split][str(thr)] = p
        best = min(THRESHOLDS, key=lambda t: res["validate"][str(t)][select])
        best_mae = min(THRESHOLDS, key=lambda t: res["validate"][str(t)]["mae"])
        summary[name] = {
            "results": res,
            "best_val_thr": best,
            "selected_by": f"val {select}",
            "best_val_mae_thr": best_mae,
        }
        print(f"\n== {name} (val-selected threshold by {select}: {best:g}; by val MAE: {best_mae:g})")
        print(
            f"{'thr':>5s} | {'val MAE':>7s} {'bias':>6s} {'F1':>5s} {'MODA':>6s} {'GAME3':>6s} {'far2m':>6s} {'dense':>6s} | "
            f"{'test MAE':>8s} {'bias':>6s} {'F1':>5s} {'MODA':>6s} {'GAME3':>6s} {'far2m':>6s} {'dense':>6s}"
        )
        for thr in THRESHOLDS:
            v, t = res["validate"][str(thr)], res["test"][str(thr)]

            def cell(p):
                return (
                    f"{p['mae']:7.1f} {p['bias']:+6.0f} {p['f1']:5.1f} {p['moda']:6.0f} {p['game3']:6.0f} "
                    f"{p['far2m']:5.1f}% {100 * p['dense_zone_frac']:5.1f}%"
                )

            print(f"{thr:5g} | {cell(v)} |  {cell(t)}{'   <- val best' if thr == best else ''}")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[saved] {os.path.join(OUT, 'summary.json')}")
    if write_routing:
        thr = summary[names[0]]["best_val_thr"]
        assert np.isfinite(thr), f"selected threshold {thr} is not a usable density rule"
        with open(write_routing, "w") as f:
            f.write(f"dzone{thr:g}p")
        print(f"[saved] {write_routing}: dzone{thr:g}p ({names[0]}, val {select})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=checkpoint (repeatable)")
    ap.add_argument("--frames_per_scene", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cls_thres", type=float, default=0.2)
    ap.add_argument("--nms_radius_px", type=float, default=2.5)
    ap.add_argument("--match_radius_m", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    ap.add_argument("--out_dir", default=None, help="output folder (default diagnostics/density_sweep)")
    ap.add_argument(
        "--select",
        default="game3",
        choices=["game3", "mae", "far2m"],
        help="val metric that selects the threshold (lowest wins)",
    )
    ap.add_argument(
        "--write_routing",
        default=None,
        help="write the first --ckpt's selected routing token (e.g. dzone0.15p) to this file",
    )
    args = ap.parse_args()
    global OUT
    if args.out_dir:
        OUT = os.path.abspath(args.out_dir)
    specs = [c.split("=", 1) for c in args.ckpt]
    if not args.analyze_only:
        device = torch.device("cuda")
        kernel = get_map_kernel(sigma=NA.MAP_SIGMA, kernel_size=NA.MAP_KERNEL_SIZE).to(device)
        kernel = (kernel / kernel.sum()).float().reshape(1, 1, *kernel.shape[-2:])
        with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
            cams_cfg = json.load(f)
        excluded = load_excluded_scenes()
        data = {s: NA.load_split(s, cams_cfg, excluded, "master_refs.pt", 4) for s in ("validate", "test")}
        for name, weights in specs:
            sweep(name, weights, args, kernel, device, data)
    analyze([n for n, _ in specs], args.select, args.write_routing)


if __name__ == "__main__":
    main()
