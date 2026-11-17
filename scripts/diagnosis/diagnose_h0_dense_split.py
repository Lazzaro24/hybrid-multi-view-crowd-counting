"""
diagnose_h0_dense_split.py -- baseline decoder (H0) in the sparse areas, the proposed dense expert in the dense areas?
(doc/head_ablation.md §6j). No training.

On exactly the map-quality frames (eval_map_quality.py: K-means cameras, 50 seeded frames per held-out scene), both
models predict the whole scene:
  H0     the neck-ablation run with the baseline decoder (runs/neck_ablation/<run>/<variant>, EMA, precise BN as in
         eval_map_quality.py)
  dense  the proposed dense expert alone (a Stage 2/3 checkpoint, zone = everywhere)
The split zone comes from the dense expert's OWN predicted local density (5 x 5 m window) >= --thr persons/m^2 (the
GT-free 'dzone' rule of eval_map_quality.py). Per frame and zone it records each model's count and the GT count, and
scores whole-frame counts for:
  H0 alone | dense alone | split (H0 outside the zone, dense inside) | reverse split | average of both maps
The split only shows specialisation if it beats the average (an ensemble effect) and the per-zone errors show H0
better outside and dense better inside.

Usage (src/):
    uv run python -u scripts/diagnosis/diagnose_h0_dense_split.py \
        --h0 ablation_v2:homography_256 --dense multiscene_best_stage_2_abl_homography_256_unmasked_joint_genloss_ep30.pth
"""

import os
import sys
import csv
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

import neck_ablation as NA
import diagnose_hca as DH
import eval_per_frame as PF
import evaluate_paper_protocol as EP
from utils.exclusions import load_excluded_scenes

COMBOS = ("H0", "dense", "split", "reverse", "average")


def bootstrap_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h0", default="ablation_v2:homography_256", help="run:variant of the baseline-head model")
    ap.add_argument("--dense", default="multiscene_best_stage_2_abl_homography_256_unmasked_joint_genloss_ep30.pth")
    ap.add_argument("--thr", type=float, default=0.25, help="dense zone: predicted density >= thr persons/m^2")
    ap.add_argument("--frames_per_scene", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(src_dir, "diagnostics", "map_quality", "h0_dense_split"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    excluded = load_excluded_scenes()
    run, variant = args.h0.split(":")
    h0, a = DH.load_model(
        os.path.join(src_dir, "runs", "neck_ablation", run),
        variant,
        {"z_anchors": 4, "refs_file": "master_refs.pt"},
        device,
    )
    data = {s: NA.load_split(s, cams_cfg, excluded, a.refs_file, a.z_anchors) for s in ("train", "validate", "test")}
    NA.precise_bn(h0, NA.make_loader(data["train"], 1, a.val_seed + 7, args.num_workers), device)
    dense, _ = EP.load_model("proposed", os.path.join(src_dir, args.dense), device)
    print(f"[load] H0 = {args.h0} | dense = {args.dense} | zone: dense-expert density >= {args.thr} p/m^2")

    win = 25  # px = 5 m at 5 px/m
    rows = []
    for split in ("validate", "test"):
        scenes = data[split]
        names, bounds = [n for n, _ in scenes], np.cumsum([0] + [len(ds) for _, ds in scenes])
        loader = NA.make_loader(
            scenes, args.frames_per_scene, args.seed + (0 if split == "validate" else 13), args.num_workers
        )
        for sample, idx in zip(tqdm(loader, desc=split, mininterval=30), loader.sampler):
            k = int(np.searchsorted(bounds, idx, side="right") - 1)
            p_h0 = PF.whole_map(h0, sample, device) / EP.SCALE  # persons per px
            shape = tuple(p_h0.shape)
            p_d = EP.predict_proposed(dense, sample, device, torch.ones(shape, dtype=torch.bool)) / EP.SCALE
            dens = F.avg_pool2d(p_d[None, None], win, stride=1, padding=win // 2, count_include_pad=False)
            zone = dens[0, 0] * win * win / (win / 5.0) ** 2 >= args.thr
            gts = EP.gt_pixels(sample["gt_world_xy"], sample["wld_map_paras"], shape)
            z_np = zone.cpu().numpy()
            gt_in = int(z_np[gts[:, 0].astype(int), gts[:, 1].astype(int)].sum()) if len(gts) else 0
            zf = zone.float()
            row = {
                "split": split,
                "scene": names[k],
                "snapshot": scenes[k][1].snapshots[idx - bounds[k]],
                "gt": len(gts),
                "gt_in": gt_in,
                "gt_out": len(gts) - gt_in,
                "zone_frac": float(zf.mean()),
                "h0_in": float((p_h0 * zf).sum()),
                "h0_out": float((p_h0 * (1 - zf)).sum()),
                "d_in": float((p_d * zf).sum()),
                "d_out": float((p_d * (1 - zf)).sum()),
            }
            rows.append(row)

    with open(os.path.join(args.out, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def count(r, combo):
        return {
            "H0": r["h0_in"] + r["h0_out"],
            "dense": r["d_in"] + r["d_out"],
            "split": r["d_in"] + r["h0_out"],
            "reverse": r["h0_in"] + r["d_out"],
            "average": 0.5 * (r["h0_in"] + r["h0_out"] + r["d_in"] + r["d_out"]),
        }[combo]

    summary = {}
    for split in ("validate", "test"):
        rs = [r for r in rows if r["split"] == split]
        scenes = sorted({r["scene"] for r in rs})
        print(
            f"\n== {split}: {len(rs)} frames, dense zone {100 * np.mean([r['zone_frac'] for r in rs]):.1f}% of the map, "
            f"{100 * sum(r['gt_in'] for r in rs) / max(sum(r['gt'] for r in rs), 1):.1f}% of the people"
        )
        res = {"combos": {}, "zones": {}, "paired": {}}
        per_scene = defaultdict(dict)
        print(f"   {'combination':<10}{'MAE':>8}{'bias':>8}")
        for c in COMBOS:
            err = np.array([count(r, c) - r["gt"] for r in rs])
            res["combos"][c] = dict(mae=float(np.abs(err).mean()), bias=float(err.mean()))
            for s in scenes:
                per_scene[s][c] = float(np.mean([abs(count(r, c) - r["gt"]) for r in rs if r["scene"] == s]))
            print(f"   {c:<10}{np.abs(err).mean():>8.1f}{err.mean():>+8.1f}")
        print("   per zone, mean |error| / bias per frame:  outside zone (sparse areas) | inside zone (dense areas)")
        for m, (ki, ko) in (("H0", ("h0_in", "h0_out")), ("dense", ("d_in", "d_out"))):
            eo = np.array([r[ko] - r["gt_out"] for r in rs])
            ei = np.array([r[ki] - r["gt_in"] for r in rs])
            res["zones"][m] = dict(
                out_mae=float(np.abs(eo).mean()),
                out_bias=float(eo.mean()),
                in_mae=float(np.abs(ei).mean()),
                in_bias=float(ei.mean()),
            )
            print(
                f"   {m:<10} outside {np.abs(eo).mean():7.1f} / {eo.mean():+7.1f} | inside {np.abs(ei).mean():7.1f} / {ei.mean():+7.1f}"
            )
        print("   paired per scene (MAE difference, 95% bootstrap interval, scenes where the first is better):")
        for a_, b_ in (
            ("split", "H0"),
            ("split", "dense"),
            ("split", "average"),
            ("average", "H0"),
            ("average", "dense"),
            ("H0", "dense"),
        ):
            d = np.array([per_scene[s][a_] - per_scene[s][b_] for s in scenes])
            lo, hi = bootstrap_ci(d)
            res["paired"][f"{a_} - {b_}"] = dict(mean=float(d.mean()), ci=[lo, hi], better=int((d < 0).sum()), n=len(d))
            print(f"   {a_:>8} - {b_:<8} {d.mean():+7.1f} [{lo:+.1f}, {hi:+.1f}]  {int((d < 0).sum())}/{len(d)}")
        res["per_scene"] = per_scene
        summary[split] = res
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "results": summary}, f, indent=1)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
