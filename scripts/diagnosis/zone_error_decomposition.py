"""
zone_error_decomposition.py

Where does the hybrid head's count gain over the baseline decoder come from (doc/head_ablation.md §8.15)? Per frame, the
hybrid's own density-rule zone (dense expert where its own predicted density >= --thr persons/m^2) splits the map into
a dense and a sparse zone, and the same zone is applied to every model, so their counts are compared on identical
areas:

  H0      the neck-ablation run's baseline decoder (EMA weights, precise BN, as eval_map_quality.py)
  D2      the hybrid's dense expert alone (in the dense zone the hybrid IS D2)
  hybrid  D2 in the dense zone, the sparse expert's NMS peaks in the sparse zone

Per zone: GT count (heads whose pixel lies in the zone), predicted count (map mass in the zone), signed error. Since the
two zones partition the map, total error = dense-zone error + sparse-zone error; their opposite signs can cancel.
Same frames as eval_map_quality.py / sweep_density_rule.py (K-means cameras, --frames_per_scene seeded frames per held-out
scene).

Writes diagnostics/zone_errors/<tag>/frames.csv and summary.json; prints per split and per scene.

Usage (from src):
    uv run python scripts/diagnosis/zone_error_decomposition.py --h0 runs/neck_ablation/ablation_v2/homography_256/last.pth \
        --hybrid s42=multiscene_final_stage_2_abl_homography_256_dense_S4D2_s42_ema.pth --thr 0.15
"""

import os
import sys
import csv
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

import neck_ablation as NA
import evaluate_paper_protocol as EP
from utils.exclusions import load_excluded_scenes

OUT = os.path.join(src_dir, "diagnostics", "zone_errors")


def zone_counts(pred, gts, zone):
    """Predicted mass (persons) and GT heads inside / outside the zone. pred [H, W] in SCALE units, zone bool [H, W]."""
    p = (pred / EP.SCALE).float()
    z = zone.to(p.device)
    out = {"pred_dense": float(p[z].sum()), "pred_sparse": float(p[~z].sum())}
    if len(gts):
        inz = zone.cpu().numpy()[gts[:, 0].astype(int), gts[:, 1].astype(int)]
        out["gt_dense"], out["gt_sparse"] = int(inz.sum()), int((~inz).sum())
    else:
        out["gt_dense"] = out["gt_sparse"] = 0
    return out


@torch.no_grad()
def run(args):
    device = torch.device("cuda")
    h0, _ = EP.load_model("ablation", os.path.join(src_dir, args.h0) if not os.path.isabs(args.h0) else args.h0, device)
    hybrids = []
    for spec in args.hybrid:
        name, weights = spec.split("=", 1)
        m, _ = EP.load_model("proposed", os.path.join(src_dir, weights), device)
        m.T.apply_sparse_render(m, {"mode": "peaks", "kernel": 3, "threshold": 0.064})
        m.head.render_kernel = m.head.render_kernel.to(device)
        hybrids.append((name, m))
    extras = []  # further dense experts, scored alone (dense routing) on the hybrids' zones
    for spec in args.extra_dense:
        name, weights = spec.split("=", 1)
        m, _ = EP.load_model("proposed", os.path.join(src_dir, weights), device)
        extras.append((name, m))
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    data = {s: NA.load_split(s, cams_cfg, load_excluded_scenes(), "master_refs.pt", 4) for s in ("validate", "test")}
    rows = []
    for split in ("validate", "test"):
        scenes = data[split]
        names, bounds = [n for n, _ in scenes], np.cumsum([0] + [len(ds) for _, ds in scenes])
        loader = NA.make_loader(
            scenes, args.frames_per_scene, args.seed + (0 if split == "validate" else 13), args.num_workers
        )
        for sample, idx in zip(tqdm(loader, desc=split, mininterval=30), loader.sampler):
            k = int(np.searchsorted(bounds, idx, side="right") - 1)
            shape = (int(sample["wld_map_paras"][0, 3].item()), int(sample["wld_map_paras"][0, 4].item()))
            gts = EP.gt_pixels(sample["gt_world_xy"], sample["wld_map_paras"], shape)
            p_h0 = EP.predict_ablation(h0, sample, device)
            p_extra = {
                n: EP.predict_proposed(m, sample, device, torch.ones(shape, dtype=torch.bool)) for n, m in extras
            }
            for name, m in hybrids:
                dense = EP.predict_proposed(m, sample, device, torch.ones(shape, dtype=torch.bool))
                sparse = EP.predict_proposed(m, sample, device, torch.zeros(shape, dtype=torch.bool))
                zone = EP.density_zone(m, sample, device, args.thr, shape)
                z = zone.to(device).float()
                hyb = z * dense + (1 - z) * sparse
                row = {
                    "split": split,
                    "scene": names[k],
                    "snapshot": scenes[k][1].snapshots[idx - bounds[k]],
                    "hybrid": name,
                    "gt": len(gts),
                    "dense_zone_frac": float(zone.float().mean()),
                }
                for model, pred in [("H0", p_h0), ("D2", dense), ("hybrid", hyb)] + list(p_extra.items()):
                    c = zone_counts(pred, gts, zone)
                    row["gt_dense"], row["gt_sparse"] = c["gt_dense"], c["gt_sparse"]
                    row[f"{model}_dense"], row[f"{model}_sparse"] = c["pred_dense"], c["pred_sparse"]
                rows.append(row)
    return rows


def analyze(rows, out_dir):
    summary = {}
    models = ["H0", "D2", "hybrid"] + sorted(
        {
            k[: -len("_dense")]
            for k in rows[0]
            if k.endswith("_dense") and k not in ("gt_dense", "H0_dense", "D2_dense", "hybrid_dense")
        }
    )
    for hyb in sorted({r["hybrid"] for r in rows}):
        summary[hyb] = {}
        for split in ("validate", "test"):
            rs = [r for r in rows if r["hybrid"] == hyb and r["split"] == split]

            def g(k):
                return np.array([float(r[k]) for r in rs])

            gd, gs = g("gt_dense"), g("gt_sparse")
            res = {
                "frames": len(rs),
                "gt_dense": gd.mean(),
                "gt_sparse": gs.mean(),
                "dense_zone_frac": g("dense_zone_frac").mean(),
            }
            print(
                f"\n== {hyb}, {split} ({len(rs)} frames; dense zone {100 * res['dense_zone_frac']:.0f}% of the map, "
                f"GT per frame {gd.mean():.0f} dense + {gs.mean():.0f} sparse = {(gd + gs).mean():.0f})"
            )
            print(
                f"  {'model':8s} | {'dense: pred':>11s} {'bias':>7s} {'MAE':>6s} | {'sparse: pred':>12s} {'bias':>7s} "
                f"{'MAE':>6s} | {'total: bias':>11s} {'MAE':>6s}"
            )
            for m in models:
                ed, es = g(f"{m}_dense") - gd, g(f"{m}_sparse") - gs
                et = ed + es
                res[m] = {
                    "dense_pred": g(f"{m}_dense").mean(),
                    "dense_bias": ed.mean(),
                    "dense_mae": np.abs(ed).mean(),
                    "sparse_pred": g(f"{m}_sparse").mean(),
                    "sparse_bias": es.mean(),
                    "sparse_mae": np.abs(es).mean(),
                    "total_bias": et.mean(),
                    "total_mae": np.abs(et).mean(),
                    "cancellation": float(np.mean(np.abs(ed) + np.abs(es) - np.abs(et))),
                }
                # dense-zone error split into the per-scene offset and the within-scene (frame-to-frame) part
                sc = np.array([r["scene"] for r in rs])
                off = {s: ed[sc == s].mean() for s in set(sc)}
                res[m]["dense_scene_offset"] = float(np.mean([abs(v) for v in off.values()]))
                res[m]["dense_within_scene_mae"] = float(np.mean(np.abs(ed - np.array([off[s] for s in sc]))))
                r = res[m]
                print(
                    f"  {m:8s} | {r['dense_pred']:11.1f} {r['dense_bias']:+7.1f} {r['dense_mae']:6.1f} | "
                    f"{r['sparse_pred']:12.1f} {r['sparse_bias']:+7.1f} {r['sparse_mae']:6.1f} | "
                    f"{r['total_bias']:+11.1f} {r['total_mae']:6.1f}   (cancellation {r['cancellation']:.1f}; dense zone: "
                    f"scene offset {r['dense_scene_offset']:.1f}, within-scene {r['dense_within_scene_mae']:.1f})"
                )
            # per scene: total MAE and the zone biases of H0 and the hybrid
            by = defaultdict(list)
            for r in rs:
                by[r["scene"]].append(r)
            print("  per scene (total MAE H0 / hybrid | dense-zone bias H0 / hybrid | sparse-zone bias H0 / hybrid):")
            res["per_scene"] = {}
            for sc, srs in sorted(by.items()):

                def gsc(k):
                    return np.array([float(r[k]) for r in srs])

                e = {m: (gsc(f"{m}_dense") - gsc("gt_dense"), gsc(f"{m}_sparse") - gsc("gt_sparse")) for m in models}
                ps = {
                    m: {
                        "total_mae": float(np.abs(e[m][0] + e[m][1]).mean()),
                        "dense_bias": float(e[m][0].mean()),
                        "sparse_bias": float(e[m][1].mean()),
                    }
                    for m in models
                }
                res["per_scene"][sc] = ps
                print(
                    f"    {sc:9s} {ps['H0']['total_mae']:6.1f} / {ps['hybrid']['total_mae']:6.1f} | "
                    f"{ps['H0']['dense_bias']:+7.1f} / {ps['hybrid']['dense_bias']:+7.1f} | "
                    f"{ps['H0']['sparse_bias']:+7.1f} / {ps['hybrid']['sparse_bias']:+7.1f}"
                )
            summary[hyb][split] = res
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"[saved] {os.path.join(out_dir, 'summary.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h0", required=True, help="<run>/<variant>/last.pth of the neck-ablation run (H0)")
    ap.add_argument(
        "--hybrid", action="append", required=True, help="name=checkpoint of a hybrid (D2 + S4D2), repeatable"
    )
    ap.add_argument(
        "--extra_dense",
        action="append",
        default=[],
        help="name=checkpoint of a further dense expert, scored alone on the hybrids' zones (repeatable)",
    )
    ap.add_argument("--thr", type=float, default=0.15, help="density-rule threshold (persons/m^2)")
    ap.add_argument("--frames_per_scene", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--tag", default="hom_dz15")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.join(OUT, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "frames.csv")
    if args.analyze_only:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    else:
        rows = run(args)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[saved] {path} ({len(rows)} rows)")
    analyze(rows, out_dir)


if __name__ == "__main__":
    main()
