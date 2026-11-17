"""
eval_train_bias.py

Is the val under-count a generalisation gap or a property of the loss / target? Evaluates final neck-ablation weights
(EMA, precise BN as in the run's final evaluation) with the ablation's whole-scene protocol on TRAINING scenes, next
to the run's own val / test numbers. Unbiased on train but -70 on val -> generalisation gap; biased on train too ->
loss / target. Also reports the bias per scene against the scene's mean GT count (does the bias grow with crowding?).

Usage (from src):
    uv run python evaluation/eval_train_bias.py --models ablation_v2:homography_256,ablation_mv:mvha_res
Writes diagnostics/train_bias/<tag>.json.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

import neck_ablation as NA
import diagnose_hca as DH
from utils.exclusions import load_excluded_scenes


@torch.no_grad()
def per_frame(model, scenes, loader, device):
    """(scene, signed error, gt) per frame, same tiling as neck_ablation.evaluate."""
    names = [n for n, _ in scenes]
    bounds = np.cumsum([0] + [len(ds) for _, ds in scenes])
    out = []
    model.eval()
    for sample, idx in zip(loader, loader.sampler):
        scene = names[int(np.searchsorted(bounds, idx, side="right") - 1)]
        wld = sample["wld_map_paras"].to(device)
        ctx = {
            "ref_points_full": sample["ref_points_full"].to(device),
            "camera_paras": sample["camera_paras"].to(device),
            "wld_map_paras": wld,
        }
        H, W = int(wld[0, 3].item()), int(wld[0, 4].item())
        pred = torch.zeros((H, W), device=device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            feats = model.features(sample["images"].to(device))
            for h0 in NA.tile_starts(H, NA.TILE[0]):
                for w0 in NA.tile_starts(W, NA.TILE[1]):
                    hh, ww = min(NA.TILE[0], H - h0), min(NA.TILE[1], W - w0)
                    pred[h0 : h0 + hh, w0 : w0 + ww] = model.predict_tile(feats, ctx, h0, w0)[0, 0, :hh, :ww].float()
        gt = NA.gt_count_in_map(sample["gt_world_xy"][0], sample["wld_map_paras"][0], (H, W))
        out.append((scene, pred.sum().item() / NA.SCALE - gt, gt))
    return out


def summary(rows):
    e = np.array([r[1] for r in rows])
    g = np.array([r[2] for r in rows])
    by_scene = {}
    for s in dict.fromkeys(r[0] for r in rows):
        es = np.array([r[1] for r in rows if r[0] == s])
        gs = np.array([r[2] for r in rows if r[0] == s])
        by_scene[s] = {"gt": float(gs.mean()), "bias": float(es.mean()), "mae": float(np.abs(es).mean())}
    sg = np.array([v["gt"] for v in by_scene.values()])
    sb = np.array([v["bias"] for v in by_scene.values()])
    return {
        "frames": len(rows),
        "mae": float(np.abs(e).mean()),
        "bias": float(e.mean()),
        "mean_gt": float(g.mean()),
        "rel_bias": float(e.sum() / max(g.sum(), 1)),
        "corr_scene_gt_bias": float(np.corrcoef(sg, sb)[0, 1]) if len(sg) > 2 else float("nan"),
        "per_scene": by_scene,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ablation_v2:homography_256,ablation_mv:mvha_res,ablation_mv:mvha")
    ap.add_argument("--frames_per_scene", type=int, default=5)
    ap.add_argument("--refs_file", default="master_refs_k8.pt", help="loaded for every model; homography ignores it")
    ap.add_argument("--z_anchors", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4321, help="frame selection on the train scenes")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--tag", default="train_bias")
    args = ap.parse_args()
    device = torch.device("cuda")
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    train = NA.load_split("train", cams_cfg, load_excluded_scenes(), args.refs_file, args.z_anchors)

    results = {"args": vars(args)}
    for spec in args.models.split(","):
        run, variant = spec.split(":")
        run_dir = os.path.join(src_dir, "runs", "neck_ablation", run)
        model, a = DH.load_model(run_dir, variant, {}, device)
        NA.precise_bn(model, NA.make_loader(train, 1, a.val_seed + 7, args.num_workers), device)
        rows = per_frame(
            model, train, NA.make_loader(train, args.frames_per_scene, args.seed, args.num_workers), device
        )
        res = json.load(open(os.path.join(run_dir, variant, "result.json")))
        r = {
            "train": summary(rows),
            "val": {k: res["final_val"][k] for k in ("mae", "bias")},
            "test": {k: res["test_final"][k] for k in ("mae", "bias")},
        }
        results[spec] = r
        t = r["train"]
        print(
            f"[{spec}] TRAIN MAE {t['mae']:.1f} bias {t['bias']:+.1f} ({100 * t['rel_bias']:+.1f}% of {t['mean_gt']:.0f} "
            f"people/frame; corr(scene GT, scene bias) {t['corr_scene_gt_bias']:+.2f}) | run's VAL MAE "
            f"{r['val']['mae']:.1f} bias {r['val']['bias']:+.1f} | TEST MAE {r['test']['mae']:.1f} bias {r['test']['bias']:+.1f}"
        )
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.join(src_dir, "diagnostics", "train_bias"), exist_ok=True)
    path = os.path.join(src_dir, "diagnostics", "train_bias", f"{args.tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
