"""
diagnose_mvha.py

Diagnostics for the multi-view height attention necks of neck_ablation.py (mvha / mvha_res), final EMA weights and
the same precise-BN statistics the ablation's "final" evaluation used. Answers: did the weights become image-dependent
(unlike hca, doc/neck_ablation.md section 7), where do they put the height, and what does the correction do?

  1. count       whole-scene count with and without the attention output (mvha_res: delta = 0, same head)
  2. correction  (mvha_res) per-cell |delta| / |homography|, cosine and least-squares coefficient of delta on the
                 homography features, near people (<= --people_radius m from a GT head) vs empty cells
  3. height      mean height marginal per anchor (summed over cameras and averaged over heads), people vs empty
  4. cameras     distance of the camera marginal from "every camera that sees the cell counts equally" (BEVFormer's
                 hit-count average): total variation 0.5 * sum |w_n - 1/n_seen|, 0 = plain average
  5. image dependence  at a fixed cell of a fixed scene, the std of the height / camera marginals across frames
                 (cameras in a fixed order for this; the training samples them in random order); hca's anchor weights
                 changed by 0.025 (std, uniform weight 0.125)

Usage (from src):
    uv run python scripts/diagnosis/diagnose_mvha.py --run_name ablation_mv --variant mvha_res
Writes diagnostics/mvha/<run_name>_<variant>.json.
"""

import os
import sys
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
import diagnose_hca as DH
import datasets.SynMVCrowd.synmvcrowd as SYN
from utils.exclusions import load_excluded_scenes


class _FixedOrder:
    """Stand-in for synmvcrowd's `random` module: the configured cameras in their fixed order (and deterministic crops),
    so camera n is the same physical camera in every frame of a scene. Only used in the main process (num_workers=0)."""

    @staticmethod
    def sample(population, k):
        return list(population)[:k]

    @staticmethod
    def choices(population, k):
        return [population[i % len(population)] for i in range(k)]


@torch.no_grad()
def run_split(split, scenes, model, device, frames_per_scene, seed, radius_cells, anchor_rel):
    fusion, att = model.fusion, model.fusion.attention
    att.keep_aux = True
    Z, N = att.num_z_anchors, att.num_cameras
    acc = defaultdict(float)
    height = {k: torch.zeros(Z, dtype=torch.float64) for k in ("people", "empty")}
    tv = {k: [0.0, 0] for k in ("people", "empty")}
    errs = {"full": [], "no_attention": []}
    std_h, std_c = [], []
    for scene, ds in scenes:
        loader = NA.make_loader([(scene, ds)], frames_per_scene, seed, num_workers=0)
        mom_h, mom_c = DH.Moments(), DH.Moments()
        for sample in tqdm(loader, desc=f"{split}/{scene}", mininterval=10):
            wld = sample["wld_map_paras"].to(device)
            ctx = {
                "ref_points_full": sample["ref_points_full"].to(device),
                "camera_paras": sample["camera_paras"].to(device),
                "wld_map_paras": wld,
            }
            H, W = int(wld[0, 3].item()), int(wld[0, 4].item())
            people = DH.people_mask(sample["gt_world_xy"][0], sample["wld_map_paras"][0], (H, W), radius_cells).to(
                device
            )
            pred = {k: torch.zeros((H, W), device=device) for k in errs}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                feats = model.features(sample["images"].to(device))
                for h0 in NA.tile_starts(H, NA.TILE[0]):
                    for w0 in NA.tile_starts(W, NA.TILE[1]):
                        base, out = fusion.parts(feats, ctx, h0, w0)
                        hgt, cam = att.aux["height"][0].float(), att.aux["camera"][0].float()  # [L, Z], [L, N]
                        full = out if base is None else base + out
                        hh, ww = min(NA.TILE[0], H - h0), min(NA.TILE[1], W - w0)
                        pred["full"][h0 : h0 + hh, w0 : w0 + ww] = model.head(full.unsqueeze(1))[0, 0, :hh, :ww].float()
                        if base is not None:
                            pred["no_attention"][h0 : h0 + hh, w0 : w0 + ww] = model.head(base.unsqueeze(1))[
                                0, 0, :hh, :ww
                            ].float()
                        pm = people[h0 : h0 + NA.TILE[0], w0 : w0 + NA.TILE[1]].reshape(-1)
                        seen = hgt.sum(-1) > 0.5
                        refs = ctx["ref_points_full"][0, :, h0 : h0 + NA.TILE[0], w0 : w0 + NA.TILE[1]]
                        vis = (refs.abs() <= 1).all(-1).any(-1).reshape(N, -1).T.float()  # [L, N]
                        share = vis / vis.sum(-1, keepdim=True).clamp(min=1)
                        tvd = 0.5 * (cam - share).abs().sum(-1)
                        for region, m in (("people", pm & seen), ("empty", ~pm & seen)):
                            height[region] += hgt[m].double().sum(0).cpu()
                            tv[region][0] += tvd[m].sum().item()
                            tv[region][1] += int(m.sum())
                        if base is not None:
                            b, d = base[0].float().flatten(1).T, out[0].float().flatten(1).T  # [L, C]
                            bn, dn = b.norm(dim=-1), d.norm(dim=-1)
                            ok = bn > 1e-6
                            for region, m in (("people", pm & ok), ("empty", ~pm & ok)):
                                acc[f"{region}_ratio"] += (dn[m] / bn[m]).sum().item()
                                acc[f"{region}_cos"] += (
                                    ((b[m] * d[m]).sum(-1) / (bn[m] * dn[m]).clamp(min=1e-6)).sum().item()
                                )
                                acc[f"{region}_coef"] += ((b[m] * d[m]).sum(-1) / bn[m] ** 2).sum().item()
                                acc[f"{region}_n"] += int(m.sum())
                        if h0 == 0 and w0 == 0:
                            mom_h.add(hgt.cpu())
                            mom_c.add(cam.cpu())
            gt = NA.gt_count_in_map(sample["gt_world_xy"][0], sample["wld_map_paras"][0], (H, W))
            for k in errs:
                errs[k].append(pred[k].sum().item() / NA.SCALE - gt)
        std_h.append(mom_h.cell_std())
        std_c.append(mom_c.cell_std())
    att.keep_aux = False

    res = {"frames": len(errs["full"])}
    for k, e in errs.items():
        if base is None and k == "no_attention":
            continue
        e = np.array(e)
        res[f"count_{k}"] = {"mae": float(np.abs(e).mean()), "bias": float(e.mean())}
    for region in ("people", "empty"):
        h = height[region] / max(tv[region][1], 1)
        res[f"height_{region}"] = {f"{z:+.2f}": round(float(v), 4) for z, v in zip(anchor_rel, h)}
        res[f"camera_tv_{region}"] = tv[region][0] / max(tv[region][1], 1)
        n = acc.get(f"{region}_n", 0)
        if n:
            res[f"correction_{region}"] = {k: acc[f"{region}_{k}"] / n for k in ("ratio", "cos", "coef")}
    res["image_dependence"] = {
        "height_std": float(np.nanmean(std_h)),
        "camera_std": float(np.nanmean(std_c)),
        "uniform_height": 1.0 / Z,
        "uniform_camera": 1.0 / N,
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="ablation_mv")
    ap.add_argument("--variant", default="mvha_res")
    ap.add_argument("--splits", default="validate,test")
    ap.add_argument("--frames_per_scene", type=int, default=5)
    ap.add_argument("--people_radius", type=float, default=1.0, help="metres")
    ap.add_argument("--anchor_below", type=float, default=2.5, help="cache setting, for the anchor labels only")
    ap.add_argument("--anchor_above", type=float, default=2.5)
    ap.add_argument("--num_workers", type=int, default=4, help="precise-BN loader only")
    args = ap.parse_args()
    device = torch.device("cuda")
    run_dir = os.path.join(src_dir, "runs", "neck_ablation", args.run_name)
    model, a = DH.load_model(run_dir, args.variant, {}, device)
    Z = a.z_anchors
    anchor_rel = np.linspace(-1.75 - args.anchor_below, args.anchor_above, Z)

    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    excluded = load_excluded_scenes()
    train = NA.load_split("train", cams_cfg, excluded, a.refs_file, Z)
    # precise BN exactly as the run's final evaluation (worker-sampled crops), before fixing the camera order
    NA.precise_bn(model, NA.make_loader(train, 1, a.val_seed + 7, args.num_workers), device)
    del train
    SYN.random = _FixedOrder

    out = {"run": args.run_name, "variant": args.variant, "frames_per_scene": args.frames_per_scene}
    for split in args.splits.split(","):
        scenes = NA.load_split(split, cams_cfg, excluded, a.refs_file, Z)
        r = run_split(
            split, scenes, model, device, args.frames_per_scene, a.val_seed, round(args.people_radius * 5), anchor_rel
        )
        out[split] = r
        print(f"\n=== {split} ({r['frames']} frames) ===")
        print(json.dumps(r, indent=2))
    os.makedirs(os.path.join(src_dir, "diagnostics", "mvha"), exist_ok=True)
    path = os.path.join(src_dir, "diagnostics", "mvha", f"{args.run_name}_{args.variant}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
