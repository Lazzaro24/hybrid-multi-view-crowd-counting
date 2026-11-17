"""
eval_map_quality.py

Does the predicted density sit on the people, not just sum to the right count? The neck ablation ranks necks by
count only; the occlusion evaluation (doc/neck_ablation.md section 8) showed that ca_sa recovers far less density
mass around clearly visible people than the homography (53% vs 33% missed) while counting as well or better. Per
held-out frame, on the whole-scene map:

  - localisation as in the paper (evaluation/evaluate_paper_protocol.py: min-max normalise, > 0.2, greedy NMS 2.5 px,
    Hungarian matching within 0.5 m): pooled Precision, Recall, F1, MODA, MODP. The 0.2-of-maximum rule is sensitive
    to blurry maps (the frozen-backbone baseline has precision 29%), so compare models with each other.
  - GAME(L), L = 0..3: the map cut into 4^L equal cells, sum over cells of |predicted - true count|. GAME(0) is the
    count error; higher levels punish the right count in the wrong place. No detection threshold involved.
  - far mass: share of the predicted mass more than 1 m / 2 m from every GT person.
  - reference for every metric: the GT density map built with the training kernel (a perfect prediction). Under the
    paper's detection rule even this map has many false positives (the 0.2-of-maximum threshold and 2.5 px NMS split
    a blob of this kernel into several peaks), and GAME(L > 0) is not 0 (mass spills over cell borders).

Models: ablation checkpoints (EMA, precise BN, K-means 5 cameras, 200x200 tiles; same frames for every model) and the
whole-scene baselines under the paper protocol (seeded random 5-of-all-camera draw per frame). GT = people visible in
at least one active view (dataset), rasterised with evaluate_paper_protocol.gt_pixels.

Writes diagnostics/map_quality/<name>.csv (per frame, resumable) and summary.json; prints a table per split.

Usage (from src):
    uv run python evaluation/eval_map_quality.py
    uv run python evaluation/eval_map_quality.py --analyze_only
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
src_dir = os.path.dirname(current_dir)
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

import neck_ablation as NA
import diagnose_hca as DH
import eval_per_frame as PF
import evaluate_paper_protocol as EP
from utils.exclusions import load_excluded_scenes
from losses.gaussian_mse import get_map_kernel

OUT = os.path.join(src_dir, "diagnostics", "map_quality")
GAME_LEVELS = (0, 1, 2, 3)
FAR_M = (1.0, 2.0)
FIELDS = (
    ["split", "scene", "snapshot", "gt", "pred", "tp", "fp", "fn", "modp_sum"]
    + [f"game{L}" for L in GAME_LEVELS]
    + [f"far{d:g}m" for d in FAR_M]
    + ["ref_tp", "ref_fp", "ref_fn", "ref_modp_sum"]
    + [f"ref_game{L}" for L in GAME_LEVELS]
    + [f"ref_far{d:g}m" for d in FAR_M]
    + ["dense_zone_frac"]
)  # proposed only: share of the map routed dense


def game(pred, gts, L):
    """Sum over a 2^L x 2^L grid of |predicted - GT count| per cell (pred in people, numpy [H, W])."""
    H, W = pred.shape
    rb, cb = np.linspace(0, H, 2**L + 1).astype(int), np.linspace(0, W, 2**L + 1).astype(int)
    cs = np.pad(pred.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    cell_pred = cs[rb[1:]][:, cb[1:]] - cs[rb[:-1]][:, cb[1:]] - cs[rb[1:]][:, cb[:-1]] + cs[rb[:-1]][:, cb[:-1]]
    cell_gt = np.histogram2d(gts[:, 0], gts[:, 1], bins=[rb, cb])[0] if len(gts) else np.zeros_like(cell_pred)
    return float(np.abs(cell_pred - cell_gt).sum())


def frame_metrics(pred, gts, wld, kernel, args):
    """pred: [H, W] tensor in density units (sum / SCALE = count); gts: [N, 2] GT pixels (row, col)."""
    r = wld[0, 0].item()  # pixels per metre
    tp, fp, fn, modp = EP.match(EP.detect(pred, args.cls_thres, args.nms_radius_px), gts, args.match_radius_m * r)
    p = (pred / EP.SCALE).float().cpu().numpy()
    out = {"gt": len(gts), "pred": float(p.sum()), "tp": tp, "fp": fp, "fn": fn, "modp_sum": modp}
    for L in GAME_LEVELS:
        out[f"game{L}"] = game(p, gts, L)
    # distance of every cell to the nearest GT person, and the GT density with the training kernel as reference
    dots = np.zeros(p.shape, bool)
    if len(gts):
        dots[gts[:, 0].astype(int), gts[:, 1].astype(int)] = True
    dist = distance_transform_edt(~dots) / r if len(gts) else np.full(p.shape, np.inf)
    g = (
        F.conv2d(
            torch.from_numpy(dots.astype(np.float32))[None, None].to(kernel.device),
            kernel,
            padding=kernel.shape[-1] // 2,
        )[0, 0]
        .cpu()
        .numpy()
    )
    pos = np.clip(p, 0, None)
    for d in FAR_M:
        out[f"far{d:g}m"] = float(pos[dist > d].sum() / max(pos.sum(), 1e-8))
        out[f"ref_far{d:g}m"] = float(g[dist > d].sum() / max(g.sum(), 1e-8))
    # reference: the GT density itself (a perfect prediction) under the same detection rule and grid
    out["ref_tp"], out["ref_fp"], out["ref_fn"], out["ref_modp_sum"] = EP.match(
        EP.detect(torch.from_numpy(g), args.cls_thres, args.nms_radius_px), gts, args.match_radius_m * r
    )
    for L in GAME_LEVELS:
        out[f"ref_game{L}"] = game(g, gts, L)
    return out


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {path} ({len(rows)} frames)")


@torch.no_grad()
def run_ablation(args, kernel, device):
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    excluded = load_excluded_scenes()
    splits = {}
    for spec in [s for s in args.models.split(",") if s]:
        run, variant = spec.split(":")
        path = os.path.join(OUT, f"{run}__{variant}.csv")
        if os.path.exists(path) and not args.overwrite:
            print(f"[skip] {spec}: {path} exists")
            continue
        run_dir = os.path.join(src_dir, "runs", "neck_ablation", run)
        model, a = DH.load_model(run_dir, variant, {"z_anchors": 4, "refs_file": "master_refs.pt"}, device)
        key = (a.refs_file, a.z_anchors)
        if key not in splits:
            splits[key] = {s: NA.load_split(s, cams_cfg, excluded, *key) for s in ("train", "validate", "test")}
        data = splits[key]
        NA.precise_bn(model, NA.make_loader(data["train"], 1, a.val_seed + 7, args.num_workers), device)
        rows = []
        for split in ("validate", "test"):
            scenes = data[split]
            names, bounds = [n for n, _ in scenes], np.cumsum([0] + [len(ds) for _, ds in scenes])
            loader = NA.make_loader(
                scenes, args.frames_per_scene, args.seed + (0 if split == "validate" else 13), args.num_workers
            )
            for sample, idx in zip(tqdm(loader, desc=f"{spec} {split}", mininterval=30), loader.sampler):
                k = int(np.searchsorted(bounds, idx, side="right") - 1)
                pred = PF.whole_map(model, sample, device)
                gts = EP.gt_pixels(sample["gt_world_xy"], sample["wld_map_paras"], tuple(pred.shape))
                rows.append(
                    {
                        "split": split,
                        "scene": names[k],
                        "snapshot": scenes[k][1].snapshots[idx - bounds[k]],
                        **frame_metrics(pred, gts, sample["wld_map_paras"], kernel, args),
                    }
                )
        write(path, rows)
        del model
        torch.cuda.empty_cache()


@torch.no_grad()
def proposed_zone(model, sample, device, routing, shape):
    """Dense-expert zone of the full scene for a routing token: None = the learned router; 'dense' / 'sparse' = one
    expert everywhere; 'dzone<thr>' = the dense expert where its OWN predicted local density (5 x 5 m window)
    is at least <thr> persons/m^2, the sparse expert elsewhere -- a GT-free rule (doc/head_ablation.md §6e/§6i).
    A trailing 'p' (e.g. 'dzone0.25p', 'sparsep') is handled by the caller (sparse output rendered as peaks)."""
    base = routing[:-1] if routing.endswith("p") else routing
    if base == "learned":
        return None
    if base in ("dense", "sparse"):
        return torch.full(shape, base == "dense", dtype=torch.bool)
    if base.startswith("dzone"):  # shared with the paper protocol (evaluate_paper_protocol.density_zone)
        return EP.density_zone(model, sample, device, float(base[len("dzone") :]), shape)
    raise ValueError(f"unknown routing {routing!r}")


@torch.no_grad()
def run_proposed(args, kernel, device):
    """Proposed-model checkpoints (train_proposed_multiscene.py) on exactly the ablation frames and cameras (fixed
    K-means 5 cameras, same seeds), so a proposed head on an ablation fusion compares with that fusion's own
    baseline-head run on identical inputs. --proposed name=checkpoint:routing, routing learned | dense | sparse |
    dzone<thr> (dense expert where its own predicted density >= thr persons/m^2), each with an optional trailing 'p'
    for the sparse expert rendered as NMS peaks (kernel 3, threshold 0.064)."""
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    excluded = load_excluded_scenes()
    data = None
    for spec in [s for s in args.proposed.split(",") if s]:
        name, rest = spec.split("=")
        weights, routing = rest.rsplit(":", 1)
        path = os.path.join(OUT, f"proposed__{name}.csv")
        if os.path.exists(path) and not args.overwrite:
            print(f"[skip] proposed {name}: {path} exists")
            continue
        model, _ = EP.load_model("proposed", os.path.join(src_dir, weights), device)
        if routing.endswith("p"):
            model.T.apply_sparse_render(model, {"mode": "peaks", "kernel": 3, "threshold": 0.064})
            model.head.render_kernel = model.head.render_kernel.to(device)
        print(f"[load] proposed {name}: {weights} ({routing} routing, sparse render {model.head.sparse_render})")
        if data is None:
            data = {s: NA.load_split(s, cams_cfg, excluded, "master_refs.pt", 4) for s in ("validate", "test")}
        rows = []
        for split in ("validate", "test"):
            scenes = data[split]
            names, bounds = [n for n, _ in scenes], np.cumsum([0] + [len(ds) for _, ds in scenes])
            loader = NA.make_loader(
                scenes, args.frames_per_scene, args.seed + (0 if split == "validate" else 13), args.num_workers
            )
            for sample, idx in zip(tqdm(loader, desc=f"proposed {name} {split}", mininterval=30), loader.sampler):
                k = int(np.searchsorted(bounds, idx, side="right") - 1)
                shape = (int(sample["wld_map_paras"][0, 3].item()), int(sample["wld_map_paras"][0, 4].item()))
                zone = proposed_zone(model, sample, device, routing, shape)
                pred = EP.predict_proposed(model, sample, device, zone)
                gts = EP.gt_pixels(sample["gt_world_xy"], sample["wld_map_paras"], tuple(pred.shape))
                rows.append(
                    {
                        "split": split,
                        "scene": names[k],
                        "snapshot": scenes[k][1].snapshots[idx - bounds[k]],
                        **frame_metrics(pred, gts, sample["wld_map_paras"], kernel, args),
                        "dense_zone_frac": float(zone.float().mean()) if zone is not None else float("nan"),
                    }
                )
        write(path, rows)
        del model
        torch.cuda.empty_cache()


@torch.no_grad()
def run_baselines(args, kernel, device):
    excluded = load_excluded_scenes()
    for spec in [s for s in args.baselines.split(",") if s]:
        name, weights = spec.split("=")
        path = os.path.join(OUT, f"baseline__{name}.csv")
        if os.path.exists(path) and not args.overwrite:
            print(f"[skip] baseline {name}: {path} exists")
            continue
        model, epoch = EP.load_model("baseline", os.path.join(src_dir, weights), device)
        print(f"[load] baseline {name}: {weights} (epoch {epoch})")
        rows = []
        for split in ("validate", "test"):
            scenes, bounds, indices, loader = PF.baseline_loader(
                split, excluded, args.frames_per_scene, 1, args.seed, args.num_workers
            )
            for sample, idx in zip(tqdm(loader, desc=f"baseline {name} {split}", mininterval=30), indices):
                k = int(np.searchsorted(bounds, idx, side="right") - 1)
                pred = EP.predict_baseline(model, sample, device)
                gts = EP.gt_pixels(sample["gt_world_xy"], sample["wld_map_paras"], tuple(pred.shape))
                rows.append(
                    {
                        "split": split,
                        "scene": scenes[k][0],
                        "snapshot": scenes[k][1].snapshots[idx - bounds[k]],
                        **frame_metrics(pred, gts, sample["wld_map_paras"], kernel, args),
                    }
                )
        write(path, rows)
        del model
        torch.cuda.empty_cache()


def pooled(rs, pre):
    def v(k):
        return np.array([float(r[pre + k]) for r in rs])

    g = np.array([float(r["gt"]) for r in rs])
    tp, fp, fn, gt = v("tp").sum(), v("fp").sum(), v("fn").sum(), g.sum()
    prec, rec = tp / max(tp + fp, 1), tp / max(gt, 1)
    return {
        "mae": float(np.abs(v("pred") - g).mean()) if not pre else 0.0,
        "precision": 100 * prec,
        "recall": 100 * rec,
        "f1": 100 * 2 * prec * rec / max(prec + rec, 1e-8),
        "moda": 100 * (1 - (fp + fn) / max(gt, 1)),
        "modp": 100 * v("modp_sum").sum() / max(tp, 1),
        **{f"game{L}": float(v(f"game{L}").mean()) for L in GAME_LEVELS},
        **{f"far{d:g}m": 100 * float(v(f"far{d:g}m").mean()) for d in FAR_M},
    }


def analyze(args):
    names = [("{}:{}".format(*s.split(":")), "{}__{}.csv".format(*s.split(":"))) for s in args.models.split(",") if s]
    names += [
        (f"baseline:{b.split('=')[0]}", f"baseline__{b.split('=')[0]}.csv") for b in args.baselines.split(",") if b
    ]
    names += [
        (f"proposed:{p.split('=')[0]}", f"proposed__{p.split('=')[0]}.csv") for p in args.proposed.split(",") if p
    ]
    summary = {}
    for name, fn in names:
        path = os.path.join(OUT, fn)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        res = {}
        for split in ("validate", "test"):
            rs = [r for r in rows if r["split"] == split]
            res[split] = {"frames": len(rs), **pooled(rs, "")}
            res[split]["reference_gt_density"] = pooled(rs, "ref_")
        summary[name] = res
    for split in ("validate", "test"):
        print(f"\n== {split}")
        print(
            f"{'model':<28}{'MAE':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'MODA':>8}{'MODP':>7}"
            + "".join(f"{'GAME' + str(L):>8}" for L in GAME_LEVELS)
            + "".join(f"{f'far>{d:g}m':>9}" for d in FAR_M)
        )
        for name, res in summary.items():
            r = res[split]
            print(
                f"{name:<28}{r['mae']:7.1f}{r['precision']:7.1f}{r['recall']:7.1f}{r['f1']:7.1f}{r['moda']:8.1f}"
                f"{r['modp']:7.1f}"
                + "".join(f"{r[f'game{L}']:8.1f}" for L in GAME_LEVELS)
                + "".join(f"{r[f'far{d:g}m']:8.1f}%" for d in FAR_M)
            )
        refs = {}  # perfect-map reference, once per protocol (frame set)
        for name, res in summary.items():
            refs.setdefault("baseline frames" if name.startswith("baseline") else "ablation frames", res[split])
        for label, res in refs.items():
            r = res["reference_gt_density"]
            print(
                f"{'  GT density, ' + label:<28}{'-':>7}{r['precision']:7.1f}{r['recall']:7.1f}"
                f"{r['f1']:7.1f}{r['moda']:8.1f}{r['modp']:7.1f}"
                + "".join(f"{r[f'game{L}']:8.1f}" for L in GAME_LEVELS)
                + "".join(f"{r[f'far{d:g}m']:8.1f}%" for d in FAR_M)
            )
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "models": summary}, f, indent=1)
    print(f"[saved] {os.path.join(OUT, 'summary.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default="ablation_v2:homography_256,ablation_v2:ca_sa,ablation_k8:ca_sa,ablation_k8:hca,ablation_mv:mvha_res",
    )
    ap.add_argument("--baselines", default="run_1=runs/baseline/run_1/best.pth,ot=runs/baseline/ot/best.pth")
    ap.add_argument(
        "--proposed",
        default="",
        help="name=checkpoint:routing,... proposed checkpoints on the ablation frames; routing learned | dense | sparse",
    )
    ap.add_argument("--frames_per_scene", type=int, default=50, help="seeded frames per held-out scene (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cls_thres", type=float, default=0.2)
    ap.add_argument("--nms_radius_px", type=float, default=2.5)
    ap.add_argument("--match_radius_m", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.analyze_only:
        device = torch.device("cuda")
        kernel = get_map_kernel(sigma=NA.MAP_SIGMA, kernel_size=NA.MAP_KERNEL_SIZE).to(device)
        kernel = (kernel / kernel.sum()).float().reshape(1, 1, *kernel.shape[-2:])
        run_ablation(args, kernel, device)
        run_baselines(args, kernel, device)
        run_proposed(args, kernel, device)
    analyze(args)


if __name__ == "__main__":
    main()
