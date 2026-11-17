"""
eval_per_frame.py

Per-frame predictions of the final neck-ablation weights on the held-out scenes, to separate three things that the
per-scene MAE mixes (next step 2 in doc/neck_ablation.md, step 2b in doc/calibration_check_plan.md):

  - fusion quality: does the model follow the count from frame to frame inside a scene (within-scene Pearson r of
    predicted vs true count, averaged over scenes)?
  - count calibration: MAE after one global correction fitted on val and applied to test (scale a*pred, and affine
    a*pred + b), and MAE after removing each scene's own mean error (oracle, upper bound on what scene offsets cost).
  - camera calibration: the json camera changes per snapshot (static cameras, noisy per-frame estimates). Within a
    scene, is a frame's error (after removing the scene's mean error) larger when its reprojection miss is larger?

Evaluation is the ablation's own protocol (EMA weights from last.pth, precise BN on one frame per train scene with the
run's seed, whole-scene 200x200 tiling). Per frame we also store the camera miss: median / p90 over the 5 cameras of
|projected head - dataset pixel| (utils/check_calibration.miss_one).

Writes diagnostics/per_frame/<run>__<variant>.csv (resumable: finished models are skipped) and summary.json.

Also the whole-scene baselines (train_baseline.py: 200-epoch MSE run and the OT fine-tune) under the paper protocol
(all cameras are candidates, a seeded random 5-view draw per frame, GT = people visible in a drawn view); their miss
is measured on the drawn cameras (recovered from camera_paras). Written as baseline__<name>.csv.

Usage (from src):
    uv run python evaluation/eval_per_frame.py                          # 5 candidates, all held-out frames
    uv run python evaluation/eval_per_frame.py --analyze_only
"""

import os
import sys
import csv
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

import neck_ablation as NA
import diagnose_hca as DH
from utils.exclusions import load_excluded_scenes
from utils.check_calibration import miss_one

OUT = os.path.join(src_dir, "diagnostics", "per_frame")
FIELDS = ["split", "scene", "snapshot", "gt", "pred", "miss_median", "miss_p90"]


def frame_miss(ds, snapshot, cams=None):
    m = []
    for cam in ds.camera_ids if cams is None else cams:
        jp = os.path.join(ds.scene_dir, str(snapshot), "jsons", f"{cam}.json")
        if os.path.exists(jp):
            with open(jp) as f:
                m.append(miss_one(json.load(f))[0])
    m = np.concatenate(m) if m else np.zeros(0)
    m = np.where(np.isfinite(m), m, 1e4)
    return (float(np.median(m)), float(np.percentile(m, 90))) if len(m) else (float("nan"), float("nan"))


@torch.no_grad()
def whole_map(model, sample, device):
    """Whole-scene density map [H, W] of an ablation model (200x200 tiles, as neck_ablation.evaluate)."""
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
    return pred


@torch.no_grad()
def predict(model, split, scenes, loader, device):
    names = [n for n, _ in scenes]
    bounds = np.cumsum([0] + [len(ds) for _, ds in scenes])
    rows = []
    model.eval()
    for sample, idx in zip(tqdm(loader, desc=split, mininterval=30), loader.sampler):
        k = int(np.searchsorted(bounds, idx, side="right") - 1)
        ds = scenes[k][1]
        snap = ds.snapshots[idx - bounds[k]]
        pred = whole_map(model, sample, device)
        H, W = pred.shape
        gt = NA.gt_count_in_map(sample["gt_world_xy"][0], sample["wld_map_paras"][0], (H, W))
        mm, mp = frame_miss(ds, snap)
        rows.append(
            {
                "split": split,
                "scene": names[k],
                "snapshot": snap,
                "gt": gt,
                "pred": pred.sum().item() / NA.SCALE,
                "miss_median": mm,
                "miss_p90": mp,
            }
        )
    return rows


def run_models(args, device):
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    excluded = load_excluded_scenes()
    splits = {}  # (refs_file, z_anchors) -> {split: scenes}
    for spec in args.models.split(","):
        run, variant = spec.split(":")
        path = os.path.join(OUT, f"{run}__{variant}.csv")
        if os.path.exists(path) and not args.overwrite:
            print(f"[skip] {spec}: {path} exists")
            continue
        run_dir = os.path.join(src_dir, "runs", "neck_ablation", run)
        model, a = DH.load_model(
            run_dir, variant, {"z_anchors": 4, "refs_file": "master_refs.pt"}, device
        )  # defaults of runs before these args existed
        key = (a.refs_file if getattr(a, "refs_file", None) else "master_refs.pt", getattr(a, "z_anchors", None) or 4)
        if key not in splits:
            splits[key] = {s: NA.load_split(s, cams_cfg, excluded, *key) for s in ("train", "validate", "test")}
        data = splits[key]
        NA.precise_bn(model, NA.make_loader(data["train"], 1, a.val_seed + 7, args.num_workers), device)
        rows = []
        for split in ("validate", "test"):
            loader = NA.make_loader(
                data[split], args.frames_per_scene, a.val_seed + (0 if split == "validate" else 13), args.num_workers
            )
            rows += predict(model, split, data[split], loader, device)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"[saved] {path} ({len(rows)} frames)")
        del model
        torch.cuda.empty_cache()


def drawn_cams(ds, snapshot, camera_paras):
    """The dataset does not return which cameras it drew; recover them by matching each view's tvec (last 3 entries
    of its camera_paras row) against the cameras' jsons of this snapshot."""
    tv = {}
    for cam in ds.camera_ids:
        jp = os.path.join(ds.scene_dir, str(snapshot), "jsons", f"{cam}.json")
        if os.path.exists(jp):
            with open(jp) as f:
                tv[cam] = np.array(json.load(f)["tvec"], np.float64)
    out = []
    for row in camera_paras.reshape(-1, camera_paras.shape[-1]).double().numpy():
        out.append(min(tv, key=lambda c: np.abs(tv[c] - row[-3:]).max()))
    return out


def baseline_loader(split, excluded, frames_per_scene, draws, seed, num_workers):
    """Paper-protocol loader (evaluation/evaluate_paper_protocol.py): every camera of a scene is a candidate, the
    dataset draws 5 per sample; `draws` samples per frame; all frames or a seeded subset of frames_per_scene."""
    from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset

    sys.path.append(os.path.join(src_dir, "evaluation"))
    import evaluate_paper_protocol as EP

    split_dir = os.path.join(src_dir, "datasets", "SynMVCrowd", "data", split)
    scenes = [
        (
            s,
            SynMVCrowdDataset(
                scene_dir=os.path.join(split_dir, s), camera_ids=None, num_sampled_cams=5, load_refs=False
            ),
        )
        for s in sorted(os.listdir(split_dir), key=lambda x: int(x.split("_")[1]))
        if os.path.isdir(os.path.join(split_dir, s)) and s not in excluded
    ]
    bounds = np.cumsum([0] + [len(ds) for _, ds in scenes])
    indices = [i for i in range(bounds[-1]) for _ in range(draws)]
    if frames_per_scene:
        rng = np.random.RandomState(seed)
        indices = [
            bounds[k] + f
            for k, (_, ds) in enumerate(scenes)
            for f in sorted(rng.choice(len(ds), min(frames_per_scene, len(ds)), replace=False))
            for _ in range(draws)
        ]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([ds for _, ds in scenes]),
        batch_size=1,
        sampler=indices,
        num_workers=num_workers,
        worker_init_fn=EP.seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    return scenes, bounds, indices, loader


@torch.no_grad()
def run_baselines(args, device):
    """Baseline (whole-scene model of train_baseline.py) under the paper protocol of
    evaluation/evaluate_paper_protocol.py: every camera of the scene is a candidate, a seeded random draw of 5 per
    frame (--baseline_draws draws per frame), GT = people visible in at least one drawn view. The miss is measured on
    the drawn cameras."""
    sys.path.append(os.path.join(src_dir, "evaluation"))
    import evaluate_paper_protocol as EP

    excluded = load_excluded_scenes()
    for spec in args.baselines.split(","):
        name, weights = spec.split("=")
        path = os.path.join(OUT, f"baseline__{name}.csv")
        if os.path.exists(path) and not args.overwrite:
            print(f"[skip] baseline {name}: {path} exists")
            continue
        model, epoch = EP.load_model("baseline", os.path.join(src_dir, weights), device)
        print(f"[load] baseline {name}: {weights} (epoch {epoch})")
        rows = []
        for split in ("validate", "test"):
            scenes, bounds, indices, loader = baseline_loader(
                split, excluded, args.frames_per_scene, args.baseline_draws, args.seed, args.num_workers
            )
            for sample, idx in zip(tqdm(loader, desc=f"baseline {name} {split}", mininterval=30), indices):
                k = int(np.searchsorted(bounds, idx, side="right") - 1)
                ds = scenes[k][1]
                snap = ds.snapshots[idx - bounds[k]]
                wld = sample["wld_map_paras"]
                shape = (int(wld[0, 3].item()), int(wld[0, 4].item()))
                pred = EP.predict_baseline(model, sample, device)
                gt = len(EP.gt_pixels(sample["gt_world_xy"], wld, shape))
                cams = drawn_cams(ds, snap, sample["camera_paras"])
                mm, mp = frame_miss(ds, snap, cams)
                rows.append(
                    {
                        "split": split,
                        "scene": scenes[k][0],
                        "snapshot": snap,
                        "gt": gt,
                        "pred": pred.sum().item() / EP.SCALE,
                        "miss_median": mm,
                        "miss_p90": mp,
                    }
                )
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"[saved] {path} ({len(rows)} frames)")
        del model
        torch.cuda.empty_cache()


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze(args):
    summary = {}
    specs = [
        (spec, os.path.join(OUT, "{}__{}.csv".format(*spec.split(":")))) for spec in args.models.split(",") if spec
    ]
    specs += [
        (f"baseline:{b.split('=')[0]}", os.path.join(OUT, f"baseline__{b.split('=')[0]}.csv"))
        for b in args.baselines.split(",")
        if b
    ]
    for spec, path in specs:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        sp = np.array([r["split"] for r in rows])
        sc = np.array([r["scene"] for r in rows])
        gt = np.array([float(r["gt"]) for r in rows])
        pr = np.array([float(r["pred"]) for r in rows])
        miss = np.array([float(r["miss_p90"]) for r in rows])
        err = pr - gt
        # scene-mean error removed: what is left is the frame-to-frame error inside a scene
        resid = err.copy()
        for s in np.unique(sc):
            resid[sc == s] -= err[sc == s].mean()
        v, t = sp == "validate", sp == "test"
        a_scale = float((pr[v] * gt[v]).sum() / (pr[v] ** 2).sum())
        A = np.stack([pr[v], np.ones(v.sum())], 1)
        a_aff, b_aff = np.linalg.lstsq(A, gt[v], rcond=None)[0]
        res = {}
        for name, m in (("val", v), ("test", t)):
            within_r, slopes = [], []
            for s in np.unique(sc[m]):
                k = m & (sc == s)
                if k.sum() > 2 and gt[k].std() > 0 and pr[k].std() > 0:
                    within_r.append(np.corrcoef(pr[k], gt[k])[0, 1])
                    slopes.append(np.polyfit(gt[k], pr[k], 1)[0])
            hi, lo = m & (miss > args.bad_px), m & (miss <= args.good_px)
            res[name] = {
                "frames": int(m.sum()),
                "mae": float(np.abs(err[m]).mean()),
                "bias": float(err[m].mean()),
                "within_scene_r": float(np.mean(within_r)),
                "within_scene_slope": float(np.median(slopes)),
                "mae_scale_fit_on_val": float(np.abs(a_scale * pr[m] - gt[m]).mean()),
                "mae_affine_fit_on_val": float(np.abs(a_aff * pr[m] + b_aff - gt[m]).mean()),
                "mae_scene_offset_removed": float(np.abs(resid[m]).mean()),
                "calib_spearman_absresid_vs_miss": spearman(np.abs(resid[m]), miss[m]),
                "calib_absresid_bad_frames": float(np.abs(resid[hi]).mean()) if hi.any() else None,
                "calib_absresid_good_frames": float(np.abs(resid[lo]).mean()) if lo.any() else None,
                "calib_bad_frames": int(hi.sum()),
                "calib_good_frames": int(lo.sum()),
            }
        # within-scene, per scene that has both good and bad frames: |resid| bad - good
        per_scene = {}
        for s in np.unique(sc):
            k = sc == s
            hi, lo = k & (miss > args.bad_px), k & (miss <= args.good_px)
            if hi.sum() >= 3 and lo.sum() >= 3:
                per_scene[s] = {
                    "bad": int(hi.sum()),
                    "good": int(lo.sum()),
                    "absresid_bad": float(np.abs(resid[hi]).mean()),
                    "absresid_good": float(np.abs(resid[lo]).mean()),
                    "signed_resid_bad": float(resid[hi].mean()),
                }
        res["scale_fit"] = a_scale
        res["affine_fit"] = [float(a_aff), float(b_aff)]
        res["scenes_with_bad_and_good_frames"] = per_scene
        summary[spec] = res

    print(
        f"\n{'model':<28}{'split':<6}{'MAE':>7}{'bias':>8}{'r_in':>6}{'slope':>7}{'MAE_a':>7}{'MAE_ab':>8}"
        f"{'-offset':>8}{'rho_miss':>9}{'bad/good |res|':>18}"
    )
    for spec, res in summary.items():
        for name in ("val", "test"):
            r = res[name]
            bg = (
                (
                    f"{r['calib_absresid_bad_frames']:.1f}/{r['calib_absresid_good_frames']:.1f} "
                    f"({r['calib_bad_frames']}/{r['calib_good_frames']})"
                )
                if r["calib_absresid_bad_frames"]
                else "-"
            )
            print(
                f"{spec:<28}{name:<6}{r['mae']:7.1f}{r['bias']:+8.1f}{r['within_scene_r']:6.2f}"
                f"{r['within_scene_slope']:7.2f}{r['mae_scale_fit_on_val']:7.1f}{r['mae_affine_fit_on_val']:8.1f}"
                f"{r['mae_scene_offset_removed']:8.1f}{r['calib_spearman_absresid_vs_miss']:+9.2f}{bg:>18}"
            )
    print(
        f"\nr_in = within-scene Pearson r(pred, gt), mean over scenes; slope = median within-scene d pred / d gt; "
        f"MAE_a / MAE_ab = after scale / affine fitted on val; -offset = each scene's mean error removed; "
        f"rho_miss = Spearman(|frame error - scene mean|, frame miss p90); bad > {args.bad_px} px, good <= {args.good_px} px"
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
    ap.add_argument(
        "--baselines",
        default="run_1=runs/baseline/run_1/best.pth,ot=runs/baseline/ot/best.pth",
        help="name=weights of whole-scene baselines (paper protocol); empty to skip",
    )
    ap.add_argument("--baseline_draws", type=int, default=1, help="random 5-view draws per frame for the baselines")
    ap.add_argument("--seed", type=int, default=0, help="baseline view draws (paper protocol seed)")
    ap.add_argument("--frames_per_scene", type=int, default=0, help="0 = every frame of every held-out scene")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    ap.add_argument("--bad_px", type=float, default=20.0, help="frame p90 miss above this: badly calibrated frame")
    ap.add_argument("--good_px", type=float, default=5.0, help="frame p90 miss at most this: well calibrated frame")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.analyze_only:
        if args.models:
            run_models(args, torch.device("cuda"))
        if args.baselines:
            run_baselines(args, torch.device("cuda"))
    analyze(args)


if __name__ == "__main__":
    main()
