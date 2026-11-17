"""
eval_view_dropout.py

How much does each fusion suffer when one of the 5 cameras is missing at test time (camera failure)? Per held-out
frame, the whole-scene prediction with all 5 cameras and with each camera left out in turn (no retraining).

Leaving a camera out = its contribution is zero everywhere, which is how both fusions already treat a camera that does
not see a BEV cell:
  - homography (HomographyFusion): projected features are 0 outside a camera's view, then max over cameras ->
    the left-out camera's projected features are set to 0;
  - attention (LegacyAttentionFusion, DeformableCrossAttention): samples outside the image return 0 and the cameras are
    summed -> the left-out camera's reference points are moved outside the image, so all its samples are 0.

Writes diagnostics/view_dropout/<run>__<variant>.csv (split, scene, snapshot, gt, dropped camera slot or -1 for
none, pred) and prints per split: MAE, bias, count change when a camera is dropped, and MAE after correcting each
scene's offset from k labelled frames (offset estimated on intact frames = calibrated before the camera failed, or on
dropout frames = recalibrated after it).

Usage (from src):
    uv run python evaluation/eval_view_dropout.py
    uv run python evaluation/eval_view_dropout.py --analyze_only
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
import eval_per_frame as PF
from utils.exclusions import load_excluded_scenes

OUT = os.path.join(src_dir, "diagnostics", "view_dropout")
FIELDS = ["split", "scene", "snapshot", "gt", "dropped", "pred"]


def homography_forward(fusion, drop):
    """HomographyFusion.forward with camera slot `drop` set to 0 after projection (= sees nothing)."""

    def forward(features, ctx, h0, w0):
        B, N, C, H, W = features.shape
        if fusion.cam_compression is not None:
            features = fusion.cam_compression(features.reshape(B * N, C, H, W)).view(B, N, fusion.out_channels, H, W)
        feats = fusion.feature_selector(features)
        hw = torch.tensor([[h0, w0]], device=features.device, dtype=torch.float32).expand(B, 2)
        proj = fusion.projector(feats, ctx["camera_paras"], ctx["wld_map_paras"], hw)
        if drop >= 0:
            proj = proj.clone()
            proj[:, drop] = 0
        return proj.max(dim=1).values

    return forward


@torch.no_grad()
def predict_dropped(model, variant, sample, device, drop):
    if drop < 0:
        return PF.whole_map(model, sample, device)
    if variant.startswith("homography"):
        orig = model.fusion.forward
        model.fusion.forward = homography_forward(model.fusion, drop)
        try:
            return PF.whole_map(model, sample, device)
        finally:
            model.fusion.forward = orig
    s = dict(sample)
    refs = sample["ref_points_full"].clone()
    refs[:, drop] = -10.0  # normalised grid coords far outside [-1, 1] -> zero samples
    s["ref_points_full"] = refs
    return PF.whole_map(model, s, device)


def run(args, device):
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    excluded = load_excluded_scenes()
    splits = {}
    for spec in args.models.split(","):
        run_name, variant = spec.split(":")
        path = os.path.join(OUT, f"{run_name}__{variant}.csv")
        if os.path.exists(path) and not args.overwrite:
            print(f"[skip] {spec}: {path} exists")
            continue
        model, a = DH.load_model(
            os.path.join(src_dir, "runs", "neck_ablation", run_name),
            variant,
            {"z_anchors": 4, "refs_file": "master_refs.pt"},
            device,
        )
        if not (variant.startswith("homography") or variant in ("ca", "sa_ca", "ca_sa")):
            raise SystemExit(f"{variant}: view dropout is implemented for homography* and ca / sa_ca / ca_sa only")
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
                snap = scenes[k][1].snapshots[idx - bounds[k]]
                n_cams = sample["images"].shape[1]
                for drop in [-1] + list(range(n_cams)):
                    pred = predict_dropped(model, variant, sample, device, drop)
                    if drop < 0:
                        gt = NA.gt_count_in_map(sample["gt_world_xy"][0], sample["wld_map_paras"][0], tuple(pred.shape))
                    rows.append(
                        {
                            "split": split,
                            "scene": names[k],
                            "snapshot": snap,
                            "gt": gt,
                            "dropped": drop,
                            "pred": pred.sum().item() / NA.SCALE,
                        }
                    )
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"[saved] {path} ({len(rows)} predictions)")
        del model
        torch.cuda.empty_cache()


def kshot_mae(err_cal, err_eval, k, rng, draws):
    """Per scene: offset = mean error of k random labelled frames of err_cal, MAE on the other frames of err_eval.
    err_*: dict scene -> {snapshot: error}. Frames are drawn by snapshot so calibration and evaluation never share a
    frame."""
    out = []
    for _ in range(draws):
        maes = []
        for s in err_eval:
            snaps = sorted(err_eval[s])
            lab = set(rng.choice(snaps, k, replace=False).tolist())
            off = np.mean([err_cal[s][q] for q in lab])
            maes.append(np.mean([abs(e - off) for q in snaps if q not in lab for e in np.atleast_1d(err_eval[s][q])]))
        out.append(np.mean(maes))
    return float(np.mean(out))


def analyze(args):
    rng = np.random.default_rng(0)
    summary = {}
    for spec in args.models.split(","):
        run_name, variant = spec.split(":")
        path = os.path.join(OUT, f"{run_name}__{variant}.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        res = {}
        for split in ("validate", "test"):
            rs = [r for r in rows if r["split"] == split]
            intact = {
                (r["scene"], r["snapshot"]): float(r["pred"]) - float(r["gt"]) for r in rs if r["dropped"] == "-1"
            }
            drop = {}
            for r in rs:
                if r["dropped"] != "-1":
                    drop.setdefault((r["scene"], r["snapshot"]), []).append(float(r["pred"]) - float(r["gt"]))
            e0 = np.array(list(intact.values()))
            ed = np.concatenate([np.array(v) for v in drop.values()])
            delta = np.concatenate([np.array(drop[q]) - intact[q] for q in drop])
            worst = np.array([max(abs(x) for x in drop[q]) for q in drop])

            def by_scene(d):
                return {s: {q[1]: v for q, v in d.items() if q[0] == s} for s in {q[0] for q in d}}

            i_s, d_s = by_scene(intact), by_scene(drop)
            r = {
                "frames": len(intact),
                "mae_intact": float(np.abs(e0).mean()),
                "bias_intact": float(e0.mean()),
                "mae_dropped": float(np.abs(ed).mean()),
                "bias_dropped": float(ed.mean()),
                "count_change_mean": float(delta.mean()),
                "count_change_abs": float(np.abs(delta).mean()),
                "mae_worst_camera": float(worst.mean()),
            }
            for k in args.k:
                r[f"k{k}_intact"] = kshot_mae(i_s, i_s, k, rng, args.draws)
                r[f"k{k}_dropped_cal_intact"] = kshot_mae(i_s, d_s, k, rng, args.draws)
                r[f"k{k}_dropped_cal_dropped"] = kshot_mae(
                    {s: {q: np.mean(v) for q, v in d.items()} for s, d in d_s.items()}, d_s, k, rng, args.draws
                )
            res[split] = r
        summary[spec] = res
    for split in ("validate", "test"):
        print(f"\n== {split}")
        print(
            f"{'model':<28}{'MAE 5cam':>9}{'MAE 4cam':>9}{'worst':>7}{'count chg':>10}{'|chg|':>7}"
            + "".join(f"{f'k={k} 5cam':>10}{f'k={k} 4cam':>10}{f'k={k} recal':>11}" for k in args.k)
        )
        for spec, res in summary.items():
            r = res[split]
            print(
                f"{spec:<28}{r['mae_intact']:9.1f}{r['mae_dropped']:9.1f}{r['mae_worst_camera']:7.1f}"
                f"{r['count_change_mean']:+10.1f}{r['count_change_abs']:7.1f}"
                + "".join(
                    f"{r[f'k{k}_intact']:10.1f}{r[f'k{k}_dropped_cal_intact']:10.1f}"
                    f"{r[f'k{k}_dropped_cal_dropped']:11.1f}"
                    for k in args.k
                )
            )
    print(
        "\nMAE 4cam = mean over the 5 single-camera dropouts; worst = per frame the worst dropout; count chg = "
        "predicted count with a camera dropped minus with all 5. k=K 5cam: offset from K labelled intact frames, "
        "scored on intact frames; 4cam: same offset (calibrated before the failure) scored on dropout frames; "
        "recal: offset re-estimated from K labelled dropout frames."
    )
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "models": summary}, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ablation_v2:homography_256,ablation_v2:ca_sa")
    ap.add_argument("--frames_per_scene", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.analyze_only:
        run(args, torch.device("cuda"))
    analyze(args)


if __name__ == "__main__":
    main()
