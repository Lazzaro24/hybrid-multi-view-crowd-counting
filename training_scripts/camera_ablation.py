"""
camera_ablation.py

Camera ablation (RQ3): how does the count depend on the number of views N, and does that differ between the
baseline's homography fusion and the proposed attention fusion? Plan and results: doc/camera_ablation.md.

Built on neck_ablation.py, which stays unchanged: same frozen backbone, BaselineDecoder head, GaussianMSE loss,
schedule, EMA, precise-BN, whole-scene tile evaluation and seeds. Only the view count varies. neck_ablation reads its
module-level NUM_CAMS whenever a model or a split is built (the cross-attention's camera slots and each scene's
K-means camera set from optimal_cameras_filtered.json, which has sets for k = 3..10), so this script sets it per N.

Two measurements, on identical test frames for every N (fixed frame seed; the K-means set of size N):
  train     a fusion TRAINED with N views (short budget, as the neck ablation: 30 epochs x 280 iterations)
            -> what N views give when the model is built for them
  transfer  the N=5 model of `train`, final EMA weights, TESTED with N views, no retraining
            raw   as trained: the view softmax (homography) gives each view 1/N, ca_sa sums its cameras
            vnorm magnitude rescaled to the trained 5 views (utils/camera_slots.renormalise_views)
            ca_sa's per-camera slots are reused cyclically (utils/camera_slots.set_camera_count)
            -> how robust a 5-view model is to more / fewer views; 5->5 raw reproduces train's 5-view test_final

Usage (from src):
    uv run python training_scripts/camera_ablation.py --mode all --run_name v1          # train 3,5,7,10, then transfer
    uv run python training_scripts/camera_ablation.py --mode train --run_name v1 --views 3,5
    uv run python training_scripts/camera_ablation.py --mode transfer --run_name v1     # needs <root>/5v/<variant>
    uv run python training_scripts/camera_ablation.py --mode summarize --run_name v1
    uv run python training_scripts/camera_ablation.py --mode all --run_name v1_seed2 --seed 2   # second seed
Output: runs/camera_ablation/<run_name>/<N>v/<variant>/{best.pth,last.pth,log.csv,result.json} (neck_ablation
layout; finished runs are skipped, interrupted ones resume), transfer/<variant>_5to<N>v.json, summary.json.
For the long runs use remote_scripts/run_camera_ablation_train.ps1: one process per (N, variant) with restarts.
"""

import os
import sys
import json
import random
import argparse

import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)
sys.path.append(current_dir)

import neck_ablation as na
from utils.camera_slots import set_camera_count, renormalise_views
from utils.exclusions import load_excluded_scenes

KMEANS_VIEWS = range(3, 11)  # optimal_cameras_filtered.json has K-means camera sets for k = 3..10

# neck_ablation.py's defaults (the ablation_v2 recipe) for the arguments this script does not expose
NECK_DEFAULTS = dict(
    optimizer="adam",
    lr=1e-4,
    weight_decay=1e-4,
    warmup_ratio=0.05,
    lr_schedule="cosine",
    loss="mse",
    init_from=None,
    freeze_fusion=False,
    max_grad_norm=5.0,
    patience=0,
    ema_decay=0.999,
    no_ema=False,
    no_test=False,
    bev_layers=3,
    no_grad_checkpoint=False,
    max_oom_per_epoch=10,
)


def set_views(n):
    na.NUM_CAMS = n


def views_dir(args, n):
    return os.path.join(args.root, f"{n}v")


def load_data(n, cams_cfg, excluded, args):
    set_views(n)
    data = tuple(
        na.load_split(s, cams_cfg, excluded, args.refs_file, args.z_anchors) for s in ("train", "validate", "test")
    )
    print(f"[{n} views] scenes: {len(data[0])} train / {len(data[1])} val / {len(data[2])} test")
    return data


def train(args, cams_cfg, excluded, device):
    for n in args.views:
        data = load_data(n, cams_cfg, excluded, args)
        args.out_root = views_dir(args, n)  # run_variant writes <out_root>/<variant>/
        os.makedirs(args.out_root, exist_ok=True)
        for v in args.variants:
            set_views(n)
            na.run_variant(v, args, data, device)
        na.summarize(args.out_root)
        del data


def transfer(args, cams_cfg, excluded, device):
    """The source_views model's final EMA weights, tested with N views (raw and vnorm)."""
    out_dir = os.path.join(args.root, "transfer")
    os.makedirs(out_dir, exist_ok=True)
    for n in args.views:
        todo = [
            v
            for v in args.variants
            if args.overwrite or not os.path.isfile(os.path.join(out_dir, f"{v}_{args.source_views}to{n}v.json"))
        ]
        if not todo:
            continue
        data = load_data(n, cams_cfg, excluded, args)
        train_scenes, _, test_scenes = data
        # the same loaders (seeds) as run_variant's test, so the frames match the trained-at-N results

        def bn_loader():
            return na.make_loader(train_scenes, 1, args.val_seed + 7, args.num_workers)

        def test_loader():
            return na.make_loader(test_scenes, args.test_frames_per_scene, args.val_seed + 13, args.num_workers)

        for v in todo:
            src = os.path.join(views_dir(args, args.source_views), v)
            if not os.path.isfile(os.path.join(src, "result.json")):
                print(f"[skip] {v}: no finished {args.source_views}-view run in {src} (run --mode train first)")
                continue
            ck = torch.load(os.path.join(src, "last.pth"), map_location="cpu", weights_only=False)
            state = ck["ema_state_dict"]["model"] if ck.get("ema_state_dict") else ck["model_state_dict"]
            set_views(args.source_views)
            model = na.AblationModel(v, args).to(device)
            model.load_state_dict(state)
            resized = set_camera_count(model, n)
            res = {
                "variant": v,
                "trained_views": args.source_views,
                "views": n,
                "weights": os.path.join(src, "last.pth"),
                "epoch": ck["epoch"],
                "ema": bool(ck.get("ema_state_dict")),
                "resized": resized,
                "test_frames_per_scene": args.test_frames_per_scene,
            }
            del ck, state
            for mode in ["raw"] if n == args.source_views else ["raw", "vnorm"]:
                if mode == "vnorm":
                    res["scales"] = renormalise_views(model, n, args.source_views)
                na.precise_bn(model, bn_loader(), device)
                res[mode] = na.evaluate(
                    model, test_scenes, test_loader(), device, f"[{v}] {args.source_views} -> {n} views ({mode})"
                )
                print(
                    f">>> [{v}] trained {args.source_views} views, tested {n} ({mode}): MAE {res[mode]['mae']:.2f} "
                    f"bias {res[mode]['bias']:+.2f}"
                )
            with open(os.path.join(out_dir, f"{v}_{args.source_views}to{n}v.json"), "w") as f:
                json.dump(res, f, indent=2)
            del model
            torch.cuda.empty_cache()
        del data


def summarize(args):
    rows = []
    for v in args.variants:
        for n in KMEANS_VIEWS:
            row = {"variant": v, "views": n}
            p = os.path.join(views_dir(args, n), v, "result.json")
            if os.path.isfile(p):
                with open(p) as f:
                    r = json.load(f)
                row["trained"] = {
                    "test_final": r["test_final"],
                    "test_best": r["test"],
                    "val": r["val"],
                    "best_epoch": r["best_epoch"],
                }
            p = os.path.join(args.root, "transfer", f"{v}_{args.source_views}to{n}v.json")
            if os.path.isfile(p):
                with open(p) as f:
                    r = json.load(f)
                row["transfer"] = {m: r[m] for m in ("raw", "vnorm") if m in r}
            if len(row) > 2:
                rows.append(row)
    if not rows:
        print(f"[summary] nothing finished in {args.root}")
        return
    nan = {"mae": float("nan"), "bias": float("nan")}
    print(
        f"\n{'=' * 112}\nCAMERA ABLATION SUMMARY  ({args.root})  test MAE (bias), K-means views, "
        f"{args.test_frames_per_scene} frames/scene\n{'=' * 112}"
    )
    print(
        f"{'variant':<16}{'views':>6} | {'trained N: final':>20}{'best-val':>18} | "
        f"{f'{args.source_views}->N raw':>20}{f'{args.source_views}->N vnorm':>20}"
    )
    for r in rows:
        t, tr = r.get("trained", {}), r.get("transfer", {})

        def cell(m):
            return f"{(m or nan)['mae']:>8.2f} ({(m or nan)['bias']:+7.1f})"

        print(
            f"{r['variant']:<16}{r['views']:>6} | {cell(t.get('test_final')):>20}{cell(t.get('test_best')):>18} | "
            f"{cell(tr.get('raw')):>20}{cell(tr.get('vnorm')):>20}"
        )
    with open(os.path.join(args.root, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Camera ablation: neck_ablation recipe, only the number of views varies.")
    ap.add_argument(
        "--mode",
        choices=["train", "transfer", "all", "summarize"],
        default="all",
        help="all = train, then transfer, then summarize",
    )
    ap.add_argument("--run_name", required=True, help="folder under runs/camera_ablation/")
    ap.add_argument("--views", default="3,5,7,10", help=f"comma-separated view counts, each in {list(KMEANS_VIEWS)}")
    ap.add_argument("--variants", default="homography_256,ca_sa", help=f"comma-separated subset of {na.VARIANTS}")
    ap.add_argument("--source_views", type=int, default=5, help="transfer: the trained view count that is re-tested")
    ap.add_argument("--overwrite", action="store_true", help="retrain / re-test runs that already have results")
    ap.add_argument(
        "--backbone", default=os.path.join(src_dir, "runs", "train_2d", "new_best", "resnet18_2d_pretrained_ep4.pth")
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--frames_per_scene", type=int, default=10, help="training frames per scene per epoch")
    ap.add_argument("--val_frames_per_scene", type=int, default=20)
    ap.add_argument(
        "--val_every",
        type=int,
        default=1,
        help="validate every k epochs (and the last). Validation does not touch the training weights or "
        "data, so the final weights (test_final) are the same for any k; only the best-val pick "
        "(test) changes. 5 saves ~40%% of the time",
    )
    ap.add_argument("--test_frames_per_scene", type=int, default=20, help="0 = all test frames")
    ap.add_argument("--refs_file", default="master_refs.pt")
    ap.add_argument("--z_anchors", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1, help="neck ablation (ablation_v2) used 1")
    ap.add_argument("--val_seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_root", default=None, help="output folder (default: runs/camera_ablation/<run_name>)")
    args = ap.parse_args()
    for k, val in NECK_DEFAULTS.items():
        setattr(args, k, val)
    args.root = (
        os.path.abspath(args.out_root)
        if args.out_root
        else os.path.join(src_dir, "runs", "camera_ablation", args.run_name)
    )
    args.views = [int(x) for x in args.views.split(",") if x.strip()]
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [n for n in args.views if n not in KMEANS_VIEWS] + [v for v in args.variants if v not in na.VARIANTS]
    if bad or args.source_views not in KMEANS_VIEWS:
        raise SystemExit(
            f"invalid views / variants {bad or [args.source_views]}: views in {list(KMEANS_VIEWS)}, "
            f"variants in {na.VARIANTS}"
        )
    os.makedirs(args.root, exist_ok=True)
    if args.mode == "summarize":
        summarize(args)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"--- camera ablation '{args.run_name}' on {device}: {args.variants} x {args.views} views ---\n{vars(args)}")
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    excluded = load_excluded_scenes()
    if args.mode in ("train", "all"):
        train(args, cams_cfg, excluded, device)
    if args.mode in ("transfer", "all"):
        transfer(args, cams_cfg, excluded, device)
    summarize(args)


if __name__ == "__main__":
    main()
