"""
scripts/visualisation/visualize_paper_prediction.py

Paper-style qualitative figure for one test sample: the 5 input views, the GT density map, the
predicted density map and the detections (matched / false / missed) on the ground plane, plus a
zoom into the densest region. Uses exactly the post-processing of evaluate_paper_protocol.py.

The sample is picked to be representative, not cherry-picked:
- scene: the one whose per-scene MAE (from the eval JSON) is closest to the overall MAE
- frame: every frame of that scene is predicted once (seeded views); the one whose absolute
  error is closest to the scene's MAE is plotted

Usage (from src/):
    python scripts/visualisation/visualize_paper_prediction.py --weights runs/baseline/run_1/best.pth \
        --eval_json runs/baseline/run_1/eval_test_best.json
"""

import os
import sys
import json
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm import tqdm

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
from evaluation.evaluate_paper_protocol import load_model, predict_baseline, gt_pixels, detect, SCALE
from losses.gaussian_mse import get_map_kernel

# Colours (validated reference palette): one-hue sequential blue for magnitude,
# first three categorical slots for the detection outcome (+ distinct marker shapes).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE_RAMP = [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DENSITY_CMAP = LinearSegmentedColormap.from_list("density_blue", BLUE_RAMP)
TP_COLOR, FP_COLOR, FN_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
IMG_MEAN, IMG_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def match_assign(dets, gts, radius):
    """Same matching as evaluate_paper_protocol.match, returning the matched index pairs."""
    if len(dets) == 0 or len(gts) == 0:
        return np.empty(0, int), np.empty(0, int)
    pairs = cKDTree(dets).sparse_distance_matrix(cKDTree(gts), radius, output_type="coo_matrix")
    if pairs.nnz == 0:
        return np.empty(0, int), np.empty(0, int)
    n_det = len(dets)
    graph = coo_matrix((np.ones(pairs.nnz), (pairs.row, pairs.col + n_det)), shape=(n_det + len(gts),) * 2)
    _, labels = connected_components(graph, directed=False)
    det_idx, gt_idx = [], []
    for comp in np.unique(labels[pairs.row]):
        d_i = np.where(labels[:n_det] == comp)[0]
        g_i = np.where(labels[n_det:] == comp)[0]
        dist = np.linalg.norm(dets[d_i][:, None, :] - gts[g_i][None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(np.where(dist > radius, 1e6, dist))
        ok = dist[rows, cols] <= radius
        det_idx.extend(d_i[rows[ok]])
        gt_idx.extend(g_i[cols[ok]])
    return np.array(det_idx, int), np.array(gt_idx, int)


def load_sample(ds, idx, seed):
    random.seed(seed * 100_003 + idx)  # reproducible view draw per frame
    return ds[idx]


def batched(sample):
    """Mimic the DataLoader's batch dimension so the evaluator's helpers can be reused."""
    return {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in sample.items()}


def gt_density(gts, shape, kernel):
    point_map = torch.zeros((1, 1) + tuple(shape))
    if len(gts):
        point_map[0, 0, gts[:, 0].astype(int), gts[:, 1].astype(int)] = 1.0
    blurred = F.conv2d(point_map, kernel, padding=kernel.shape[-1] // 2)
    return (blurred / (blurred.sum() + 1e-8) * len(gts))[0, 0].numpy()  # people per pixel


def densest_window(density, size):
    box = F.avg_pool2d(torch.from_numpy(density)[None, None], size, stride=4)
    r, c = np.unravel_index(box[0, 0].argmax().item(), box.shape[-2:])
    return int(r * 4), int(c * 4)


def style_map_axis(ax, title):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c3c2b7")
        s.set_linewidth(0.6)


def plot_detections(ax, dets, gts, tp_det, tp_gt, window=None, size=2.0):
    fp_mask = np.ones(len(dets), bool)
    fp_mask[tp_det] = False
    fn_mask = np.ones(len(gts), bool)
    fn_mask[tp_gt] = False
    groups = [(dets[tp_det], TP_COLOR, "o"), (dets[fp_mask], FP_COLOR, "x"), (gts[fn_mask], FN_COLOR, "^")]
    for pts, color, marker in groups:
        if window is not None:
            (r0, c0, n) = window
            keep = (pts[:, 0] >= r0) & (pts[:, 0] < r0 + n) & (pts[:, 1] >= c0) & (pts[:, 1] < c0 + n)
            pts = pts[keep] - [r0, c0]
        ax.scatter(pts[:, 1], pts[:, 0], s=size, c=color, marker=marker, linewidths=0.6 if marker == "x" else 0)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = load_model(args.model, args.weights, device)
    with open(args.eval_json) as f:
        eval_res = json.load(f)

    scene = args.scene or min(
        eval_res["per_scene_MAE"], key=lambda s: abs(eval_res["per_scene_MAE"][s] - eval_res["MAE"])
    )
    scene_mae = eval_res["per_scene_MAE"][scene]
    ds = SynMVCrowdDataset(
        scene_dir=os.path.join(src_dir, "datasets", "SynMVCrowd", "data", args.split, scene),
        camera_ids=None,
        num_sampled_cams=5,
        load_refs=False,
    )

    # 1. pick the frame whose error is closest to the scene MAE
    if args.frame_index is None:
        errors = []
        for idx in tqdm(range(len(ds)), desc=f"Scanning {scene}", mininterval=10):
            sample = batched(load_sample(ds, idx, args.seed))
            pred = predict_baseline(model, sample, device)
            gts = gt_pixels(sample["gt_world_xy"], sample["wld_map_paras"], pred.shape)
            errors.append(abs(pred.sum().item() / SCALE - len(gts)))
        frame_index = int(np.argmin(np.abs(np.array(errors) - scene_mae)))
        print(
            f"{scene}: scanned MAE {np.mean(errors):.2f} (eval JSON: {scene_mae:.2f}); "
            f"picked index {frame_index} with error {errors[frame_index]:.2f}"
        )
    else:
        frame_index = args.frame_index

    # 2. predict + post-process the chosen sample exactly like the evaluator
    raw = load_sample(ds, frame_index, args.seed)
    sample = batched(raw)
    pred_t = predict_baseline(model, sample, device)
    wld = sample["wld_map_paras"]
    gts = gt_pixels(sample["gt_world_xy"], wld, pred_t.shape)
    dets = detect(pred_t, args.cls_thres, args.nms_radius_px)
    tp_det, tp_gt = match_assign(dets, gts, args.match_radius_m * wld[0, 0].item())
    pred = pred_t.cpu().numpy() / SCALE  # people per pixel
    kernel = get_map_kernel(sigma=5, kernel_size=20)
    gt = gt_density(gts, pred.shape, kernel)
    n_tp, n_fp, n_fn = len(tp_det), len(dets) - len(tp_det), len(gts) - len(tp_gt)
    vmax = max(gt.max(), pred.max())

    # 3. figure
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
            "text.color": INK,
            "axes.labelcolor": INK_2,
        }
    )
    fig = plt.figure(figsize=(15, 13.2), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 15, height_ratios=[0.42, 1.0, 1.0], hspace=0.16, wspace=0.35)

    images = raw["images"].permute(0, 2, 3, 1).numpy() * IMG_STD + IMG_MEAN
    for v in range(len(images)):
        ax = fig.add_subplot(gs[0, 3 * v : 3 * v + 3])
        ax.imshow(np.clip(images[v], 0, 1))
        style_map_axis(ax, f"View {v + 1}")

    zoom = args.zoom
    r0, c0 = densest_window(gt, zoom)
    titles = [
        f"Ground truth  ·  {len(gts)} people",
        f"Prediction  ·  {pred.sum():.1f} people (error {pred.sum() - len(gts):+.1f})",
        f"Detections  ·  {n_tp} matched, {n_fp} false, {n_fn} missed",
    ]
    for row, window in ((1, None), (2, (r0, c0, zoom))):
        for col, title in enumerate(titles):
            ax = fig.add_subplot(gs[row, 5 * col : 5 * col + 5])
            ax.set_facecolor(SURFACE)
            crop = (lambda m: m[r0 : r0 + zoom, c0 : c0 + zoom]) if window else (lambda m: m)
            if col < 2:
                im = ax.imshow(
                    crop(gt if col == 0 else pred), cmap=DENSITY_CMAP, vmin=0, vmax=vmax, interpolation="nearest"
                )
            else:
                shape = (zoom, zoom) if window else pred.shape
                ax.imshow(np.zeros(shape), cmap=DENSITY_CMAP, vmin=0, vmax=1)  # blank surface, same extent
                plot_detections(ax, dets, gts, tp_det, tp_gt, window, size=14 if window else 2.5)
            if window is None:
                ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), zoom, zoom, fill=False, ec=INK, lw=0.9, ls="--"))
                style_map_axis(ax, title)
            else:
                style_map_axis(
                    ax, f"Zoom (densest {zoom}×{zoom} px = {zoom * 0.2:.0f}×{zoom * 0.2:.0f} m)" if col == 0 else ""
                )
    cax = fig.add_axes([0.13, 0.055, 0.40, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("density (people per 0.2 m cell)", color=INK_2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    legend = [
        Line2D([], [], ls="", marker="o", color=TP_COLOR, label="matched detection (TP)"),
        Line2D([], [], ls="", marker="x", color=FP_COLOR, label="false detection (FP)"),
        Line2D([], [], ls="", marker="^", color=FN_COLOR, label="missed person (FN)"),
    ]
    fig.legend(handles=legend, loc="lower right", bbox_to_anchor=(0.9, 0.035), ncol=3, frameon=False, fontsize=9)
    snapshot = ds.snapshots[frame_index]
    fig.suptitle(
        f"{args.label or args.model.capitalize()} (epoch {epoch})  ·{args.split}/{scene}, snapshot {snapshot}  ·  "
        f"scene MAE {scene_mae:.1f}, overall MAE {eval_res['MAE']:.1f}",
        x=0.125,
        ha="left",
        y=0.93,
        fontsize=12,
        color=INK,
    )

    out = args.out or os.path.join(src_dir, "figs", f"{args.model}_prediction_{scene}_snap{snapshot}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"GT {len(gts)} | pred {pred.sum():.1f} | TP {n_tp} FP {n_fp} FN {n_fn} | zoom at ({r0}, {c0})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper-style prediction figure")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--eval_json", type=str, required=True, help="evaluate_paper_protocol.py output")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--scene", type=str, default=None, help="Default: scene with MAE closest to the overall MAE")
    parser.add_argument("--frame_index", type=int, default=None, help="Default: frame with error closest to scene MAE")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zoom", type=int, default=80, help="Zoom window size in BEV pixels")
    parser.add_argument("--cls_thres", type=float, default=0.2)
    parser.add_argument("--nms_radius_px", type=float, default=2.5)
    parser.add_argument("--match_radius_m", type=float, default=0.5)
    parser.add_argument(
        "--label", type=str, default=None, help='Model name in the figure title, e.g. "Baseline, frozen backbone"'
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out", type=str, default=None)
    main(parser.parse_args())
