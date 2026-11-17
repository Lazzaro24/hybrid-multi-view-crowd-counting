"""
scripts/visualisation/visualize_learned_routing.py

Learned routing of Stage 3 checkpoints side by side. For each scene (full scene BEV, one frame, the
training/validation camera set) and each checkpoint, one row:

  GT density | dense expert (unmasked) | sparse expert (unmasked) | router mask p(dense) | fused = p * dense + (1 - p) * sparse

The fused panel carries the p > 0.5 outline (cells the hard router sends to the dense expert). Panel titles give the
counts: GT, dense alone, sparse alone, fused soft and fused hard. Density panels share one gamma-0.5 colour scale per
scene so low-level mass stays visible; the mask panel is on [0, 1].

Usage (from src/):
    python scripts/visualisation/visualize_learned_routing.py --scenes scene_67,scene_18 \
        --weights multiscene_best_stage_3_s8_dout0.7_cw0.01_f10_unmasked_joint_genloss_perf_c10.pth,multiscene_best_stage_3_s8_dout0.7_cw0.01_f10_unmasked_joint_genloss_ep25_perf_c10.pth \
        --labels "Router on joint ep 13,Router on joint ep 25"
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LinearSegmentedColormap

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
from evaluation.evaluate_paper_protocol import load_model, gt_pixels, SCALE
from losses.gaussian_mse import get_map_kernel
import training_scripts.train_proposed_multiscene as T
from visualize_paper_prediction import SURFACE, INK, INK_2, MUTED, DENSITY_CMAP, batched, gt_density

MASK_CMAP = LinearSegmentedColormap.from_list("mask", [SURFACE, "#9ec5f4", "#256abf", "#0d366b"])
OUTLINE = "#eb6834"  # categorical slot 2: hard-routing outline on the blue density ramp


@torch.no_grad()
def predict_routing(model, sample, device):
    """Full-scene unmasked dense / sparse outputs (people per pixel) and the router mask, with the same 200x200
    tiling as evaluate_paper_protocol.predict_proposed."""
    images, refs = sample["images"].to(device), sample["ref_points_full"].to(device)
    wld = sample["wld_map_paras"]
    h_full, w_full = int(wld[0, 3].item()), int(wld[0, 4].item())
    th, tw = model.bev_shape

    def starts(full, tile):
        return sorted(set(list(range(0, full - tile + 1, tile)) + [full - tile]))

    parts = {k: torch.zeros((h_full, w_full), device=device) for k in ("dense", "sparse", "mask")}
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        B, N = images.shape[:2]
        feats = model.backbone(images.flatten(0, 1))
        feats = feats.view(B, N, *feats.shape[1:])
        for h in starts(h_full, th):
            for w in starts(w_full, tw):
                bev = model.neck(feats, refs[:, :, h : h + th, w : w + tw])
                parts["dense"][h : h + th, w : w + tw] = F.relu(model.head.dense_expert(bev))[0, 0].float()
                parts["sparse"][h : h + th, w : w + tw] = F.relu(model.head.sparse_expert(bev))[0, 0].float()
                parts["mask"][h : h + th, w : w + tw] = model.head.gating_net(bev)[0, 0].float()
    to_people = model.to_scale / SCALE
    out = {k: v.cpu().numpy() for k, v in parts.items()}
    out["dense"] *= to_people
    out["sparse"] *= to_people
    m = out["mask"]
    out["fused"] = m * out["dense"] + (1 - m) * out["sparse"]
    out["fused_hard"] = np.where(m > 0.5, out["dense"], out["sparse"])
    return out


def scene_figure(scene, frame, gt, rows, out_path, dpi):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"]})
    h, w = gt.shape
    fig, axes = plt.subplots(
        len(rows), 5, figsize=(24, 4.8 * len(rows) * h / w + 1.3), squeeze=False, facecolor=SURFACE
    )
    vmax = (
        max([np.percentile(gt, 99.8)] + [np.percentile(r[k], 99.8) for r in rows for k in ("dense", "sparse", "fused")])
        + 1e-6
    )

    def norm():
        return PowerNorm(0.5, vmin=0, vmax=vmax)

    n_gt = gt.sum()
    for i, r in enumerate(rows):
        hard = r["mask"] > 0.5
        panels = [
            (gt, norm(), DENSITY_CMAP, f"GT · {n_gt:.0f} people"),
            (r["dense"], norm(), DENSITY_CMAP, f"Dense expert · alone {r['dense'].sum():.0f}"),
            (r["sparse"], norm(), DENSITY_CMAP, f"Sparse expert · alone {r['sparse'].sum():.0f}"),
            (r["mask"], None, MASK_CMAP, f"Router p(dense) · {100 * hard.mean():.1f}% of scene > 0.5"),
            (
                r["fused"],
                norm(),
                DENSITY_CMAP,
                f"Fused · soft {r['fused'].sum():.0f} · hard {r['fused_hard'].sum():.0f}",
            ),
        ]
        for j, (img, nrm, cmap, title) in enumerate(panels):
            ax = axes[i, j]
            if nrm is None:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(img, cmap=cmap, norm=nrm, interpolation="nearest")
            if j == 4 and hard.any() and not hard.all():
                ax.contour(hard.astype(float), levels=[0.5], colors=[OUTLINE], linewidths=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(MUTED)
                sp.set_linewidth(0.5)
            ax.set_title(title, fontsize=9, color=INK_2, loc="left")
            if j == 0:
                err_s, err_h = r["fused"].sum() - n_gt, r["fused_hard"].sum() - n_gt
                ax.set_ylabel(
                    f"{r['label']}\nfused error soft {err_s:+.0f} · hard {err_h:+.0f}", fontsize=10, color=INK
                )
    fig.suptitle(
        f"{scene} · frame {frame} — learned performance routing; orange outline in the fused panel = "
        f"router p(dense) > 0.5 (hard choice of the dense expert)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.003,
        "People per pixel, gamma 0.5 colour scale shared by all density panels of the scene; "
        "router mask on [0, 1] (dark = dense expert). Counts in people.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = [w.strip() for w in args.weights.split(",")]
    labels = [s.strip() for s in args.labels.split(",")] if args.labels else weights
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams = json.load(f)
    kernel = get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE)
    models = [(label, load_model("proposed", wpath, device)[0]) for wpath, label in zip(weights, labels)]
    os.makedirs(args.out_dir, exist_ok=True)
    report = {}
    for scene in [s.strip() for s in args.scenes.split(",")]:
        ds = SynMVCrowdDataset(
            scene_dir=os.path.join(src_dir, "datasets", "SynMVCrowd", "data", args.split, scene),
            camera_ids=cams[scene][str(5)],
        )
        frame = len(ds) // 2 if args.frame is None else args.frame
        sample = batched(ds[frame])
        wld = sample["wld_map_paras"]
        shape = (int(wld[0, 3].item()), int(wld[0, 4].item()))
        gt = gt_density(gt_pixels(sample["gt_world_xy"], wld, shape), shape, kernel)
        rows = []
        for label, model in models:
            pred = predict_routing(model, sample, device)
            rows.append({"label": label, **pred})
            stats = {
                "gt": float(gt.sum()),
                "dense_alone": float(pred["dense"].sum()),
                "sparse_alone": float(pred["sparse"].sum()),
                "fused_soft": float(pred["fused"].sum()),
                "fused_hard": float(pred["fused_hard"].sum()),
                "share_dense": float((pred["mask"] > 0.5).mean()),
            }
            report.setdefault(scene, {"frame": frame})[label] = stats
            print(
                f"{scene} | {label:>24} | GT {stats['gt']:6.1f} | dense {stats['dense_alone']:6.1f} | sparse "
                f"{stats['sparse_alone']:6.1f} | fused soft {stats['fused_soft']:6.1f} hard {stats['fused_hard']:6.1f} | "
                f"{100 * stats['share_dense']:5.1f}% routed dense"
            )
        scene_figure(scene, frame, gt, rows, os.path.join(args.out_dir, f"routing_{scene}.png"), args.dpi)
    with open(os.path.join(args.out_dir, "routing.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="comma-separated Stage 3+ checkpoints")
    ap.add_argument("--labels", default=None, help="comma-separated row labels, same order as --weights")
    ap.add_argument("--scenes", default="scene_67,scene_18")
    ap.add_argument("--split", default="validate")
    ap.add_argument("--frame", type=int, default=None, help="default: middle frame of each scene")
    ap.add_argument("--out_dir", default=os.path.join(src_dir, "figs", "learned_routing"))
    ap.add_argument("--dpi", type=int, default=120)
    main(ap.parse_args())
