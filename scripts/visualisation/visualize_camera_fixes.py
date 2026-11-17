"""
visualize_camera_fixes.py

Thesis / slide figures for the two reference-point validity fixes (doc/camera_bug.md, doc/camera_off_axis.md),
computed with the production `utils.reference_points.compute_reference_points` on the debug scene of both docs
(train/scene_2, cameras 18, 26, 33, 39, 47, snapshot of dataset index 0).

  fig_camera_fixes_overview.png  3 x 5 grid, BEV coverage per camera (number of valid height anchors per cell):
                                 row 1 original check (z_cam > 0 only), row 2 + per-camera z-sign,
                                 row 3 + off-axis rejection (= production code).
  fig_camera_zsign.png           the z-sign bug alone: rows 1-2 of the overview plus, per camera, the share of its
                                 annotated visible people with negative z_cam.
  fig_camera_offaxis.png         the off-axis fold-back on camera 26: coverage split into main wedge and
                                 disconnected band, off-axis ray magnitude r with the threshold, r histogram of both
                                 regions, and where the band's rays land in the camera image.

Usage (from src):
    uv run python scripts/visualisation/visualize_camera_fixes.py
    uv run python scripts/visualisation/visualize_camera_fixes.py --scene scene_2 --cams 18,26,33,39,47 --detail_cam 26
"""

import os
import sys
import json
import argparse

import numpy as np
import cv2
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import ndimage

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, src_dir)

from utils.reference_points import compute_reference_points, DEFAULT_OFFAXIS_THRESHOLD
from utils.check_offaxis_scene import load_scene_locations
from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset

NO_OFFAXIS = 1e12  # threshold that disables the off-axis check
IMG_H, IMG_W = 720, 1280
COV_CMAP = ListedColormap(["#1a1a1a", "#3b4cc0", "#7b9ff9", "#c0d4f5", "#f7f7f7"])
COV_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], COV_CMAP.N)


def valid_anchors(refs):
    """refs [C, H, W, Z, 2] normalised image coords (-10 = invalid) -> bool [C, H, W, Z]: in front and in frame."""
    x, y = refs[..., 0], refs[..., 1]
    return ((x >= -1) & (x <= 1) & (y >= -1) & (y <= 1)).numpy()


def offaxis_r(wld, cam_paras):
    """tan(angle from the optical axis) per grid cell and anchor, same math as compute_reference_points."""
    r, a, b, h_actual, w_actual, d_mean, w_min, h_min, w_max, h_max, d_delta = [float(v) for v in wld]
    xs = np.arange(int(w_actual)) / r + w_min - a
    ys = np.arange(int(h_actual)) / r + h_min - b
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    zs = np.linspace(-d_delta, 1.8 + d_delta, 4) + d_mean
    pts = np.stack([np.stack([gx, gy, np.full_like(gx, z)], -1) for z in zs], -2).reshape(-1, 3)
    rvec = cam_paras[9:12].numpy().astype(np.float64)
    tvec = cam_paras[12:15].numpy().astype(np.float64)
    R, _ = cv2.Rodrigues(rvec)
    pc = pts @ R.T + tvec
    Z = np.where(np.abs(pc[:, 2]) < 1e-6, 1e-6, pc[:, 2])
    rr = np.sqrt((pc[:, 0] / Z) ** 2 + (pc[:, 1] / Z) ** 2)
    return rr.reshape(int(h_actual), int(w_actual), 4)


def world_to_grid(xy, wld):
    r, a, b, _, _, _, w_min, h_min = [float(v) for v in wld[:8]]
    return (xy[:, 0] - w_min + a) * r, (xy[:, 1] - h_min + b) * r  # col, row


def negative_zcam_share(scene_dir, snapshot_id, cam_id):
    """Share of the camera's annotated visible people with negative camera-space depth (raw extrinsics), their
    number, and their world xy."""
    with open(os.path.join(scene_dir, str(snapshot_id), "jsons", f"{cam_id}.json")) as f:
        d = json.load(f)
    world = np.array([p["world"] for p in d["image_info"] if p.get("pixel") is not None], np.float64)
    if len(world) == 0:
        return float("nan"), 0, np.zeros((0, 2))
    R, _ = cv2.Rodrigues(np.array(d["rvec"], np.float64))
    z = (world @ R.T + np.array(d["tvec"], np.float64))[:, 2]
    return float((z < 0).mean()), len(world), world[:, :2]


def show_coverage(ax, n_valid, gt_cr, title):
    ax.imshow(n_valid, cmap=COV_CMAP, norm=COV_NORM, origin="lower", interpolation="nearest")
    ax.scatter(gt_cr[0], gt_cr[1], s=1.2, c="#ffb000", linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def denorm(img):
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene_2")
    ap.add_argument("--cams", default="18,26,33,39,47")
    ap.add_argument("--detail_cam", type=int, default=26)
    ap.add_argument("--ref_cam", type=int, default=18, help="unaffected camera shown as the expected coverage")
    ap.add_argument("--out_dir", default=os.path.join(src_dir, "figs", "camera_fixes"))
    args = ap.parse_args()
    cams = [int(c) for c in args.cams.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    root = os.path.join(src_dir, "datasets", "SynMVCrowd")
    _, scene_dir = load_scene_locations(root)[args.scene]
    scene_dir = scene_dir if os.path.isabs(scene_dir) else os.path.join(src_dir, scene_dir)
    with open(os.path.join(src_dir, "utils", "z_sign_corrections_full.json")) as f:
        zs_table = json.load(f).get(args.scene, {})
    z_sign = [float(zs_table.get(str(c), 1.0)) for c in cams]

    ds = SynMVCrowdDataset(scene_dir=scene_dir, camera_ids=cams, is_caching=True)
    s = ds[0]
    wld, cp = s["wld_map_paras"], s["camera_paras"]
    gt_cr = world_to_grid(s["gt_world_xy"].numpy(), wld)

    settings = [
        ("Original check (z_cam > 0)", [1.0] * len(cams), NO_OFFAXIS),
        ("+ per-camera z-sign", z_sign, NO_OFFAXIS),
        (f"+ off-axis rejection (r > {DEFAULT_OFFAXIS_THRESHOLD:g})", z_sign, DEFAULT_OFFAXIS_THRESHOLD),
    ]
    valid = {}
    for name, zsg, thr in settings:
        valid[name] = valid_anchors(compute_reference_points(wld, cp, z_sign=zsg, offaxis_threshold=thr))
    neg = [negative_zcam_share(scene_dir, s["snapshot_id"], c) for c in cams]

    # ---------------- overview: 3 x 5 ----------------
    def coverage_grid(rows, fname, suptitle, extra_title=None):
        fig, axes = plt.subplots(len(rows), len(cams), figsize=(3.0 * len(cams), 3.0 * len(rows) + 0.8), squeeze=False)
        for ri, name in enumerate(rows):
            for ci, cam in enumerate(cams):
                n_valid = valid[name][ci].sum(-1)
                cov = (n_valid > 0).mean() * 100
                t = f"cam {cam}: {cov:.1f} % covered"
                extra = extra_title(ri, ci) if extra_title else None
                if extra:
                    t += "\n" + extra
                show_coverage(axes[ri, ci], n_valid, gt_cr, t)
            axes[ri, 0].set_ylabel(name, fontsize=10)
        handles = [plt.Rectangle((0, 0), 1, 1, color=COV_CMAP(i)) for i in range(5)]
        handles.append(plt.Line2D([], [], ls="", marker="o", ms=4, color="#ffb000"))
        fig.legend(
            handles,
            ["0 anchors valid", "1", "2", "3", "4 anchors valid", "annotated people"],
            loc="lower center",
            ncol=6,
            fontsize=8,
            frameon=False,
        )
        fig.suptitle(suptitle, fontsize=11)
        fig.subplots_adjust(left=0.04, right=0.99, top=0.9, bottom=0.07, wspace=0.08, hspace=0.32)
        fig.savefig(os.path.join(args.out_dir, fname), dpi=200)
        plt.close(fig)

    names = [n for n, _, _ in settings]
    h_g, w_g = valid[names[0]].shape[1:3]
    lost = []
    for ci in range(len(cams)):
        col, row = world_to_grid(neg[ci][2], wld)
        col = np.clip(np.round(col).astype(int), 0, w_g - 1)
        row = np.clip(np.round(row).astype(int), 0, h_g - 1)
        before = valid[names[1]][ci].any(-1)[row, col]
        after = valid[names[2]][ci].any(-1)[row, col]
        lost.append(int((before & ~after).sum()))

    def overview_extra(ri, ci):
        if ri == 2 and lost[ci]:
            return f"removes {lost[ci]} of its {neg[ci][1]} visible people"
        return None

    coverage_grid(
        names,
        "fig_camera_fixes_overview.png",
        f"{args.scene}: BEV cells each camera can sample (valid height anchors per cell)",
        extra_title=overview_extra,
    )
    coverage_grid(
        names[:2],
        "fig_camera_zsign.png",
        f"{args.scene}: z-sign convention fix",
        extra_title=lambda ri, ci: (
            f"z_cam < 0: {neg[ci][0] * 100:.0f} % of {neg[ci][1]} people → z-sign {z_sign[ci]:+.0f}"
        )
        if ri == 0
        else None,
    )

    # ---------------- thesis example: one unaffected camera, then the affected camera fix by fix ----------------
    ci_ref, ci_det = cams.index(args.ref_cam), cams.index(args.detail_cam)
    panels = [
        (valid[names[2]][ci_ref], f"(a) camera {args.ref_cam}: expected"),
        (valid[names[0]][ci_det], f"(b) camera {args.detail_cam}: original test"),
        (valid[names[1]][ci_det], "(c) + per-camera depth sign"),
        (valid[names[2]][ci_det], "(d) + off-axis rejection"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.55))
    for ax, (v, t) in zip(axes, panels):
        show_coverage(ax, v.sum(-1), gt_cr, t)
        ax.collections[-1].set_sizes([0.5])
        ax.title.set_fontsize(8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COV_CMAP(i), ec="#888888", lw=0.3) for i in range(5)]
    handles.append(plt.Line2D([], [], ls="", marker="o", ms=3, color="#ffb000"))
    fig.legend(
        handles,
        ["0", "1", "2", "3", "4 valid height anchors", "annotated people"],
        loc="lower center",
        ncol=6,
        fontsize=7.5,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.13, wspace=0.05)
    fig.savefig(os.path.join(args.out_dir, "fig_camera_fix_example.png"), dpi=300)
    plt.close(fig)

    # ---------------- off-axis detail ----------------
    ci = cams.index(args.detail_cam)
    v_before = valid[names[1]][ci]  # z-sign fixed, no off-axis check
    v_after = valid[names[2]][ci]
    rmap = offaxis_r(wld, cp[ci])
    cov = v_before.any(-1)
    labels, n = ndimage.label(cov, structure=np.ones((3, 3)))
    sizes = ndimage.sum(cov, labels, range(1, n + 1))
    main_id = int(np.argmax(sizes)) + 1
    main_mask = labels == main_id
    band_mask = cov & ~main_mask
    r_main = rmap[v_before & main_mask[..., None]]
    r_band = rmap[v_before & band_mask[..., None]]
    thr = DEFAULT_OFFAXIS_THRESHOLD

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), gridspec_kw={"width_ratios": [1, 1.15, 1, 1.45]})
    region = np.zeros(cov.shape, np.int8)
    region[main_mask] = 1
    region[band_mask] = 2
    axes[0].imshow(
        region, cmap=ListedColormap(["#1a1a1a", "#4c78a8", "#e45756"]), origin="lower", interpolation="nearest"
    )
    axes[0].set_title(
        f"(a) cam {args.detail_cam} after the z-sign fix\nmain wedge (blue) + disconnected band (red)", fontsize=9
    )

    r_disp = np.where(v_before, rmap, np.nan)
    with np.errstate(invalid="ignore"):
        r_disp = np.nanmean(r_disp, -1)
    axes[1].set_facecolor("#1a1a1a")
    im = axes[1].imshow(r_disp, cmap="viridis", origin="lower", interpolation="nearest", vmin=0, vmax=5.5)
    axes[1].contour(np.nan_to_num(r_disp, nan=0.0) > thr, levels=[0.5], colors="white", linewidths=0.8)
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="r = tan(angle from optical axis)")
    axes[1].set_title(
        f"(b) off-axis magnitude r of covered cells\nwhite outline: r > {thr:g} (≈ {np.degrees(np.arctan(thr)):.0f}°)",
        fontsize=9,
    )

    bins = np.linspace(0, 5.5, 56)
    axes[2].hist(r_main, bins=bins, color="#4c78a8", alpha=0.85, label=f"main wedge (max {r_main.max():.2f})")
    if r_band.size:
        axes[2].hist(r_band, bins=bins, color="#e45756", alpha=0.85, label=f"band (min {r_band.min():.2f})")
    axes[2].axvline(thr, color="k", ls="--", lw=1)
    axes[2].text(thr + 0.05, axes[2].get_ylim()[1] * 0.9, f"threshold {thr:g}", fontsize=8)
    axes[2].set_xlabel("r = tan(angle from optical axis)")
    axes[2].set_ylabel("valid anchor points")
    axes[2].legend(fontsize=8)
    axes[2].set_title("(c) the two regions do not overlap in r", fontsize=9)

    refs = compute_reference_points(wld, cp[ci : ci + 1], z_sign=[z_sign[ci]], offaxis_threshold=NO_OFFAXIS)[0]
    uv = refs.numpy()
    img = denorm(s["images"][ci])
    axes[3].imshow(img)
    rng = np.random.default_rng(0)
    for mask, col, lab in [(main_mask, "#4c78a8", "main wedge"), (band_mask, "#e45756", "band (fold-back)")]:
        sel = v_before & mask[..., None]
        p = uv[sel]
        if len(p) > 4000:
            p = p[rng.choice(len(p), 4000, replace=False)]
        axes[3].scatter(
            (p[:, 0] + 1) / 2 * IMG_W, (p[:, 1] + 1) / 2 * IMG_H, s=1.5, c=col, alpha=0.6, linewidths=0, label=lab
        )
    axes[3].set_xlim(0, IMG_W)
    axes[3].set_ylim(IMG_H, 0)
    axes[3].legend(fontsize=8, markerscale=6, loc="lower right")
    axes[3].set_title(
        "(d) where the anchors land in the camera image:\nband rays ~78° off-axis fold back into the frame", fontsize=9
    )
    for ax in (axes[0], axes[1], axes[3]):
        ax.set_xticks([])
        ax.set_yticks([])
    removed = (v_before.any(-1) & ~v_after.any(-1)).sum()
    fig.suptitle(
        f"{args.scene}, camera {args.detail_cam}: off-axis distortion fold-back "
        f"(off-axis check removes {removed} cells; band = {band_mask.sum()} cells, "
        f"main wedge unchanged: {(main_mask & ~v_after.any(-1)).sum()} cells lost)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(args.out_dir, "fig_camera_offaxis.png"), dpi=200)
    plt.close(fig)

    # ---------------- console summary ----------------
    print(f"{args.scene}, snapshot {s['snapshot_id']}, cameras {cams}, z-sign {z_sign}")
    for name in names:
        print(f"  {name:42s}", " ".join(f"{c}:{valid[name][i].any(-1).mean() * 100:5.1f}%" for i, c in enumerate(cams)))
    print("  negative z_cam share:", " ".join(f"{c}:{neg[i][0] * 100:.0f}% of {neg[i][1]}" for i, c in enumerate(cams)))
    print(
        "  visible people in cells removed by the off-axis check:",
        " ".join(f"{c}:{lost[i]}" for i, c in enumerate(cams)),
    )
    print(
        f"  cam {args.detail_cam}: components {n}, main {int(main_mask.sum())} px, band {int(band_mask.sum())} px, "
        f"r main max {r_main.max():.3f}, r band min {r_band.min() if r_band.size else float('nan'):.3f}"
    )
    print(f"  written to {args.out_dir}")


if __name__ == "__main__":
    main()
