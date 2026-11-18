"""
scripts/visualisation/plot_thesis_ch6.py

The three figures of thesis Chapter 6, drawn at the text width of the thesis (388.5 pt = 5.4 in) so that their
fonts print at the nominal size (8-9 pt), with the model names of the thesis:

  final_models  maps of all final models on one test frame: whole-scene density, zoom into the densest 16 x 16 m,
                routing (oracle zone / router mask / density-rule zone). Same frame pick and prediction paths as
                plot_final_models.py (doc/final_evaluations.md E8); runs the five models on that one frame.
  views         test MAE, F1 and signed error over 3 / 5 / 7 / 10 views with view normalization
                (camera_ablation/summary.json, doc/camera_ablation.md section 7.1); the 5-view bias of the two
                learned-router models comes from the final-evaluation reruns (doc/final_evaluations.md section 7.4).
  per_scene     test MAE per scene of both baselines and the Homography-Hybrid with learned and oracle routing
                (per-sample CSVs of runs/final_evals, doc/final_evaluations.md E1).

Usage (from src):
    uv run python scripts/visualisation/plot_thesis_ch6.py views per_scene final_models --out_dir figs/thesis
"""

import os
import sys
import csv
import json
import random
import argparse
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, "evaluation"))
sys.path.append(os.path.join(src_dir, "scripts", "visualisation"))

import plot_camera_ablation as CA

TEXT_WIDTH = 5.4  # inches, the text block of the thesis template
FS, FS_SMALL = 9, 8
SURFACE, INK, INK_2, MUTED, GRID, AXIS = CA.SURFACE, CA.INK, CA.INK_2, CA.MUTED, CA.GRID, CA.AXIS
SERIES = CA.SERIES
# code name -> thesis name
NAMES = {
    "baseline_ot": "Baseline (OT)",
    "baseline_mse": "Baseline (Gaussian)",
    "h1_hom": "Homography-Hybrid",
    "h1_casa": "Attention-Hybrid",
    "hybrid_hom": "Homography-Rule",
}
COLORS = {"baseline_ot": SERIES[0], "h1_hom": SERIES[2], "h1_casa": SERIES[1], "hybrid_hom": SERIES[3]}
# 5-view bias of the learned-router models, not recorded by the camera-ablation run (final_evaluations.md 7.4)
BIAS_5V = {"h1_hom": -7.4, "h1_casa": -5.6}


def setup():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
            "svg.fonttype": "path",
            "text.color": INK,
        }
    )


def style_axes(ax, ylabel=None, title=None, xlabel=None):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=FS_SMALL, length=2.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=FS)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=FS)
    if title:
        ax.set_title(title, loc="left", fontsize=FS, color=INK)


def save(fig, out, exts=("svg", "png")):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for ext in exts:
        fig.savefig(f"{out}.{ext}", dpi=300, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved {out}.{ext}")
    plt.close(fig)


def views(out_dir):
    with open(os.path.join(src_dir, "camera_ablation", "summary.json")) as f:
        summary = json.load(f)
    for m, b in BIAS_5V.items():
        summary["tier_b"]["vnorm"][m].setdefault("5", {})["bias"] = b
    models = ["baseline_ot", "h1_hom", "h1_casa", "hybrid_hom"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), facecolor=SURFACE)
    for ax, key, title in (
        (axes[0], "MAE", "(a) MAE"),
        (axes[1], "F1", "(b) F1 (%)"),
        (axes[2], "bias", "(c) Mean signed error"),
    ):
        for m in models:
            ys = CA.tier_b(summary, "vnorm", m, key)
            ax.plot(CA.VIEWS, ys, color=COLORS[m], linewidth=1.5, marker="o", markersize=3.5)
        style_axes(ax, title=title, xlabel="views")
        ax.set_xticks(CA.VIEWS)
        if key == "bias":
            ax.axhline(0, color=AXIS, linewidth=0.9, zorder=1)
    fig.legend(
        handles=[
            Line2D([], [], color=COLORS[m], linewidth=1.5, marker="o", markersize=3.5, label=NAMES[m]) for m in models
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=FS_SMALL,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.2)
    save(fig, os.path.join(out_dir, "fig_6_2_views"))


def per_scene(out_dir):
    models = [
        ("baseline_mse", "Baseline (Gaussian)", MUTED, True),
        ("baseline_ot", "Baseline (OT)", SERIES[0], True),
        ("h1_hom", "Homography-Hybrid", SERIES[2], True),
        ("h1_hom_oracle", "Homography-Hybrid, oracle routing", SERIES[2], False),
    ]
    mae = {}
    for m, *_ in models:
        acc = defaultdict(list)
        with open(os.path.join(src_dir, "runs", "final_evals", m + ".samples.csv")) as f:
            for r in csv.DictReader(f):
                acc[r["scene"]].append(abs(float(r["err"])))
        mae[m] = {s: float(np.mean(v)) for s, v in acc.items()}
    scenes = sorted(mae["baseline_mse"], key=lambda s: mae["baseline_mse"][s])
    for s in scenes:
        print(s, " ".join(f"{m}={mae[m][s]:.1f}" for m, *_ in models))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.6), facecolor=SURFACE)
    for i, s in enumerate(scenes):
        vals = [mae[m][s] for m, *_ in models]
        ax.plot([i, i], [min(vals), max(vals)], color=GRID, linewidth=1.0, zorder=1)
    for k, (m, name, color, filled) in enumerate(models):
        off = (k - 1.5) * 0.14
        ax.scatter(
            [i + off for i in range(len(scenes))],
            [mae[m][s] for s in scenes],
            s=20,
            color=color if filled else SURFACE,
            edgecolors=color,
            linewidths=1.2,
            zorder=3,
            label=name,
        )
    style_axes(ax, ylabel="test MAE", xlabel="test scene")
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels([s.replace("scene_", "") for s in scenes])
    ax.legend(loc="upper left", frameon=False, fontsize=FS_SMALL, ncol=2, handletextpad=0.3, columnspacing=1.0)
    ax.set_ylim(0, 175)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "fig_6_3_per_scene"))


def final_models(out_dir, pick="dense", seed=0, zoom=80):
    import torch
    import evaluate_paper_protocol as EP
    import plot_final_models as PF
    from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
    from losses.gaussian_mse import get_map_kernel
    from visualize_paper_prediction import DENSITY_CMAP, FP_COLOR, densest_window, gt_density

    args = argparse.Namespace(
        pick=pick,
        samples_csv=os.path.join(src_dir, "runs", "final_evals", "baseline_ot.samples.csv"),
        occlusion_csv=os.path.join(src_dir, "diagnostics", "occlusion", "heldout14_final", "persons.csv"),
        test_scenes="scene_5,scene_7,scene_11,scene_12,scene_13,scene_16,scene_29,scene_50,scene_78,scene_80",
    )
    os.chdir(src_dir)  # the model paths of plot_final_models are relative to src
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene, frame, cams = PF.pick_frame(args)
    ds = SynMVCrowdDataset(scene_dir=PF.scene_dir(scene), camera_ids=cams, num_sampled_cams=5, load_refs=True)
    random.seed(seed * 100_003 + frame)  # the same draw as plot_final_models.py
    raw = ds[frame]
    sample = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in raw.items()}
    wld = sample["wld_map_paras"]
    shape = (int(wld[0, 3].item()), int(wld[0, 4].item()))
    gts = EP.gt_pixels(sample["gt_world_xy"], wld, shape)

    results, T = [], None
    for spec in PF.MODELS:
        pred_t, zone, t = PF.predict(spec, sample, device)
        T = T or t
        results.append((spec, pred_t.cpu().numpy() / EP.SCALE, zone))
        print(f"  {spec[0]:<12} count {results[-1][1].sum():7.1f} (GT {len(gts)})")
    oracle = EP.oracle_zone(T, gts, shape).float().numpy()
    gt = gt_density(gts, shape, get_map_kernel(sigma=5, kernel_size=20))
    vmax = gt.max()
    r0, c0 = densest_window(gt, zoom)

    def two_lines(name):
        return name.replace("-", "-\n", 1) if "-" in name else name.replace(" (", "\n(")

    columns = [("Ground\ntruth", gt, oracle, f"{len(gts)}", "oracle zone")] + [
        (
            two_lines(NAMES[spec[0]]),
            pred,
            zone,
            f"{pred.sum():.0f} ({pred.sum() - len(gts):+.0f})",
            None if zone is None else ("router mask" if spec[3] == "learned" else "rule zone"),
        )
        for spec, pred, zone in results
    ]
    ncol = len(columns)
    fig, axes = plt.subplots(3, ncol, figsize=(TEXT_WIDTH, 3.75), facecolor=SURFACE)
    plt.subplots_adjust(left=0.07, right=1.0, top=0.86, bottom=0.12, wspace=0.06, hspace=0.16)
    for c, (label, dmap, zone, count, zone_title) in enumerate(columns):
        ax = axes[0, c]
        im = ax.imshow(dmap, cmap=DENSITY_CMAP, vmin=0, vmax=vmax, interpolation="nearest")
        ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), zoom, zoom, fill=False, ec=INK, lw=0.5, ls="--"))
        ax.set_title(f"{label}\n{count}", fontsize=FS_SMALL, color=INK, loc="left", pad=3)
        axes[1, c].imshow(dmap[r0 : r0 + zoom, c0 : c0 + zoom], cmap=DENSITY_CMAP, vmin=0, vmax=vmax)
        ax = axes[2, c]
        if zone is not None:
            ax.imshow(zone, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(zone_title, fontsize=7, color=INK_2, loc="left", pad=1.5)
            if c > 0:
                ax.contour(oracle, levels=[0.5], colors=[FP_COLOR], linewidths=0.35)
        else:
            ax.set_facecolor(SURFACE)
        for a in axes[:, c]:
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_color(AXIS if a is not axes[2, c] or zone is not None else SURFACE)
                s.set_linewidth(0.5)
    for r, name in enumerate(("density map", "zoom 16 × 16 m", "routing")):
        axes[r, 0].set_ylabel(name, fontsize=FS_SMALL, color=INK_2)
    cax = fig.add_axes([0.07, 0.065, 0.36, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("people per 0.2 m cell", color=INK_2, fontsize=FS_SMALL, labelpad=1)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, labelsize=7, length=2)
    fig.legend(
        handles=[
            Line2D([], [], ls="", marker="s", color="#444444", label="dense (zone or router mask)"),
            Line2D([], [], color=FP_COLOR, lw=0.8, label="oracle zone outline"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 0.03),
        ncol=1,
        frameon=False,
        fontsize=FS_SMALL,
    )
    save(fig, os.path.join(out_dir, "fig_6_1_final_models_dense"), exts=("png",))


# Short-budget head ablation on homography_256 (doc/head_ablation.md 8.9-8.11, 8.22-8.24): label, model type,
# routing, weights -- the same checkpoints and routing as the test evals of thesis Table "tab:rq2-routing".
HEAD_ABLATION = [
    ("Decoder", "ablation", "learned", "runs/neck_ablation/ablation_v2/homography_256/last.pth"),
    ("Decoder\n(long)", "ablation", "learned", "runs/head_ablation/homography_256/h0_long/1_decoder/homography_256/last.pth"),
    (
        "Pointwise\nmask",
        "proposed",
        "learned",
        "runs/head_ablation/homography_256/og_mask/4_router_anchor/"
        "multiscene_final_stage_4_abl_homography_256_og_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth",
    ),
    (
        "Window\nmask",
        "proposed",
        "learned",
        "runs/head_ablation/homography_256/window_mask_D0.4/4_router_anchor/"
        "multiscene_final_stage_4_abl_homography_256_W25D0.4_ma50_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth",
    ),
    (
        "Performance-\nbased",
        "proposed",
        "learned",
        "runs/head_ablation/homography_256/perf_router/c10_sel/"
        "multiscene_final_stage_3_abl_homography_256_unmasked_joint_genloss_ep30_perf_c10_sel.pth",
    ),
    (
        "Density\nrule",
        "proposed",
        "dzone0.15p",
        "runs/head_ablation/homography_256/dense_recipe/S4D2_s42_ema/"
        "multiscene_final_stage_2_abl_homography_256_dense_S4D2_s42_ema.pth",
    ),
]


def head_ablation(out_dir, pick="scene_78/55", seed=0, zoom=80):
    """Maps of the short-budget heads on one test frame, laid out like final_models. Default frame: the test frame
    closest to the median GT count (513) and the median oracle dense share (2.8%) of all test frames, scene 16
    excluded since final_models already shows it. The closest frame, scene_12/57, has a tall, narrow map; the
    default is the third closest, scene_78/55 (513 people on average over the E1 draws, dense share 2.7%), whose
    wide map fills the text width."""
    import torch
    import evaluate_paper_protocol as EP
    import plot_final_models as PF
    from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
    from losses.gaussian_mse import get_map_kernel
    from visualize_paper_prediction import DENSITY_CMAP, FP_COLOR, densest_window, gt_density

    os.chdir(src_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene, frame = pick.split("/")
    frame = int(frame)
    ds = SynMVCrowdDataset(scene_dir=PF.scene_dir(scene), camera_ids=None, num_sampled_cams=5, load_refs=True)
    random.seed(seed * 100_003 + frame)  # the same draw rule as plot_final_models.py
    raw = ds[frame]
    sample = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in raw.items()}
    wld = sample["wld_map_paras"]
    shape = (int(wld[0, 3].item()), int(wld[0, 4].item()))
    gts = EP.gt_pixels(sample["gt_world_xy"], wld, shape)

    results, T = [], None
    for label, model_type, routing, weights in HEAD_ABLATION:
        if model_type == "ablation":
            model, _ = EP.load_model("ablation", os.path.join(src_dir, weights), device)
            pred_t, zone = EP.predict_ablation(model, sample, device), None
            del model
            torch.cuda.empty_cache()
        else:
            pred_t, zone, t = PF.predict((label, label, model_type, routing, weights), sample, device)
            T = T or t
        pred = pred_t.cpu().numpy() / EP.SCALE
        results.append((label, pred, zone, routing))
        print(f"  {label.replace(chr(10), ' '):<20} count {pred.sum():7.1f} (GT {len(gts)})")
    oracle = EP.oracle_zone(T, gts, shape).float().numpy()
    gt = gt_density(gts, shape, get_map_kernel(sigma=5, kernel_size=20))
    vmax = gt.max()
    r0, c0 = densest_window(gt, zoom)

    columns = [("Ground\ntruth", gt, oracle, f"{len(gts)}", "oracle zone")] + [
        (
            label,
            pred,
            zone,
            f"{pred.sum():.0f} ({pred.sum() - len(gts):+.0f})",
            None if zone is None else ("router mask" if routing == "learned" else "rule zone"),
        )
        for label, pred, zone, routing in results
    ]
    ncol = len(columns)
    fig, axes = plt.subplots(3, ncol, figsize=(TEXT_WIDTH, 3.15), facecolor=SURFACE)
    plt.subplots_adjust(left=0.05, right=1.0, top=0.84, bottom=0.15, wspace=0.06, hspace=0.2)
    for c, (label, dmap, zone, count, zone_title) in enumerate(columns):
        ax = axes[0, c]
        im = ax.imshow(dmap, cmap=DENSITY_CMAP, vmin=0, vmax=vmax, interpolation="nearest")
        ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), zoom, zoom, fill=False, ec=INK, lw=0.5, ls="--"))
        ax.set_title(f"{label}\n{count}", fontsize=7.5, color=INK, loc="left", pad=3)
        axes[1, c].imshow(dmap[r0 : r0 + zoom, c0 : c0 + zoom], cmap=DENSITY_CMAP, vmin=0, vmax=vmax)
        ax = axes[2, c]
        if zone is not None:
            ax.imshow(zone, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(zone_title, fontsize=7, color=INK_2, loc="left", pad=1.5)
            if c > 0:
                ax.contour(oracle, levels=[0.5], colors=[FP_COLOR], linewidths=0.35)
        else:
            ax.set_facecolor(SURFACE)
        for a in axes[:, c]:
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_color(AXIS if a is not axes[2, c] or zone is not None else SURFACE)
                s.set_linewidth(0.5)
    for r, name in enumerate(("density map", "zoom", "routing")):
        axes[r, 0].set_ylabel(name, fontsize=FS_SMALL, color=INK_2)
    cax = fig.add_axes([0.05, 0.065, 0.36, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("people per 0.2 m cell", color=INK_2, fontsize=FS_SMALL, labelpad=1)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, labelsize=7, length=2)
    fig.legend(
        handles=[
            Line2D([], [], ls="", marker="s", color="#444444", label="dense (zone or router mask)"),
            Line2D([], [], color=FP_COLOR, lw=0.8, label="oracle zone outline"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        ncol=1,
        frameon=False,
        fontsize=FS_SMALL,
    )
    save(fig, os.path.join(out_dir, "fig_6_5_head_ablation"), exts=("png",))


def occlusion_curves(out_dir, run="heldout14_ca_sa4", scene="scene_1"):
    """Recovery against the number of blocked views for people in view of all five cameras, homography_256 vs ca_sa
    (4 anchors), from the per-person CSV of the occlusion run (doc/neck_ablation.md section 8).
      (a) one scene, views behind static occluders (the scene of the occluder-mask figure);
      (b) all 14 held-out scenes, views blocked by other people, scene-centred: each scene's mean recovery is removed
          and the overall mean added back, so the curve compares people within a scene, as the within-scene
          regression does. The last bin pools 3-5 blocked views.
    Pooled raw means are not used: they mix scenes and positions and do not decrease with the static blocks."""
    with open(os.path.join(src_dir, "diagnostics", "occlusion", run, "persons.csv")) as f:
        rows = [r for r in csv.DictReader(f) if int(r["n_fov"]) == 5]
    necks = [("homography_256", "Homography", "#2E86AB"), ("ca_sa", "CA→SA (proposed)", "#E4572E")]

    def curve(sel, key, cap, centre):
        smean = defaultdict(lambda: [0.0, 0.0, 0])
        for r in sel:
            a = smean[r["scene"]]
            for i, (m, _, _) in enumerate(necks):
                a[i] += float(r["recovery_" + m])
            a[2] += 1
        n_all = sum(v[2] for v in smean.values())
        gmean = [sum(v[i] for v in smean.values()) / n_all for i in (0, 1)]
        acc = defaultdict(lambda: [0.0, 0.0, 0])
        for r in sel:
            s, a = smean[r["scene"]], acc[min(int(r[key]), cap)]
            for i, (m, _, _) in enumerate(necks):
                v = float(r["recovery_" + m])
                a[i] += v - s[i] / s[2] + gmean[i] if centre else v
            a[2] += 1
        ks = sorted(acc)
        print(key, {k: (round(acc[k][0] / acc[k][2], 3), round(acc[k][1] / acc[k][2], 3), acc[k][2]) for k in ks})
        return ks, [[acc[k][i] / acc[k][2] for k in ks] for i in (0, 1)]

    panels = (
        (
            [r for r in rows if r["scene"] == scene],
            "n_block_static",
            5,
            False,
            "(a) Validation scene 1",
            "views behind static occluders",
            [str(k) for k in range(6)],
        ),
        (rows, "n_block_people", 3, True, "(b) 14 held-out scenes", "views blocked by other people", ["0", "1", "2", "3–5"]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.4), facecolor=SURFACE)
    for ax, (sel, key, cap, centre, title, xlabel, ticks) in zip(axes, panels):
        ks, ys = curve(sel, key, cap, centre)
        for (m, name, color), y in zip(necks, ys):
            ax.plot(ks, y, color=color, linewidth=1.5, marker="o", markersize=3.5, label=name)
        style_axes(ax, ylabel="mean recovery" if ax is axes[0] else None, title=title, xlabel=xlabel)
        ax.set_xticks(ks)
        ax.set_xticklabels(ticks)
        ax.set_ylim(0, 0.85)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=FS_SMALL,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
    save(fig, os.path.join(out_dir, "fig_6_occlusion_recovery"))


def occluder(out_dir, scene="validate/scene_1", cam=25, min_hits=5, min_empty=5, cell=8):
    """Camera image and static-occluder mask of one camera (utils/occluder_masks.py, diagnostics/occluders/)."""
    import cv2
    from matplotlib.patches import Patch

    split, name = scene.split("/")
    scene_dir = os.path.join(src_dir, "datasets", "SynMVCrowd", "data", split, name)
    d = np.load(os.path.join(src_dir, "diagnostics", "occluders", name, f"cam{cam}_contrast.npz"))
    occ = d["occ"] > 0
    labelled = (d["n_head"] >= min_hits) & (d["n_empty"] >= min_empty) & ~occ
    # brightest snapshot of a sample, so that the scene content is visible
    snaps = sorted([s for s in os.listdir(scene_dir) if s.isdigit()], key=int)
    best, best_v = None, -1.0
    for s in snaps[::5]:
        ip = os.path.join(scene_dir, s, "jpgs", f"{cam}.jpg")
        if os.path.exists(ip):
            im = cv2.imread(ip)
            v = float(im.mean())
            if v > best_v:
                best, best_v = im, v
    h, w = 540, 960
    img = cv2.cvtColor(cv2.resize(best, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) / 255.0

    def up(a):
        u = np.kron(a.astype(np.uint8), np.ones((cell, cell), np.uint8))
        return np.pad(u, ((0, h - u.shape[0]), (0, w - u.shape[1]))) > 0

    occ_px, lab_px = up(occ), up(labelled)
    ov = img * 0.75
    red, green = np.array([0.85, 0.15, 0.10]), np.array([0.20, 0.70, 0.30])
    ov[occ_px] = 0.45 * ov[occ_px] + 0.55 * red
    ov[lab_px] = 0.70 * ov[lab_px] + 0.30 * green

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, TEXT_WIDTH / 2 * h / w + 0.55))
    for ax, im, title in zip(axes, (img, ov), ("(a) Camera image", "(b) Occluder mask")):
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=FS, color=INK)
    fig.legend(
        handles=[
            Patch(facecolor=red, alpha=0.8, label="static occluder"),
            Patch(facecolor=green, alpha=0.6, label="labeled, visible"),
            Patch(facecolor="none", edgecolor=AXIS, label="no label (too few heads or empty snapshots)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=FS_SMALL,
        handlelength=1.2,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0.14, wspace=0.03)
    save(fig, os.path.join(out_dir, "fig_6_occluder_mask"), exts=("png",))


def attention(out_dir, scene="test/scene_16", slot=0, n_cells=7, ckpt="runs/neck_ablation/ablation_v2/ca_sa/last.pth"):
    """What the cross-attention of the proposed neck learned (doc/neck_ablation.md, 'What the cross-attention
    learned'). Offsets and weights come from the learned queries only, so they are computed from the EMA weights
    without any image: (a) mean weight per height anchor over all cells and camera slots, (b) offset per anchor in
    image pixels, (c) reference points and learned sampling points of a few cells at heads on one camera image."""
    import cv2
    import torch

    st = torch.load(os.path.join(src_dir, ckpt), map_location="cpu", weights_only=False)["ema_state_dict"]["model"]
    q = st["fusion.bev_queries"].float()
    ca = "fusion.cross_attention."
    off = (q @ st[ca + "offset_predictor.weight"].T + st[ca + "offset_predictor.bias"]).view(-1, 5, 4, 2) * 0.1
    w = torch.softmax((q @ st[ca + "weight_predictor.weight"].T + st[ca + "weight_predictor.bias"]).view(-1, 5, 4), -1)
    img_w, img_h = 1920, 1080
    off_px = torch.stack([off[..., 0] * img_w / 2, off[..., 1] * img_h / 2], -1)  # normalized -> image pixels
    w_mean = w.mean((0, 1)).numpy()
    off_len = off_px.norm(dim=-1)  # [cells, slots, anchors]
    off_med = off_len.median(0).values.median(0).values.numpy()
    off_p90 = off_len.flatten(0, 1).quantile(0.9, dim=0).numpy()
    print("weight per anchor", np.round(w_mean, 3), "offset median px", np.round(off_med, 1), "p90", np.round(off_p90, 1))

    # one camera of the scene's k-means set, its first snapshot, the heads it sees
    split, name = scene.split("/")
    scene_dir = os.path.join(src_dir, "datasets", "SynMVCrowd", "data", split, name)
    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cam = json.load(f)[name]["5"][slot]
    refs = torch.load(os.path.join(scene_dir, "master_refs.pt"), map_location="cpu", weights_only=False)[cam]
    snaps = sorted([s for s in os.listdir(scene_dir) if s.isdigit()], key=int)
    with open(os.path.join(scene_dir, snaps[0], "jsons", f"{cam}.json")) as f:
        paras = json.load(f)["wld_map_paras"]  # the reference points use the grid of the first snapshot
    paras = np.ravel(json.loads(paras) if isinstance(paras, str) else paras).astype(float)
    w_min, h_min = paras[8], paras[9]  # s, r0, a0, b0, h4, w4, d_delta, d_mean, w_min, h_min
    # brightest of a sample of snapshots, so that the scene is visible; its heads are mapped on the first grid
    snap = max(snaps[::10], key=lambda s: float(cv2.imread(os.path.join(scene_dir, s, "jpgs", f"{cam}.jpg")).mean()))
    with open(os.path.join(scene_dir, snap, "jsons", f"{cam}.json")) as f:
        js = json.load(f)
    heads = [p["world"] for p in js["image_info"] if p.get("pixel") is not None]
    cells = sorted({(int((y - h_min + 5) * 5), int((x - w_min + 5) * 5)) for x, y, _ in heads})  # as the loader
    tile = 200  # the first test tile starts at (0, 0): query index = row * 200 + col
    ok = [(r, c) for r, c in cells if r < tile and c < tile and bool((refs[r, c].abs() <= 0.95).all())]
    rng = random.Random(0)
    pick = [ok[0]] if ok else []
    while len(pick) < min(n_cells, len(ok)):  # farthest-point spread over the tile
        cand = rng.sample(ok, min(len(ok), 400))
        pick.append(max(cand, key=lambda rc: min((rc[0] - p[0]) ** 2 + (rc[1] - p[1]) ** 2 for p in pick)))

    img = cv2.imread(os.path.join(scene_dir, snap, "jpgs", f"{cam}.jpg"))
    disp_w, disp_h = 960, 540
    img = cv2.cvtColor(cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) / 255.0
    sx, sy = disp_w / img_w, disp_h / img_h

    def mix(c, t, a):
        c, t = np.array(matplotlib.colors.to_rgb(c)), np.array(matplotlib.colors.to_rgb(t))
        return tuple((1 - a) * c + a * t)

    base = SERIES[1]  # the Attention-Hybrid's colour in the other Chapter 6 figures; one hue, ordered by height
    anchor_col = [mix(base, "white", 0.55), base, mix(base, "black", 0.3), mix(base, "black", 0.55)]
    anchor_lab = ["−1.75 m\n(feet)", "0 m\n(head)", "+1.78 m", "+3.55 m"]

    fig = plt.figure(figsize=(TEXT_WIDTH, 4.7), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.75], hspace=0.38, wspace=0.35)
    ax_w, ax_o, ax_i = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    x = np.arange(4)
    ax_w.bar(x, w_mean * 100, color=anchor_col, width=0.6, edgecolor=SURFACE, linewidth=1)
    ax_w.axhline(25, color=INK_2, linewidth=0.9, linestyle="--", zorder=3)
    ax_w.text(3.5, 33.5, "dashed: uniform (25%)", ha="right", va="bottom", fontsize=FS_SMALL, color=INK_2)
    for i, v in enumerate(w_mean * 100):  # inside the bar, clear of the dashed line
        ax_w.text(i, 2.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=FS_SMALL, color=INK if i == 0 else "white")
    style_axes(ax_w, ylabel="mean weight (%)", title="(a) Weight per height anchor")
    ax_w.set_ylim(0, 38)
    ax_o.bar(x, off_med, color=anchor_col, width=0.6, edgecolor=SURFACE, linewidth=1)
    ax_o.errorbar(x, off_med, yerr=[np.zeros(4), off_p90 - off_med], fmt="none", ecolor=INK_2, elinewidth=0.9, capsize=2)
    style_axes(ax_o, ylabel="offset (image px)", title="(b) Offset, median to 90th percentile")
    for ax in (ax_w, ax_o):
        ax.set_xticks(x)
        ax.set_xticklabels(anchor_lab, fontsize=FS_SMALL - 0.5)
        ax.tick_params(axis="x", length=0)

    ax_i.imshow(img * 0.8 + 0.1)
    for r, c in pick:
        qi = r * tile + c
        ref = refs[r, c].numpy()  # [4, 2] normalized image coordinates
        rp = np.stack([(ref[:, 0] + 1) / 2 * img_w * sx, (ref[:, 1] + 1) / 2 * img_h * sy], -1)
        sp = rp + off_px[qi, slot].numpy() * np.array([sx, sy])
        ax_i.plot(rp[:, 0], rp[:, 1], color="white", linewidth=0.8, linestyle=(0, (2, 2)), alpha=0.9, zorder=2)
        for z in range(4):
            ax_i.plot([rp[z, 0], sp[z, 0]], [rp[z, 1], sp[z, 1]], color="white", linewidth=0.8, zorder=3)
            ax_i.scatter(rp[z, 0], rp[z, 1], s=22, facecolors="none", edgecolors=anchor_col[z], linewidths=1.3, zorder=4)
            ax_i.scatter(sp[z, 0], sp[z, 1], s=10 + 140 * float(w[qi, slot, z]), color=anchor_col[z],
                         edgecolors="white", linewidths=0.6, zorder=5)
    ax_i.set_xlim(0, disp_w)
    ax_i.set_ylim(disp_h, 0)
    ax_i.set_axis_off()
    ax_i.set_title(f"(c) Reference and sampling points of {len(pick)} cells at heads, one camera", loc="left",
                   fontsize=FS, color=INK)
    handles = [
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="none", markeredgecolor=INK_2, markersize=5,
               label="reference point (projected anchor)"),
        Line2D([], [], linestyle="none", marker="o", color=INK_2, markersize=5, label="sampling point (size = weight)"),
    ] + [Line2D([], [], linestyle="none", marker="s", color=anchor_col[z], markersize=6,
                label=anchor_lab[z].replace("\n", " ")) for z in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=FS_SMALL,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.0)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.95, bottom=0.1)
    print(f"scene {scene}, camera {cam}, snapshot {snap}, cells {pick}")
    save(fig, os.path.join(out_dir, "fig_6_4_attention"))


# Paired MAE effects with 95% intervals over the 10 test scenes (positive = the change costs accuracy).
# Sources: final_evaluations.md 7.4 / 7.8 (per-sample scene bootstrap); head_ablation.md 8.20, joint_training.md 7.2
# and camera_ablation.md 7.3 (bootstrap over the per-scene differences).
EFFECTS = [
    ("Training and pipeline", None),
    ("Proposed instead of baseline pipeline\n(same head, both with OT fine-tune)", (25.0, 7.4, 40.8)),
    ("MSE instead of OT fine-tune,\nbaseline pipeline", (20.7, -2.9, 46.2)),
    ("MSE instead of OT fine-tune,\nproposed pipeline (decoder)", (11.2, -7.9, 30.7)),
    ("Single loop instead of stages\n(Homography-Hybrid)", (18.9, 10.9, 26.7)),
    ("Components under study", None),
    ("Learned instead of oracle routing\n(Homography-Hybrid)", (11.6, 3.0, 22.2)),
    ("Attention instead of homography neck\n(learned-router head)", (10.3, -0.9, 20.2)),
    ("Density-rule head instead of decoder\n(same fusion and budget)", (-4.7, -22.1, 12.0)),
]


def effects(out_dir):
    n = len(EFFECTS)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.9), facecolor=SURFACE)
    ticks, labels = [], []
    for i, (lab, v) in enumerate(EFFECTS):
        y = n - 1 - i
        if v is None:
            ax.text(-25, y, lab, ha="left", va="center", fontsize=FS, color=INK, fontweight="bold")
            continue
        m, lo, hi = v
        sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [y, y], color=INK_2, linewidth=1.6, solid_capstyle="round", zorder=2)
        ax.scatter([m], [y], s=34, color=INK_2 if sig else SURFACE, edgecolors=INK_2, linewidths=1.4, zorder=3)
        label = f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]".replace("-", "−")
        ax.text(49, y, label, ha="left", va="center", fontsize=FS_SMALL, color=INK_2)
        ticks.append(y)
        labels.append(lab)
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=FS_SMALL)
    style_axes(ax, xlabel="change in test MAE (people); positive = less accurate")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-25, 48)
    ax.set_ylim(-0.6, n - 0.4)
    ax.legend(
        handles=[
            Line2D([], [], linestyle="none", marker="o", color=INK_2, markersize=5, label="interval excludes zero"),
            Line2D([], [], linestyle="none", marker="o", markerfacecolor=SURFACE, markeredgecolor=INK_2,
                   markersize=5, label="interval includes zero"),
        ],
        loc="lower left", frameon=False, fontsize=FS_SMALL, bbox_to_anchor=(0.0, 1.0), ncol=2,
    )
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "fig_7_1_effects"))


def main():
    ap = argparse.ArgumentParser(description="The figures of thesis Chapters 6 and 7")
    ap.add_argument(
        "figures",
        nargs="+",
        choices=["final_models", "head_ablation", "views", "per_scene", "occluder", "occlusion_curves", "attention", "effects"],
    )
    ap.add_argument("--out_dir", default=os.path.join(src_dir, "figs", "thesis"))
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    setup()
    for name in args.figures:
        {
            "final_models": final_models,
            "head_ablation": head_ablation,
            "views": views,
            "per_scene": per_scene,
            "occluder": occluder,
            "occlusion_curves": occlusion_curves,
            "attention": attention,
            "effects": effects,
        }[name](out_dir)


if __name__ == "__main__":
    main()
