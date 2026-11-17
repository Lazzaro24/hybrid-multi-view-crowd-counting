"""
diagnose_hca.py

Diagnostics for the forward-backward hybrid neck of neck_ablation.py (variants hca / hca_mv / hca_static), on exactly the val / test
frames, weights (final EMA) and precise-BN statistics the ablation's "final" evaluation used.

  1. count     whole-scene count of the hybrid, of the hybrid with its cross-attention correction removed (delta = 0,
               same head), and of homography_256 (--ref_run); MAE / bias overall and bias per scene.
  2. correction  per-cell L2 norm of the correction vs the homography features, their cosine, and the density mass the
               correction adds / removes, in cells near people (<= --people_radius m from a GT head) vs empty cells.
  3. sampling  mean anchor weight per height (camera-cell pairs where the camera sees the pillar), mean |offset| in
               feature-map px, and in_share of the anchor weights / offsets: the fraction of their variance that
               changes with the image at a fixed tile cell (0 = the same for every image, as in the legacy necks).

Removing the correction is out of distribution for the head (it was trained on base + delta), so (1)/(2) show the
direction and size of what the correction does, not the count of a homography-only model -- that is homography_256.

Usage (from src):
    uv run python scripts/diagnosis/diagnose_hca.py --run_name ablation_k8 --variant hca
    uv run python scripts/diagnosis/diagnose_hca.py --run_name ablation_k8 --variant hca --frames_per_scene 2   # quick
Writes diagnostics/hca/<run_name>_<variant>.json.
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

import neck_ablation as NA

ANCHOR_REL = None  # filled from the cache settings for the printout (heights relative to d_mean)


def load_model(run_dir, variant, ckpt_args, device):
    ck = torch.load(os.path.join(run_dir, variant, "last.pth"), map_location="cpu", weights_only=False)
    a = argparse.Namespace(**{**ckpt_args, **ck["args"]})
    a.no_grad_checkpoint = True
    if getattr(a, "backbone", None) and not os.path.isfile(a.backbone):
        # a checkpoint trained on another machine stores that machine's absolute path; the file only initialises the
        # (frozen) backbone, whose weights the state dict below overwrites, so the local copy is equivalent
        local = os.path.join(src_dir, "runs", "train_2d", "new_best", os.path.basename(a.backbone))
        print(f"[load] backbone init {a.backbone} not found here; using {local} (weights come from the checkpoint)")
        a.backbone = local
    model = NA.AblationModel(variant, a)
    state = ck["ema_state_dict"]["model"] if ck.get("ema_state_dict") else ck["model_state_dict"]
    model.load_state_dict(state)
    print(f"[load] {variant}: epoch {ck['epoch']} ({'EMA' if ck.get('ema_state_dict') else 'raw'} weights)")
    return model.to(device).eval(), a


def people_mask(gt_xy, wld, shape, radius_cells):
    """Cells within radius_cells of a GT head (same world -> map conversion as neck_ablation.gt_count_in_map)."""
    m = torch.zeros(shape)
    xy = gt_xy.reshape(-1, 2)
    if len(xy):
        r, a, b, w_min, h_min = (wld[i].item() for i in (0, 1, 2, 6, 7))
        cols = torch.round((xy[:, 0] - w_min + a) * r).long()
        rows = torch.round((xy[:, 1] - h_min + b) * r).long()
        ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
        m[rows[ok], cols[ok]] = 1
    k = 2 * radius_cells + 1
    return F.max_pool2d(m[None, None], k, stride=1, padding=radius_cells)[0, 0] > 0


class Moments:
    """Per-cell and global first / second moments of a [L, D] feature, for in_share."""

    def __init__(self):
        self.n = 0
        self.s = self.q = None

    def add(self, x):
        x = x.double()
        if self.s is None:
            self.s, self.q = torch.zeros_like(x), torch.zeros_like(x)
        self.s += x
        self.q += x * x
        self.n += 1

    def cell_std(self):
        """Typical change across images at a fixed cell (sqrt of the mean within-cell variance), in the feature's units."""
        if self.n < 2:
            return float("nan")
        mean = self.s / self.n
        return float((self.q / self.n - mean**2).clamp(min=0).mean().sqrt())

    def in_share(self):
        if self.n < 2:
            return float("nan")
        mean = self.s / self.n
        var_cell = (self.q / self.n - mean**2).clamp(min=0).mean()  # within-cell variance across images
        g_mean = mean.mean(0, keepdim=True)
        var_tot = (self.q / self.n).mean(0) - g_mean[0] ** 2  # variance over images and cells
        return float(var_cell / var_tot.clamp(min=1e-12).mean())


@torch.no_grad()
def run_split(name, scenes, loader, hyb, hom, device, radius_cells, P):
    names = [n for n, _ in scenes]
    bounds = np.cumsum([0] + [len(ds) for _, ds in scenes])
    fusion = hyb.fusion
    store = {}
    hooks = [
        fusion.cross_attention.weight_predictor.register_forward_hook(lambda m, i, o: store.__setitem__("w", o)),
        fusion.cross_attention.offset_predictor.register_forward_hook(lambda m, i, o: store.__setitem__("o", o)),
    ]
    frames = []
    reg = defaultdict(float)  # correction statistics per region
    anchor = {k: torch.zeros(P, dtype=torch.float64) for k in ("people", "empty")}
    anchor_n = {"people": 0, "empty": 0}
    off_sum, off_n = 0.0, 0
    mom_w, mom_o = Moments(), Moments()
    N, Hh = NA.NUM_CAMS, fusion.cross_attention.num_heads

    for sample, idx in zip(tqdm(loader, desc=name, mininterval=10), loader.sampler):
        scene = names[int(np.searchsorted(bounds, idx, side="right") - 1)]
        wld = sample["wld_map_paras"].to(device)
        ctx = {
            "ref_points_full": sample["ref_points_full"].to(device),
            "camera_paras": sample["camera_paras"].to(device),
            "wld_map_paras": wld,
        }
        H, W = int(wld[0, 3].item()), int(wld[0, 4].item())
        maps = {k: torch.zeros((H, W), device=device) for k in ("full", "base", "hom")}
        pm = people_mask(sample["gt_world_xy"][0], sample["wld_map_paras"][0], (H, W), radius_cells).to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            feats = hyb.features(sample["images"].to(device))
            for h0 in NA.tile_starts(H, NA.TILE[0]):
                for w0 in NA.tile_starts(W, NA.TILE[1]):
                    base, delta = fusion.parts(feats, ctx, h0, w0)
                    out_full = hyb.head((base + delta).unsqueeze(1))[0, 0].float()
                    out_base = hyb.head(base.unsqueeze(1))[0, 0].float()
                    out_hom = hom.predict_tile(feats, ctx, h0, w0)[0, 0].float()
                    hh, ww = min(NA.TILE[0], H - h0), min(NA.TILE[1], W - w0)
                    for k, o in (("full", out_full), ("base", out_base), ("hom", out_hom)):
                        maps[k][h0 : h0 + hh, w0 : w0 + ww] = o[:hh, :ww]

                    # --- correction vs homography features, per region (valid part of the tile only)
                    tm = pm[h0 : h0 + hh, w0 : w0 + ww]
                    b, d = base[0, :, :hh, :ww].float(), delta[0, :, :hh, :ww].float()
                    nb, nd = b.norm(dim=0), d.norm(dim=0)
                    cos = (b * d).sum(0) / (nb * nd).clamp(min=1e-6)
                    alpha = (b * d).sum(0) / (nb**2).clamp(min=1e-6)  # delta = alpha * base + orthogonal part
                    orth = (d - alpha * b).norm(dim=0)
                    for rname, rm in (("people", tm), ("empty", ~tm)):
                        reg[f"{rname}_cells"] += float(rm.sum())
                        reg[f"{rname}_norm_base"] += float(nb[rm].sum())
                        reg[f"{rname}_norm_delta"] += float(nd[rm].sum())
                        reg[f"{rname}_cos"] += float(cos[rm].sum())
                        reg[f"{rname}_alpha"] += float(alpha[rm].sum())
                        reg[f"{rname}_norm_orth"] += float(orth[rm].sum())

                    # --- sampling: anchor weights / offsets of every camera that sees the cell
                    L = NA.TILE[0] * NA.TILE[1]
                    w = store["w"].float().view(L, N, Hh, P).softmax(-1)  # B = 1
                    o = store["o"].float().view(L, N, Hh, P, 2)  # feature-map px
                    refs = ctx["ref_points_full"][0, :, h0 : h0 + NA.TILE[0], w0 : w0 + NA.TILE[1]].reshape(N, L, P, 2)
                    inside = ((refs[..., 0].abs() <= 1) & (refs[..., 1].abs() <= 1)).any(-1).T  # [L, N]
                    valid = torch.zeros(NA.TILE, dtype=torch.bool, device=device)
                    valid[:hh, :ww] = True
                    tmf = torch.zeros(NA.TILE, dtype=torch.bool, device=device)
                    tmf[:hh, :ww] = tm
                    wm = w.mean(2)  # [L, N, P] over heads
                    for rname, cell_m in (("people", tmf.view(-1)), ("empty", (valid & ~tmf).view(-1))):
                        sel = inside & cell_m[:, None]
                        anchor[rname] += wm[sel].sum(0).double().cpu()
                        anchor_n[rname] += int(sel.sum())
                    vis = inside & valid.view(-1)[:, None]
                    off_sum += float(o.norm(dim=-1).mean((2, 3))[vis].sum())
                    off_n += int(vis.sum())
                    mom_w.add(w.reshape(L, -1))
                    mom_o.add(o.reshape(L, -1))

        gt = NA.gt_count_in_map(sample["gt_world_xy"][0], sample["wld_map_paras"][0], (H, W))
        rec = {"scene": scene, "gt": gt}
        for k, m in maps.items():
            rec[f"cnt_{k}"] = m.sum().item() / NA.SCALE
            rec[f"mass_people_{k}"] = m[pm].sum().item() / NA.SCALE
            rec[f"mass_empty_{k}"] = m[~pm].sum().item() / NA.SCALE
        frames.append(rec)
    for h in hooks:
        h.remove()

    def err(k):
        e = np.array([f[f"cnt_{k}"] - f["gt"] for f in frames])
        return {"mae": float(np.abs(e).mean()), "bias": float(e.mean())}

    per_scene = defaultdict(lambda: defaultdict(list))
    for f in frames:
        for k in ("full", "base", "hom"):
            per_scene[f["scene"]][k].append(f[f"cnt_{k}"] - f["gt"])
        per_scene[f["scene"]]["gt"].append(f["gt"])

    def mean(k):
        return float(np.mean([f[k] for f in frames]))

    return {
        "n_frames": len(frames),
        "count": {k: err(k) for k in ("full", "base", "hom")},
        "mass": {f"{r}_{k}": mean(f"mass_{r}_{k}") for r in ("people", "empty") for k in ("full", "base", "hom")},
        "per_scene_bias": {s: {k: float(np.mean(v)) for k, v in d.items()} for s, d in per_scene.items()},
        "correction": {
            r: {
                "norm_delta_over_base": reg[f"{r}_norm_delta"] / max(reg[f"{r}_norm_base"], 1e-9),
                "cos_delta_base": reg[f"{r}_cos"] / max(reg[f"{r}_cells"], 1),
                "alpha_along_base": reg[f"{r}_alpha"] / max(reg[f"{r}_cells"], 1),
                "norm_orth_over_base": reg[f"{r}_norm_orth"] / max(reg[f"{r}_norm_base"], 1e-9),
            }
            for r in ("people", "empty")
        },
        "anchor_weight": {r: (anchor[r] / max(anchor_n[r], 1)).tolist() for r in ("people", "empty")},
        "offset_px": off_sum / max(off_n, 1),
        "in_share": {"anchor_weights": mom_w.in_share(), "offsets": mom_o.in_share()},
        "cell_std": {"anchor_weights": mom_w.cell_std(), "offsets_px": mom_o.cell_std()},
        "frames": frames,
    }


def report(name, r, anchor_rel):
    c, m = r["count"], r["mass"]
    print(f"\n=== {name} ({r['n_frames']} frames) ===")
    print(
        f"count      MAE / bias:  hybrid {c['full']['mae']:.1f} / {c['full']['bias']:+.1f} | hybrid without correction "
        f"{c['base']['mae']:.1f} / {c['base']['bias']:+.1f} | homography_256 {c['hom']['mae']:.1f} / {c['hom']['bias']:+.1f}"
    )
    print(
        f"mass/frame near people: hybrid {m['people_full']:.1f}  without corr. {m['people_base']:.1f}  homography_256 "
        f"{m['people_hom']:.1f}"
    )
    print(
        f"mass/frame empty cells: hybrid {m['empty_full']:.1f}  without corr. {m['empty_base']:.1f}  homography_256 "
        f"{m['empty_hom']:.1f}"
    )
    for reg in ("people", "empty"):
        cr = r["correction"][reg]
        print(
            f"correction ({reg:<6}): |delta|/|base| {cr['norm_delta_over_base']:.3f}   cos(delta, base) "
            f"{cr['cos_delta_base']:+.3f}   delta = {cr['alpha_along_base']:+.3f} x base + orthogonal part of "
            f"{cr['norm_orth_over_base']:.3f} x |base|"
        )
    print("anchor weight per height (rel. to head plane d_mean, m): " + "  ".join(f"{h:+.2f}" for h in anchor_rel))
    for reg in ("people", "empty"):
        print(f"   {reg:<6}" + " " * 45 + "  ".join(f"{x:5.3f}" for x in r["anchor_weight"][reg]))
    print(
        f"mean |offset| {r['offset_px']:.2f} feature px | in_share: anchor weights {r['in_share']['anchor_weights']:.3f}, "
        f"offsets {r['in_share']['offsets']:.3f} | change across images at a fixed cell (std): anchor weight "
        f"{r['cell_std']['anchor_weights']:.4f} (uniform = {1 / len(anchor_rel):.3f}), offset {r['cell_std']['offsets_px']:.3f} px"
    )
    print(f"{'scene':<10}{'GT':>7}{'bias hybrid':>13}{'w/o corr.':>11}{'hom_256':>9}")
    for s, d in sorted(r["per_scene_bias"].items(), key=lambda kv: int(kv[0].split("_")[1])):
        print(f"{s:<10}{d['gt']:>7.0f}{d['full']:>+13.1f}{d['base']:>+11.1f}{d['hom']:>+9.1f}")


def main():
    ap = argparse.ArgumentParser(description="Diagnose the hybrid neck (hca / hca_mv) of a neck ablation run.")
    ap.add_argument("--run_name", default="ablation_k8")
    ap.add_argument("--variant", default="hca", choices=["hca", "hca_mv", "hca_static"])
    ap.add_argument("--ref_run", default="ablation_v2", help="run holding homography_256/last.pth")
    ap.add_argument("--splits", default="validate,test")
    ap.add_argument("--frames_per_scene", type=int, default=None, help="default: the run's val/test frames per scene")
    ap.add_argument(
        "--people_radius", type=float, default=1.0, help='"near people" = within this many metres of a head'
    )
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(src_dir, "runs", "neck_ablation", args.run_name)

    hyb, a = load_model(run_dir, args.variant, {}, device)
    hom, _ = load_model(os.path.join(src_dir, "runs", "neck_ablation", args.ref_run), "homography_256", vars(a), device)
    P = a.z_anchors
    d_delta, below, above = 1.75, 2.5, 2.5  # master_refs_k8.pt (utils/cache_reference.py); printout only
    anchor_rel = np.linspace(-d_delta - below, above, P) if P == 8 else np.linspace(-d_delta, 1.8 + d_delta, P)

    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    excluded = NA.load_excluded_scenes()
    train = NA.load_split("train", cams_cfg, excluded, a.refs_file, a.z_anchors)
    # precise-BN exactly as before the run's evaluation (same frames), for both models
    for m in (hyb, hom):
        NA.precise_bn(m, NA.make_loader(train, 1, a.val_seed + 7, args.num_workers), device)
    del train

    radius_cells = int(round(args.people_radius * 5.0))  # r = 5 cells per metre (wld_map_paras[0])
    out = {
        "run": args.run_name,
        "variant": args.variant,
        "people_radius_m": args.people_radius,
        "anchor_rel_m": list(map(float, anchor_rel)),
    }
    for split in args.splits.split(","):
        scenes = NA.load_split(split, cams_cfg, excluded, a.refs_file, a.z_anchors)
        if split == "validate":
            fps, seed = args.frames_per_scene or a.val_frames_per_scene, a.val_seed
        else:
            fps, seed = args.frames_per_scene or a.test_frames_per_scene, a.val_seed + 13
        r = run_split(
            split, scenes, NA.make_loader(scenes, fps, seed, args.num_workers), hyb, hom, device, radius_cells, P
        )
        report(split, r, anchor_rel)
        out[split] = r
        del scenes

    os.makedirs(os.path.join(src_dir, "diagnostics", "hca"), exist_ok=True)
    path = os.path.join(src_dir, "diagnostics", "hca", f"{args.run_name}_{args.variant}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
