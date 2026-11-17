"""
visualize_experts.py

Shows what the DENSE and SPARSE experts of a multi-scene checkpoint actually output, on TRAIN scenes (including scene_14, the
single-scene debug scene) and VAL scenes, in the layout of the single-scene debug figures:

    gt_mask | true density | dense raw (unmasked) | dense masked (mass pred vs true) | sparse raw (unmasked) | sparse masked
    [+ Stage 1 unified head, if --stage1_checkpoint is given]

All density panels share ONE colour scale per figure (max of the true density), so a flat plateau is visible as a plateau.
It also prints per-scene numbers that make the pictures comparable:
    corr(dense raw, true density) and corr(sparse raw, true density) over ALL pixels of a patch (spatial localisation),
    mean dense output inside / outside the oracle mask, in-mask and sparse-region mass (pred vs true, persons/patch),
and writes response curves E[output | true density bin] for the train pool and the val pool (flat curve = constant output).

The question it answers: in the single-scene run the experts localise people from the jointly trained neck features. Does the
multi-scene checkpoint do the same (a) on a TRAIN scene, especially scene_14, (b) on unseen VAL scenes, (c) scene by scene?

Do NOT run while training is running.
Usage (from src):
    uv run scripts/visualisation/visualize_experts.py
    uv run scripts/visualisation/visualize_experts.py --checkpoints multiscene_final_stage_2_ring.pth --stage1_checkpoint multiscene_best_stage_1.pth
    uv run scripts/visualisation/visualize_experts.py --train_scenes scene_14 --val_scenes scene_18,scene_49 --frames_per_scene 3 --precise_bn
"""

import os
import sys
import json
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

BIN_EDGES = np.array([0.0, 0.02, 0.1, 0.3, 0.6, 1.0, 1.27, 1.6, 2.0, 3.0, np.inf])


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


class BinAccumulator:
    """Accumulates mean-output-per-true-density-bin for several named outputs."""

    def __init__(self, names):
        n = len(BIN_EDGES) - 1
        self.names = list(names)
        self.count = np.zeros(n)
        self.sum_t = np.zeros(n)
        self.sums = {k: np.zeros(n) for k in self.names}

    def add(self, true_density, outputs):
        n = len(self.count)
        idx = np.clip(np.digitize(true_density.ravel(), BIN_EDGES) - 1, 0, n - 1)
        self.count += np.bincount(idx, minlength=n)
        self.sum_t += np.bincount(idx, weights=true_density.ravel(), minlength=n)
        for k, arr in outputs.items():
            self.sums[k] += np.bincount(idx, weights=arr.ravel(), minlength=n)

    def curves(self):
        c = np.maximum(self.count, 1.0)
        return self.sum_t / c, {k: v / c for k, v in self.sums.items()}, self.count


def draw_panels(path, title, panels):
    """panels: list of (title, 2D array, cmap, vmin, vmax)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.0))
    for ax, (t, arr, cmap, vmin, vmax) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_response_curves(path, title, accs):
    """accs: {'train': BinAccumulator, 'val': BinAccumulator}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = next(iter(accs.values())).names
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.0))
    for ax, name in zip(np.atleast_1d(axes), names):
        hi = 0.0
        for split, style in (("train", "o-"), ("val", "s--")):
            if split not in accs:
                continue
            mean_t, outs, cnt = accs[split].curves()
            keep = cnt > 0
            if not keep.any():
                continue
            ax.plot(mean_t[keep], outs[name][keep], style, label=f"{split} scenes")
            hi = max(hi, float(mean_t[keep].max()), float(outs[name][keep].max()))
        ax.axvline(1.273, color="grey", lw=0.8)
        ax.set_xlabel("true density T (pixel-bin mean)")
        ax.set_ylabel(f"mean {name} output")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _snap_int(sample):
    v = sample["snapshot_id"][0]
    v = v.item() if hasattr(v, "item") else v
    return int(v)


def build_filtered_samples(T, entries, n_per_scene, base_seed, skip, only, max_rounds):
    """Like T.build_cached_val_samples, but keeps only frames whose snapshot id passes the filter
    (skip: id % skip != 0, only: id % only == 0). Redraws with new seeds until n_per_scene frames per scene."""
    if not entries:
        return []
    if not skip and not only:
        return T.build_cached_val_samples(entries, n_per_scene, val_seed=base_seed)[0]

    def ok(sid):
        return not (skip and sid % skip == 0) and not (only and sid % only != 0)

    draw = 8 if only else n_per_scene  # 'only' has ~10% hit rate: draw more per round (kept small: samples are large)
    got = {}
    for r in range(max_rounds):
        need = [e for e in entries if len(got.get(e["scene_name"], {})) < n_per_scene]
        if not need:
            break
        for s in T.build_cached_val_samples(need, draw, val_seed=base_seed + 1000 * r)[0]:
            sid = _snap_int(s)
            d = got.setdefault(s["scene_name"], {})
            if ok(sid) and sid not in d and len(d) < n_per_scene:
                d[sid] = s
    for e in entries:
        ids = sorted(got.get(e["scene_name"], {}))
        print(f"[frames] {e['scene_name']}: {len(ids)}/{n_per_scene} -> snapshot ids {ids}")
    return [s for d in got.values() for s in d.values()]


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import train_proposed_multiscene as T

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="comma-separated Stage>=2 checkpoints (default: p0_epoch_10 and final_stage_2_ring)",
    )
    parser.add_argument(
        "--stage1_checkpoint", default=None, help="optional Stage 1 checkpoint: adds the unified-head panel and curve"
    )
    parser.add_argument("--train_scenes", default="scene_14,scene_3,scene_36,scene_42")
    parser.add_argument(
        "--val_scenes", default="scene_18,scene_49,scene_38,scene_23", help="comma-separated, or 'none'"
    )
    parser.add_argument("--frames_per_scene", type=int, default=2)
    parser.add_argument(
        "--skip_multiples_of",
        type=int,
        default=0,
        help="UNSEEN test: drop frames whose snapshot id is a multiple of N (10 = the 21 frames the single-scene run trained on)",
    )
    parser.add_argument(
        "--only_multiples_of",
        type=int,
        default=0,
        help="SEEN control: keep only frames whose snapshot id is a multiple of N",
    )
    parser.add_argument("--max_rounds", type=int, default=60)
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="do not abort if head/neck/backbone keys are missing in a checkpoint",
    )
    parser.add_argument("--patch", default="auto", help="'auto' = the patch with most oracle-dense pixels, or 0/1/2")
    parser.add_argument(
        "--precise_bn",
        action="store_true",
        help="re-estimate layer4/neck BatchNorm statistics over 1 frame of every train scene before evaluating",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--val_seed", type=int, default=1234)
    parser.add_argument("--sparse_arch", choices=["shallow", "deep"], default="shallow")
    parser.add_argument("--out_dir", default=os.path.join(T.src_dir, "diagnostics"))
    args = parser.parse_args()
    assert not (args.skip_multiples_of and args.only_multiples_of), (
        "use either --skip_multiples_of or --only_multiples_of"
    )
    T.SPARSE_ARCH = args.sparse_arch
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = (
        [c.strip() for c in args.checkpoints.split(",")]
        if args.checkpoints
        else [
            os.path.join(T.src_dir, "multiscene_checkpoint_stage_2_p0_epoch_10.pth"),
            os.path.join(T.src_dir, "multiscene_final_stage_2_ring.pth"),
        ]
    )

    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json"), "r") as f:
        cams_cfg = json.load(f)
    print("Building registries (train + val)...")
    train_all = T.build_scene_registry("train", args.k, cams_cfg, {}, validate_setup=False)
    val_all = T.build_scene_registry("validate", args.k, cams_cfg, {}, validate_setup=False)

    def pick(reg, names_csv, split):
        names = [n.strip() for n in names_csv.split(",") if n.strip() and n.strip().lower() != "none"]
        chosen = [e for e in reg if e["scene_name"] in names]
        missing = sorted(set(names) - {e["scene_name"] for e in chosen})
        if missing:
            print(f"[warning] {split} scenes not in the registry (excluded or in another split): {missing}")
        return chosen

    train_sel, val_sel = pick(train_all, args.train_scenes, "train"), pick(val_all, args.val_scenes, "val")
    train_samples = build_filtered_samples(
        T,
        train_sel,
        args.frames_per_scene,
        args.val_seed,
        args.skip_multiples_of,
        args.only_multiples_of,
        args.max_rounds,
    )
    val_samples = build_filtered_samples(
        T,
        val_sel,
        args.frames_per_scene,
        args.val_seed,
        args.skip_multiples_of,
        args.only_multiples_of,
        args.max_rounds,
    )
    for s in train_samples:
        s["split"] = "train"
    for s in val_samples:
        s["split"] = "val"
    samples = train_samples + val_samples
    recal_samples = None
    if args.precise_bn:
        recal_samples, _ = T.build_cached_val_samples(train_all, 1, val_seed=args.val_seed + 7)
    print(
        f"{len(train_samples)} train + {len(val_samples)} val samples"
        f"{f', {len(recal_samples)} BN-recalibration samples' if recal_samples else ''}"
    )

    map_kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(device)
    KMP = float(map_kernel.sum())
    pad = map_kernel.shape[-1] // 2
    print(
        f"[units] map_kernel.max() = {float(map_kernel.max()):.4f}, map_kernel.sum() = {KMP:.4f}, "
        f"mask threshold = {T.oracle_density_threshold():.4f}"
    )

    gt_cache = {}

    def gt(i, p):
        if (i, p) not in gt_cache:
            s = samples[i]
            wld = s["wld_map_paras"].to(device)
            hw_offset = s["hw_random"].to(device)[:, :, p]
            tpatch = T.create_ground_truth_heatmap(s["gt_world_xy"], wld, hw_offset, bev_shape=T.BEV_SHAPE).to(device)
            conv = F.conv2d(tpatch, map_kernel, padding=pad)
            gm = T.oracle_mask_from_conv(conv)
            gt_cache[(i, p)] = (conv[0, 0].cpu().numpy(), gm[0, 0].cpu().numpy())
        return gt_cache[(i, p)]

    def load_model(path):
        from collections import Counter

        model = T.ProposedModel(num_cam=args.k).to(device)
        ck = torch.load(path, map_location=device)
        sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
        # strict=False only tolerates missing/unexpected KEYS, not shape mismatches on keys present
        # in both -- drop those explicitly (e.g. gating_net after an architecture change; dense/sparse
        # expert panels here don't need it) so an unrelated router edit doesn't block this script.
        model_sd = model.state_dict()
        shape_mismatch = [k for k in sd if k in model_sd and sd[k].shape != model_sd[k].shape]
        if shape_mismatch:
            print(
                f"[load] {os.path.basename(path)}: dropping {len(shape_mismatch)} shape-mismatched "
                f"key(s) (e.g. {shape_mismatch[:3]}) -- likely an architecture change since this "
                f"checkpoint was saved; those submodules will use fresh init instead."
            )
            sd = {k: v for k, v in sd.items() if k not in shape_mismatch}
        res = model.load_state_dict(sd, strict=False)

        def grp(ks):
            return dict(Counter(".".join(k.split(".")[:2]) for k in ks))

        print(
            f"[load] {os.path.basename(path)}: missing={len(res.missing_keys)} {grp(res.missing_keys)} | "
            f"unexpected={len(res.unexpected_keys)} {grp(res.unexpected_keys)}"
        )
        bad = [
            k
            for k in res.missing_keys
            if k.startswith(("head.dense_expert", "head.sparse_expert", "neck.", "backbone."))
        ]
        if bad and not args.allow_missing:
            raise RuntimeError(
                f"{len(bad)} model keys missing in {path} (e.g. {bad[:3]}); "
                f"wrong key names or --sparse_arch? use --allow_missing to override"
            )
        model.eval()
        return model

    def recalibrate_bn(model, stage):
        mods = [
            m
            for part in (model.backbone.feature_extractor[-1], model.neck)
            for m in part.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)
        ]
        if not mods:
            return
        old = [m.momentum for m in mods]
        for m in mods:
            m.reset_running_stats()
            m.momentum = None
        model.eval()
        for m in mods:
            m.train()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            for s in recal_samples:
                model(s["images"].to(device), s["ref_points_full"].to(device), s["hw_random"].to(device), stage=stage)
        for m, mom in zip(mods, old):
            m.momentum = mom
        model.eval()

    def forward(model, s, stage):
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out, _ = model(
                s["images"].to(device), s["ref_points_full"].to(device), s["hw_random"].to(device), stage=stage
            )
        return out.float()

    # optional Stage 1 unified head (computed once per sample; converted from GAUSSIAN_SCALE-per-person to kernel units)
    unified = None
    if args.stage1_checkpoint:
        m1 = load_model(args.stage1_checkpoint)
        if args.precise_bn:
            recalibrate_bn(m1, 1)
        unified = [forward(m1, s, 1) * (KMP / T.GAUSSIAN_SCALE) for s in samples]
        del m1
        torch.cuda.empty_cache()

    summary_rows = []
    for ck in ckpts:
        if not os.path.exists(ck):
            print(f"[skip] {ck} not found")
            continue
        label = os.path.splitext(os.path.basename(ck))[0]
        model = load_model(ck)
        if args.precise_bn:
            recalibrate_bn(model, 3)
        names = ["dense", "sparse"] + (["unified"] if unified is not None else [])
        accs = {sp: BinAccumulator(names) for sp in ("train", "val")}
        per_scene = {}
        for i, s in enumerate(samples):
            proj = forward(model, s, 3)
            P = proj.shape[1]
            if args.patch == "auto":
                show_p = int(np.argmax([gt(i, p)[1].sum() for p in range(P)]))
            else:
                show_p = min(int(args.patch), P - 1)
            for p in range(P):
                Tn, gm = gt(i, p)
                with torch.no_grad():
                    D = torch.relu(model.head.dense_expert(proj[:, p]))[0, 0].float().cpu().numpy()
                    S = torch.relu(model.head.sparse_expert(proj[:, p]))[0, 0].float().cpu().numpy()
                U = unified[i][0, p, 0].cpu().numpy() if unified is not None else None
                outs = {"dense": D, "sparse": S}
                if U is not None:
                    outs["unified"] = U
                accs[s["split"]].add(Tn, outs)
                inside = gm > 0.5
                row = {
                    "in_mask_pred": float((D * gm).sum() / KMP),
                    "in_mask_true": float((Tn * gm).sum() / KMP),
                    "sparse_pred": float((S * (1 - gm)).sum() / KMP),
                    "sparse_true": float((Tn * (1 - gm)).sum() / KMP),
                    "corr_dense": pearson(D, Tn),
                    "corr_sparse": pearson(S, Tn),
                    "dense_in": float(D[inside].mean()) if inside.any() else float("nan"),
                    "dense_out": float(D[~inside].mean()),
                    "r0": float(s["wld_map_paras"].reshape(-1)[0]),
                }
                if U is not None:
                    row.update(
                        {
                            "unified_pred": float(U.sum() / KMP),
                            "unified_true": float(Tn.sum() / KMP),
                            "corr_unified": pearson(U, Tn),
                        }
                    )
                per_scene.setdefault((s["scene_name"], s["split"]), []).append(row)

                if p == show_p:
                    vmax = max(float(Tn.max()), 1.0)
                    panels = [
                        (f"gt_mask (dense=white)\ncoverage={100 * gm.mean():.1f}%", gm, "gray", 0, 1),
                        (f"true density T (max {Tn.max():.2f})", Tn, "hot", 0, vmax),
                        (f"dense RAW (max {D.max():.2f})\ncorr with T = {row['corr_dense']:.2f}", D, "hot", 0, vmax),
                        (
                            f"dense MASKED\nmass: pred={row['in_mask_pred']:.1f} vs true={row['in_mask_true']:.1f}",
                            D * gm,
                            "hot",
                            0,
                            vmax,
                        ),
                        (f"sparse RAW (max {S.max():.2f})\ncorr with T = {row['corr_sparse']:.2f}", S, "hot", 0, vmax),
                        (
                            f"sparse MASKED (1-mask)\nmass: pred={row['sparse_pred']:.1f} vs true={row['sparse_true']:.1f}",
                            S * (1 - gm),
                            "hot",
                            0,
                            vmax,
                        ),
                    ]
                    if U is not None:
                        panels.append(
                            (
                                f"Stage 1 unified head\ntotal pred={row['unified_pred']:.1f} vs true={row['unified_true']:.1f}",
                                U,
                                "hot",
                                0,
                                vmax,
                            )
                        )
                    snap = str(s["snapshot_id"][0])
                    fname = f"experts_{label}_{s['split']}_{s['scene_name']}_snap{snap}_p{p}.png"
                    draw_panels(
                        os.path.join(args.out_dir, fname),
                        f"{label} | {s['split']} scene {s['scene_name']} | snapshot {snap} | patch {p} | shared colour scale (0..{vmax:.2f})",
                        panels,
                    )
        # numbers
        print("\n" + "=" * 118)
        print(f"{label}   (means over {P} patches per sample; mass in persons/patch)")
        print("=" * 118)
        hdr = f"{'scene':<10}{'split':<6}{'n':>3} | {'in-mask pred/true':>19} | {'sparse-reg pred/true':>21} | {'corr D':>7}{'corr S':>7} | {'D in':>6}{'D out':>7}"
        if unified is not None:
            hdr += f" | {'unified pred/true':>18}{'corr U':>7}"
        hdr += f" | {'r0':>6}"
        print(hdr)
        for (scene, split), rows in sorted(per_scene.items(), key=lambda kv: (kv[0][1], kv[0][0])):

            def m(k):
                return float(np.nanmean([r[k] for r in rows]))

            line = (
                f"{scene:<10}{split:<6}{len(rows):>3} | {m('in_mask_pred'):>8.1f} /{m('in_mask_true'):>8.1f} | "
                f"{m('sparse_pred'):>9.1f} /{m('sparse_true'):>8.1f} | {m('corr_dense'):>7.2f}{m('corr_sparse'):>7.2f} | "
                f"{m('dense_in'):>6.2f}{m('dense_out'):>7.2f}"
            )
            if unified is not None:
                line += f" | {m('unified_pred'):>8.1f} /{m('unified_true'):>8.1f}{m('corr_unified'):>7.2f}"
            line += f" | {m('r0'):>6.2f}"
            print(line)
            summary_rows.append(
                {"checkpoint": label, "scene": scene, "split": split, **{k: m(k) for k in rows[0].keys()}}
            )
        # response-curve table + figure
        print("\nresponse curve: mean output per true-density bin (train pool | val pool)")
        for split in ("train", "val"):
            mean_t, outs, cnt = accs[split].curves()
            print(f"  [{split}]  bin            count    mean T   " + "   ".join(f"{n:>8}" for n in names))
            for b in range(len(cnt)):
                if cnt[b] == 0:
                    continue
                hi = "+" if not np.isfinite(BIN_EDGES[b + 1]) else f"{BIN_EDGES[b + 1]:.2f}"
                print(
                    f"        [{BIN_EDGES[b]:.2f},{hi:>5})  {int(cnt[b]):>9}  {mean_t[b]:>7.3f}   "
                    + "   ".join(f"{outs[n][b]:>8.3f}" for n in names)
                )
        draw_response_curves(
            os.path.join(args.out_dir, f"expert_response_{label}.png"),
            f"{label}: mean expert output vs true density",
            accs,
        )
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "expert_visualization_summary.json"), "w") as f:
        json.dump(summary_rows, f, indent=1)
    print(f"\nFigures and expert_visualization_summary.json written to {args.out_dir}")


if __name__ == "__main__":
    main()
