"""
eval_neck.py

Neck-focused evaluation of one or more Stage >= 1 checkpoints on the training-time validation cache (same val_seed,
frames and crops as training validation, precise-BN like training). No training of the model itself. Works for the
legacy neck (one deformable self-attention + one cross-attention) and the BEVFormer-style encoder neck alike, so the
two can be compared on identical patches.

Reports
  [probe]   head-independent: ridge regression from the neck output (64 ch) to the true density T, fitted on TRAIN
            scenes, scored on VAL scenes. 1x1 = per-cell linear read-out, 5x5 = linear read-out of a 5x5 neighbourhood
            (receptive field of the Stage 1 throwaway head). R2 / corr / AUROC / count MAE of the read-out.
  [input]   how much of a representation changes with the input images at a fixed BEV cell:
              in_share = mean over cells of Var_samples(x) / total variance.
            0 = identical for every image (a learned constant), 1 = no position-only component at all.
            Measured on every cross-attention's query input (what predicts its offsets/weights) and on the neck output.
  [attn]    per deformable-attention module: normalised attention entropy (1 = uniform over its K points, 0 = one point)
            and mean |offset| in pixels of the map it samples (BEV px for self-attention, feature px for cross-attention).
  [coh]     spatial coherence of the neck output: round-trip error ||x - up(down(x, k))||^2 / Var(x) for k = 2, 4, 8
            (lower = smoother); the true density's own value is printed as a reference.
  [head]    the checkpoint's own head, as in training validation: Stage 1 -> throwaway head count MAE and val loss;
            Stage >= 2 -> oracle-routed fused count MAE, dense pedestal (bg) and dense corr_in.

Do NOT run while training is running (GPU memory).
Usage (from src):
    uv run python evaluation/eval_neck.py --checkpoints multiscene_best_stage_1_s8_dout0.7_cw0.01_f10_a0.7.pth,multiscene_best_stage_2_s8_dout0.7_cw0.01_f10_a0.7.pth
"""

import os
import re
import sys
import json
import math
import argparse
from collections import defaultdict

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))


def auroc(score, label):
    score = np.asarray(score).ravel()
    label = np.asarray(label).ravel().astype(bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(cnt) - (cnt - 1) / 2.0
    ranks = avg_rank[inv]
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def infer_stage(path):
    m = re.search(r"stage_(\d)", os.path.basename(path))
    return int(m.group(1)) if m else 2


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import train_proposed_multiscene as T

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoints", required=True, help="comma-separated; 'path@stage' overrides the stage read from the file name"
    )
    ap.add_argument("--val_frames", type=int, default=10, help="frames per val scene (training val cache: 10)")
    ap.add_argument("--probe_frames", type=int, default=2, help="frames per TRAIN scene used to fit the probe")
    ap.add_argument("--val_seed", type=int, default=T.VAL_SEED_DEFAULT)
    ap.add_argument("--no_precise_bn", action="store_true")
    ap.add_argument("--precise_bn_frames", type=int, default=1)
    ap.add_argument("--probe_radius", type=int, default=2, help="neighbourhood probe: (2r+1)^2 cells")
    ap.add_argument("--probe_lambda", type=float, default=1e-3, help="ridge strength, relative to mean feature energy")
    ap.add_argument("--empty_thr", type=float, default=0.02)
    ap.add_argument("--crowd_thr", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out_dir", default=os.path.join(T.src_dir, "diagnostics", "neck_eval"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    # ---------------------------------------------------------------- samples (identical to training validation)
    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json"), "r") as f:
        cams_cfg = json.load(f)
    train_reg = T.build_scene_registry("train", args.k, cams_cfg, {}, validate_setup=False)
    val_reg = T.build_scene_registry("validate", args.k, cams_cfg, {}, validate_setup=False)
    val_samples, _ = T.build_cached_val_samples(val_reg, args.val_frames, val_seed=args.val_seed)
    probe_samples, _ = T.build_cached_val_samples(train_reg, args.probe_frames, val_seed=args.val_seed + 101)
    bn_samples = None
    if not args.no_precise_bn:
        bn_samples, _ = T.build_cached_val_samples(train_reg, args.precise_bn_frames, val_seed=args.val_seed + 7)

    kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(device)
    KMP = float(kernel.sum())
    pad = kernel.shape[-1] // 2
    mse_criterion = T.GaussianMSE_norm(scale=T.GAUSSIAN_SCALE).to(device).float()

    def ground_truth(s, p):
        wld = s["wld_map_paras"].to(device)
        hw = s["hw_random"].to(device)[:, :, p]
        dots = T.create_ground_truth_heatmap(s["gt_world_xy"], wld, hw, bev_shape=T.BEV_SHAPE).to(device)
        conv = F.conv2d(dots, kernel, padding=pad)
        zone = T.oracle_mask_from_conv(conv)
        n_dots = T.extract_target_coordinates(s["gt_world_xy"], wld, hw, bev_shape=T.BEV_SHAPE).shape[2]
        return dots, conv, zone, n_dots

    print("[gt] building ground truth ...")
    gt_val = {(i, p): ground_truth(s, p) for i, s in enumerate(val_samples) for p in range(s["hw_random"].shape[-1])}
    gt_probe = {
        (i, p): ground_truth(s, p)[1] for i, s in enumerate(probe_samples) for p in range(s["hw_random"].shape[-1])
    }
    print(
        f"[data] val: {len(val_samples)} samples / {len(gt_val)} patches | probe-fit (train scenes): "
        f"{len(probe_samples)} samples / {len(gt_probe)} patches"
    )

    # ---------------------------------------------------------------- spatial coherence reference: the true density
    def round_trip(x, k):
        # x: [B, C, H, W]; error of a k-times average-pool + nearest upsample, relative to x's variance
        H, W = x.shape[-2:]
        y = F.interpolate(F.avg_pool2d(x, k, ceil_mode=True), scale_factor=k, mode="nearest")[..., :H, :W]
        var = x.var(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        return (((x - y) ** 2).mean(dim=(-2, -1), keepdim=True) / var).mean().item()

    KS = (2, 4, 8)
    coh_gt = {k: float(np.mean([round_trip(v[1], k) for v in gt_val.values() if v[1].sum() > 0])) for k in KS}

    # ---------------------------------------------------------------- model helpers
    def load_model(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        model = T.build_model_for_checkpoint(ck, num_cam=args.k).to(device)
        msd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k not in msd or msd[k].shape == v.shape}
        res = model.load_state_dict(sd, strict=False)
        bad = [
            k
            for k in res.missing_keys
            if k.startswith(("neck.", "backbone.", "throwaway_head.", "head.dense_expert", "head.sparse_expert"))
        ]
        if bad:
            raise RuntimeError(f"{len(bad)} required keys missing in {path} (e.g. {bad[:3]})")
        model.eval()
        return model

    def recalibrate_bn(model, stage):
        mods = [m for m in model.neck.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        if not mods or bn_samples is None:
            return
        old = [m.momentum for m in mods]
        for m in mods:
            m.reset_running_stats()
            m.momentum = None
            m.train()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
            for s in bn_samples:
                model(s["images"].to(device), s["ref_points_full"].to(device), s["hw_random"].to(device), stage=stage)
        for m, mom in zip(mods, old):
            m.momentum = mom
        model.eval()

    class Moments:
        """Running per-position moments over samples of a [L, C] representation (float64 sums on the device)."""

        def __init__(self):
            self.n = 0
            self.s = self.s2 = None

        def add(self, x):  # x: [L, C]
            x = x.double()
            if self.s is None:
                self.s, self.s2 = torch.zeros_like(x), torch.zeros_like(x)
            self.s += x
            self.s2 += x * x
            self.n += 1

        def in_share(self):
            if self.n < 2:
                return float("nan")
            mean = self.s / self.n
            within = (self.s2 / self.n - mean * mean).clamp_min(0).sum()  # sum over cells/ch of Var_samples
            grand = self.s.sum(0) / (self.n * self.s.shape[0])  # per-channel grand mean
            total = (
                (self.s2.sum(0) / self.n - 2 * grand * self.s.sum(0) / self.n + self.s.shape[0] * grand * grand)
                .clamp_min(0)
                .sum()
            )  # sum of (x - grand_c)^2 / n
            return float(within / total.clamp_min(1e-30))

    def attention_modules(neck):
        out = []
        for name, m in neck.named_modules():
            if hasattr(m, "offset_predictor") and hasattr(m, "weight_predictor") and hasattr(m, "num_points"):
                kind = "cross" if "cross" in type(m).__name__.lower() else "self"
                out.append((name, kind, m))
        return out

    # ---------------------------------------------------------------- evaluation of one checkpoint
    def evaluate(path, stage):
        label = os.path.splitext(os.path.basename(path))[0]
        model = load_model(path)
        if not args.no_precise_bn:
            recalibrate_bn(model, 1 if stage == 1 else 2)
        head_stage = 1 if stage == 1 else 2
        neck = model.neck
        neck_desc = getattr(neck, "describe", lambda: "legacy: 1x deformable self-attention + 1x cross-attention")()
        n_params = sum(p.numel() for p in neck.parameters())

        # hooks: neck output, cross-attention query inputs, attention logits/offsets
        captured = {"neck": [], "q": defaultdict(list)}
        with torch.no_grad():
            captured["feat_w"] = model.backbone(val_samples[0]["images"][0, :1].to(device)).shape[-1]
        attn = attention_modules(neck)
        attn_acc = {name: {"ent": 0.0, "off": 0.0, "n": 0} for name, _, _ in attn}
        handles = [neck.register_forward_hook(lambda m, i, o: captured["neck"].append(o.detach()))]
        for name, kind, m in attn:
            if kind == "cross":

                def pre(mod, a, kw, name=name):
                    q = kw.get("bev_queries", kw.get("query", a[0] if a else None))
                    captured["q"][name].append(q.detach())

                handles.append(m.register_forward_pre_hook(pre, with_kwargs=True))

            def w_hook(mod, i, o, name=name, parent=m):
                logits = o.detach().float().reshape(-1, parent.num_points)
                prob = logits.softmax(-1)
                ent = -(prob * prob.clamp_min(1e-12).log()).sum(-1) / math.log(parent.num_points)
                a = attn_acc[name]
                a["ent"] += ent.mean().item()
                a["n"] += 1

            handles.append(m.weight_predictor.register_forward_hook(w_hook))

            def o_hook(mod, i, o, name=name, parent=m, kind=kind):
                off = o.detach().float().reshape(-1, 2)
                if getattr(parent, "offset_units", "norm0.1") == "px":
                    px = off.norm(dim=-1)
                else:  # legacy: offset * 0.1 in grid_sample coordinates ([-1, 1] spans the map)
                    half = (T.BEV_SHAPE[1] / 2.0) if kind == "self" else (captured.get("feat_w", 160) / 2.0)
                    px = (off * 0.1 * half).norm(dim=-1)
                attn_acc[name]["off"] += px.mean().item()

            handles.append(m.offset_predictor.register_forward_hook(o_hook))

        moments = {"neck_out": Moments()}
        coh = {k: [] for k in KS}
        r = args.probe_radius
        D = 64 * (2 * r + 1) ** 2
        centre = [c * (2 * r + 1) ** 2 + ((2 * r + 1) ** 2) // 2 for c in range(64)]

        def run(s):
            captured["neck"].clear()
            captured["q"].clear()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
                out, _ = model(  # noqa: F821  (model is deleted only after the last run() call)
                    s["images"].to(device), s["ref_points_full"].to(device), s["hw_random"].to(device), stage=head_stage
                )
            return out.float()

        def design(x):  # x: [1, 64, H, W] -> [H*W, D+1]
            u = F.unfold(x.float(), kernel_size=2 * r + 1, padding=r)[0].t()
            return torch.cat([u, torch.ones_like(u[:, :1])], dim=1)

        # ---- 1) probe fit on train scenes
        XtX = torch.zeros(D + 1, D + 1, dtype=torch.float64, device=device)
        XtY = torch.zeros(D + 1, dtype=torch.float64, device=device)
        for i, s in enumerate(probe_samples):
            if i % 20 == 0:
                print(f"[probe-fit] {label}: sample {i + 1}/{len(probe_samples)}")
            run(s)
            for p, f in enumerate(captured["neck"]):
                X = design(f)
                y = gt_probe[(i, p)].reshape(-1, 1).float()
                XtX += (X.t() @ X).double()
                XtY += (X.t() @ y).double().squeeze(1)

        def solve(idx):
            A = XtX[idx][:, idx]
            reg = args.probe_lambda * torch.diagonal(A)[:-1].mean()
            A = A + reg * torch.diag(
                torch.cat(
                    [
                        torch.ones(len(idx) - 1, device=device, dtype=A.dtype),
                        torch.zeros(1, device=device, dtype=A.dtype),
                    ]
                )
            )
            return torch.linalg.solve(A, XtY[idx])

        idx_full = list(range(D + 1))
        idx_1x1 = centre + [D]
        w_full, w_1x1 = solve(idx_full).float(), solve(idx_1x1).float()

        # ---- 2) validation pass
        for name, *_ in attn:
            attn_acc[name].update(ent=0.0, off=0.0, n=0)
        rows = []
        for i, s in enumerate(val_samples):
            if i % 20 == 0:
                print(f"[val] {label}: sample {i + 1}/{len(val_samples)}")
            out = run(s)
            for p, f in enumerate(captured["neck"]):
                dots, conv, zone, n_dots = gt_val[(i, p)]
                Tn = conv[0, 0]
                moments["neck_out"].add(f[0].flatten(1).t())
                for name, qs in captured["q"].items():
                    moments.setdefault(f"q_in:{name}", Moments()).add(qs[p][0])
                for k in KS:
                    coh[k].append(round_trip(f.float(), k))
                X = design(f)
                pred_full = (X @ w_full).reshape(Tn.shape)
                pred_1x1 = (X[:, idx_1x1] @ w_1x1).reshape(Tn.shape)
                t_np = Tn.cpu().numpy()
                empty, crowd = t_np < args.empty_thr, t_np >= args.crowd_thr
                sel = empty | crowd
                row = {"scene": s["scene_name"], "t_tot": float(t_np.sum() / KMP), "n_dots": n_dots}
                for tag, pr in (("p5", pred_full), ("p1", pred_1x1)):
                    pr_np = pr.cpu().numpy()
                    row[f"{tag}_sse"] = float(((pr_np - t_np) ** 2).sum())
                    row[f"{tag}_corr"] = pearson(pr_np, t_np)
                    row[f"{tag}_auc"] = auroc(pr_np[sel], crowd[sel]) if (empty.any() and crowd.any()) else float("nan")
                    row[f"{tag}_tot"] = float(pr_np.sum() / KMP)
                row["sst"] = float(((t_np - t_np.mean()) ** 2).sum())
                row["t_sum"], row["t_sum2"], row["n_px"] = float(t_np.sum()), float((t_np**2).sum()), t_np.size

                # head metrics, as in training validation
                pp = out[:, p]
                if head_stage == 1:
                    row["h_tot"] = pp.sum().item() / T.GAUSSIAN_SCALE
                    row["h_err"] = row["h_tot"] - n_dots
                    row["h_loss"] = mse_criterion(pp, dots, kernel)[0].item()
                else:
                    Dm, Sm = pp[0, 0], pp[0, 1]
                    gm = zone[0, 0]
                    row["h_tot"] = float((Dm * gm + Sm * (1 - gm)).sum() / KMP)
                    row["h_err"] = row["h_tot"] - row["t_tot"]
                    e = torch.from_numpy(empty).to(device)
                    row["bg_sum"], row["bg_n"] = float(Dm[e].sum()), int(e.sum())
                    inside = gm > 0.5
                    row["ci"] = (Dm[inside].double().cpu().numpy(), Tn[inside].double().cpu().numpy())
                rows.append(row)
        for h in handles:
            h.remove()

        # ---- 3) summary
        def pooled_r2(tag):
            sse = sum(r_[f"{tag}_sse"] for r_ in rows)
            n = sum(r_["n_px"] for r_ in rows)
            mu = sum(r_["t_sum"] for r_ in rows) / n
            sst = sum(r_["t_sum2"] for r_ in rows) - n * mu * mu
            return 1.0 - sse / sst

        res = {
            "checkpoint": label,
            "stage": stage,
            "neck": neck_desc,
            "neck_params": n_params,
            "n_val_patches": len(rows),
            "precise_bn": not args.no_precise_bn,
        }
        for tag in ("p1", "p5"):
            res[f"probe_{tag}"] = {
                "r2": pooled_r2(tag),
                "corr": float(np.nanmean([r_[f"{tag}_corr"] for r_ in rows])),
                "auroc": float(np.nanmean([r_[f"{tag}_auc"] for r_ in rows])),
                "count_mae": float(np.mean([abs(r_[f"{tag}_tot"] - r_["t_tot"]) for r_ in rows])),
                "count_bias": float(np.mean([r_[f"{tag}_tot"] - r_["t_tot"] for r_ in rows])),
            }
        res["input_share"] = {k: v.in_share() for k, v in moments.items()}
        res["attention"] = {
            name: {
                "kind": kind,
                "entropy": attn_acc[name]["ent"] / max(attn_acc[name]["n"], 1),
                "offset_px": attn_acc[name]["off"] / max(attn_acc[name]["n"], 1),
            }
            for name, kind, _ in attn
        }
        res["coherence"] = {f"k{k}": float(np.mean(v)) for k, v in coh.items()}
        res["coherence_gt"] = {f"k{k}": v for k, v in coh_gt.items()}
        errs = np.array([r_["h_err"] for r_ in rows])
        res["head"] = {
            "kind": "stage1 throwaway" if head_stage == 1 else "oracle-fused experts",
            "count_mae": float(np.abs(errs).mean()),
            "count_bias": float(errs.mean()),
        }
        if head_stage == 1:
            res["head"]["val_loss"] = float(np.mean([r_["h_loss"] for r_ in rows]))
        else:
            res["head"]["dense_bg"] = sum(r_["bg_sum"] for r_ in rows) / max(sum(r_["bg_n"] for r_ in rows), 1)
            xs = np.concatenate([r_["ci"][0] for r_ in rows]) if rows else np.zeros(0)
            ys = np.concatenate([r_["ci"][1] for r_ in rows]) if rows else np.zeros(0)
            res["head"]["dense_corr_in"] = pearson(xs, ys) if xs.size > 1 else float("nan")
        per_scene = defaultdict(list)
        for r_ in rows:
            per_scene[r_["scene"]].append(r_)
        res["per_scene"] = {
            sc: {
                "probe_p5_corr": float(np.nanmean([x["p5_corr"] for x in v])),
                "probe_p5_count_mae": float(np.mean([abs(x["p5_tot"] - x["t_tot"]) for x in v])),
                "head_count_mae": float(np.mean([abs(x["h_err"]) for x in v])),
                "mean_true": float(np.mean([x["t_tot"] for x in v])),
            }
            for sc, v in sorted(per_scene.items(), key=lambda kv: int(kv[0].split("_")[1]))
        }
        del model
        torch.cuda.empty_cache()
        return res

    def report(res):
        bar = "=" * 100
        print(
            f"\n{bar}\n{res['checkpoint']}  (stage {res['stage']})\n  neck: {res['neck']} | "
            f"{res['neck_params']:,} params | {res['n_val_patches']} val patches | precise-BN {res['precise_bn']}\n{bar}"
        )
        print("[probe]  linear read-out of the neck output -> true density (fit: train scenes, score: val scenes)")
        for tag, nm in (("p1", "1x1"), ("p5", f"{2 * args.probe_radius + 1}x{2 * args.probe_radius + 1}")):
            p = res[f"probe_{tag}"]
            print(
                f"   {nm:<4} R2 {p['r2']:7.3f} | corr {p['corr']:6.3f} | AUROC crowd/empty {p['auroc']:6.3f} | "
                f"count MAE {p['count_mae']:7.2f} (bias {p['count_bias']:+7.2f})"
            )
        print("[input]  in_share = fraction of variance that changes with the images at a fixed BEV cell")
        for k, v in res["input_share"].items():
            print(f"   {k:<45} {v:7.4f}")
        print("[attn]   entropy (1 = uniform over K points) | mean |offset| in px of the sampled map")
        for name, a in res["attention"].items():
            print(f"   {name:<45} {a['kind']:<6} entropy {a['entropy']:6.3f} | offset {a['offset_px']:7.2f} px")
        c, g = res["coherence"], res["coherence_gt"]
        print(
            "[coh]    round-trip error (lower = smoother):  "
            + "  ".join(f"k={k[1:]}: {c[k]:.3f} (GT {g[k]:.3f})" for k in c)
        )
        h = res["head"]
        extra = (
            f"val loss {h['val_loss']:.4f}"
            if "val_loss" in h
            else f"dense bg {h['dense_bg']:.3f} | dense corr_in {h['dense_corr_in']:.3f}"
        )
        print(f"[head]   {h['kind']}: count MAE {h['count_mae']:.2f} (bias {h['count_bias']:+.2f}) | {extra}")
        print(
            "[scene]  "
            + " | ".join(
                f"{sc}: probe corr {v['probe_p5_corr']:.2f}, head MAE {v['head_count_mae']:.1f}"
                for sc, v in res["per_scene"].items()
            )
        )

    results = []
    for item in args.checkpoints.split(","):
        item = item.strip()
        if not item:
            continue
        path, _, st = item.partition("@")
        stage = int(st) if st else infer_stage(path)
        res = evaluate(path, stage)
        report(res)
        results.append(res)
        out = os.path.join(args.out_dir, f"neck_eval_{res['checkpoint']}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[saved] {out}")

    if len(results) > 1:
        print("\n" + "=" * 100 + "\nSUMMARY")
        print(f"{'checkpoint':<58}{'probe5 R2':>10}{'corr':>7}{'AUROC':>7}{'cntMAE':>8}{'head MAE':>10}")
        for r_ in results:
            p = r_["probe_p5"]
            print(
                f"{r_['checkpoint'][:57]:<58}{p['r2']:>10.3f}{p['corr']:>7.3f}{p['auroc']:>7.3f}"
                f"{p['count_mae']:>8.2f}{r_['head']['count_mae']:>10.2f}"
            )


if __name__ == "__main__":
    main()
