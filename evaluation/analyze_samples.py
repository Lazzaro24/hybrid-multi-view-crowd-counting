"""
evaluation/analyze_samples.py

Analysis of the per-sample paper-protocol results (evaluate_paper_protocol.py --dump_samples; doc/final_evaluations.md
E1-E5). All models were scored on identical samples (same frames and camera draws), so every comparison is paired per
sample.

  E1  headline with 95% intervals from two resampling schemes:
        scene bootstrap   resample the test scenes (with their samples): uncertainty for new scenes;
        frame bootstrap   resample frames (with their draws) inside each scene, scenes fixed;
      the paired MAE / F1 / MODA difference of every model to the references, scenes better (k/10) and a Wilcoxon
      signed-rank test over the per-scene MAE differences; draw noise (spread over the draws of one frame); count
      tracking inside a scene (Pearson r of pred vs GT over the frames, MAE after removing each scene's mean error).
  E2  error inside / outside the oracle dense zone (built from the GT, the same area for every model), and by
      quintiles of the frame's GT count and of its dense-zone share.
  E3  error by weather group and night (brightness < 60), with the weather-by-scene table (confounding check).
  E4  detection threshold sweep (from the results JSONs), each model's best threshold marked as chosen on test.
  E5  router mask quality vs the oracle zone and the routing cost (|err learned| - |err oracle|) of h1_hom / h1_casa.

Usage (from src):
    uv run python evaluation/analyze_samples.py --dir runs/final_evals
Writes <dir>/analysis.json and prints every table.
"""

import os
import csv
import json
import argparse

import numpy as np
from scipy.stats import wilcoxon, pearsonr, spearmanr

MODELS = ["baseline_ot", "baseline_mse", "h1_hom", "h1_casa", "hybrid_hom", "h1_hom_oracle", "h1_casa_oracle"]
REFS = ["baseline_ot", "baseline_mse"]
ROUTER = {"h1_hom": "h1_hom_oracle", "h1_casa": "h1_casa_oracle"}
WEATHER = {
    "clear": ("CLEAR", "EXTRASUNNY"),
    "cloudy": ("CLOUDS", "OVERCAST"),
    "adverse": ("FOGGY", "RAIN", "THUNDER"),
}
NIGHT = 60.0  # mean grey level below which a frame counts as night (eval_occlusion.py --min_brightness)


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])
    return out


class Units:
    """Sample -> scene and frame indices, for the two bootstrap schemes."""

    def __init__(self, d, rng, n_boot):
        self.scenes = list(dict.fromkeys(d["scene"]))
        self.scene_of = np.array([self.scenes.index(s) for s in d["scene"]])
        frame_key = [f"{s}/{int(f)}" for s, f in zip(d["scene"], d["frame"])]
        self.frames = list(dict.fromkeys(frame_key))
        self.frame_of = np.array([self.frames.index(k) for k in frame_key])
        self.frame_scene = np.array([self.scene_of[np.argmax(self.frame_of == j)] for j in range(len(self.frames))])
        self.rng, self.n_boot = rng, n_boot
        S, F = len(self.scenes), len(self.frames)
        # resampling weights per unit, shared by every statistic so the intervals of differences are paired
        self.w_scene = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(n_boot)])
        idx_by_scene = [np.where(self.frame_scene == s)[0] for s in range(S)]
        w = np.zeros((n_boot, F))
        for b in range(n_boot):
            for ids in idx_by_scene:
                np.add.at(w[b], rng.choice(ids, len(ids)), 1)
        self.w_frame = w

    def sums(self, x, level):
        of, n = (self.scene_of, len(self.scenes)) if level == "scene" else (self.frame_of, len(self.frames))
        return np.bincount(of, weights=x, minlength=n)

    def boot(self, stat, cols, level):
        """stat(**{name: summed column}) -> value, on the full data and per bootstrap draw -> (value, lo, hi)."""
        sums = {k: self.sums(v, level) for k, v in cols.items()}
        w = self.w_scene if level == "scene" else self.w_frame
        full = stat(**{k: v.sum() for k, v in sums.items()})
        draws = stat(**{k: w @ v for k, v in sums.items()})
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return float(full), float(lo), float(hi)


def mae_stat(abs_err, n):
    return abs_err / n


def moda_stat(fp, fn, gt):
    return 100 * (1 - (fp + fn) / gt)


def f1_stat(tp, fp, gt):
    p = tp / np.maximum(tp + fp, 1)
    r = tp / np.maximum(gt, 1)
    return 100 * 2 * p * r / np.maximum(p + r, 1e-12)


def fmt(v):
    return f"{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]"


def e1(data, U, out):
    print("\n=== E1: headline, 95% intervals (scene bootstrap | frame bootstrap) ===")
    one = np.ones(len(U.scene_of))
    res = {}
    print(f"{'model':<16}{'MAE scene':>26}{'MAE frame':>26}{'bias':>8}{'F1 scene':>24}{'MODA scene':>28}")
    for m, d in data.items():
        cols_mae = {"abs_err": np.abs(d["err"]), "n": one}
        det = {"tp": d["tp_0.2"], "fp": d["fp_0.2"], "fn": d["fn_0.2"], "gt": d["gt"]}
        r = {
            "MAE": {lv: U.boot(mae_stat, cols_mae, lv) for lv in ("scene", "frame")},
            "bias": float(d["err"].mean()),
            "F1": {lv: U.boot(f1_stat, {k: det[k] for k in ("tp", "fp", "gt")}, lv) for lv in ("scene", "frame")},
            "MODA": {lv: U.boot(moda_stat, {k: det[k] for k in ("fp", "fn", "gt")}, lv) for lv in ("scene", "frame")},
        }
        res[m] = r
        print(
            f"{m:<16}{r['MAE']['scene'][0]:>8.1f} [{r['MAE']['scene'][1]:.1f}, {r['MAE']['scene'][2]:.1f}]"
            f"{'':>4}[{r['MAE']['frame'][1]:.1f}, {r['MAE']['frame'][2]:.1f}]{'':>10}{r['bias']:>+8.1f}"
            f"{r['F1']['scene'][0]:>8.1f} [{r['F1']['scene'][1]:.1f}, {r['F1']['scene'][2]:.1f}]"
            f"{r['MODA']['scene'][0]:>9.1f} [{r['MODA']['scene'][1]:.1f}, {r['MODA']['scene'][2]:.1f}]"
        )
    out["headline"] = res

    print("\n=== E1: paired differences to the references (model - reference); scenes where the model is better ===")
    diffs = {}
    for ref in REFS:
        dr = data[ref]
        for m, d in data.items():
            if m == ref:
                continue
            da = np.abs(d["err"]) - np.abs(dr["err"])
            mae = {lv: U.boot(mae_stat, {"abs_err": da, "n": one}, lv) for lv in ("scene", "frame")}
            per_scene = U.sums(da, "scene") / U.sums(one, "scene")
            better = int((per_scene < 0).sum())
            p = float(wilcoxon(per_scene).pvalue)

            def dstat(tp, fp, gt, tp_r, fp_r):
                return f1_stat(tp, fp, gt) - f1_stat(tp_r, fp_r, gt)

            def dmoda(fp, fn, fp_r, fn_r, gt):
                return moda_stat(fp, fn, gt) - moda_stat(fp_r, fn_r, gt)

            f1d = U.boot(
                dstat,
                {"tp": d["tp_0.2"], "fp": d["fp_0.2"], "gt": d["gt"], "tp_r": dr["tp_0.2"], "fp_r": dr["fp_0.2"]},
                "scene",
            )
            modad = U.boot(
                dmoda,
                {"fp": d["fp_0.2"], "fn": d["fn_0.2"], "fp_r": dr["fp_0.2"], "fn_r": dr["fn_0.2"], "gt": d["gt"]},
                "scene",
            )
            diffs[f"{m} - {ref}"] = {
                "MAE": mae,
                "scenes_better": better,
                "wilcoxon_p": p,
                "F1_scene": f1d,
                "MODA_scene": modad,
            }
            print(
                f"  {m + ' - ' + ref:<32} MAE {fmt(mae['scene'])} (frames {fmt(mae['frame'])}) {better}/10 better, "
                f"Wilcoxon p {p:.3f} | F1 {fmt(f1d)} | MODA {fmt(modad)}"
            )
    out["paired"] = diffs

    print("\n=== E1: draw noise and count tracking inside a scene ===")
    tr = {}
    for m, d in data.items():
        nf = len(U.frames)
        cnt = np.bincount(U.frame_of, minlength=nf)
        pred_f = np.bincount(U.frame_of, weights=d["pred"], minlength=nf) / cnt
        gt_f = np.bincount(U.frame_of, weights=d["gt"], minlength=nf) / cnt
        sd = np.sqrt(np.bincount(U.frame_of, weights=(d["pred"] - pred_f[U.frame_of]) ** 2, minlength=nf) / (cnt - 1))
        sd_gt = np.sqrt(np.bincount(U.frame_of, weights=(d["gt"] - gt_f[U.frame_of]) ** 2, minlength=nf) / (cnt - 1))
        rs, within = [], []
        for s in range(len(U.scenes)):
            j = U.frame_scene == s
            rs.append(pearsonr(pred_f[j], gt_f[j])[0])
            e = pred_f[j] - gt_f[j]
            within.append(np.abs(e - e.mean()))
        tr[m] = {
            "draw_sd_pred": float(sd.mean()),
            "draw_sd_gt": float(sd_gt.mean()),
            "mae_frame_mean_pred": float(np.abs(pred_f - gt_f).mean()),
            "pearson_r_within_scene_mean": float(np.mean(rs)),
            "mae_after_scene_offset": float(np.concatenate(within).mean()),
        }
        t = tr[m]
        print(
            f"  {m:<16} draw SD pred {t['draw_sd_pred']:5.1f} (GT {t['draw_sd_gt']:4.1f}) | MAE of the 5-draw mean "
            f"{t['mae_frame_mean_pred']:5.1f} | within-scene r {t['pearson_r_within_scene_mean']:.2f} | "
            f"MAE after scene offset {t['mae_after_scene_offset']:5.1f}"
        )
    out["tracking"] = tr


def e2(data, U, out):
    print("\n=== E2: error inside / outside the oracle dense zone (GT-based, same area for all models) ===")
    one = np.ones(len(U.scene_of))
    d0 = data["baseline_ot"]
    print(
        f"  dense zone: {100 * d0['oracle_dense_frac'].mean():.1f} % of the map, "
        f"{100 * d0['gt_dense'].sum() / d0['gt'].sum():.1f} % of the people"
    )
    res = {}
    for m, d in data.items():
        ed, es = d["pred_dense"] - d["gt_dense"], d["pred_sparse"] - d["gt_sparse"]
        res[m] = {
            "dense_abs": U.boot(mae_stat, {"abs_err": np.abs(ed), "n": one}, "scene"),
            "dense_bias": float(ed.mean()),
            "sparse_abs": U.boot(mae_stat, {"abs_err": np.abs(es), "n": one}, "scene"),
            "sparse_bias": float(es.mean()),
        }
        r = res[m]
        print(
            f"  {m:<16} dense zone |err| {r['dense_abs'][0]:6.1f} [{r['dense_abs'][1]:.1f}, {r['dense_abs'][2]:.1f}] "
            f"bias {r['dense_bias']:+7.1f} | sparse zone |err| {r['sparse_abs'][0]:6.1f} "
            f"[{r['sparse_abs'][1]:.1f}, {r['sparse_abs'][2]:.1f}] bias {r['sparse_bias']:+7.1f}"
        )
    print("  paired per zone, model - baseline_ot (scene bootstrap), scenes better:")
    for m, d in data.items():
        if m == "baseline_ot":
            continue
        row = []
        for z in ("dense", "sparse"):
            da = np.abs(d[f"pred_{z}"] - d[f"gt_{z}"]) - np.abs(d0[f"pred_{z}"] - d0[f"gt_{z}"])
            v = U.boot(mae_stat, {"abs_err": da, "n": one}, "scene")
            better = int(((U.sums(da, "scene") / U.sums(one, "scene")) < 0).sum())
            res[m][f"{z}_diff_vs_baseline_ot"] = v
            row.append(f"{z} {fmt(v)} {better}/10")
        print(f"    {m:<16} " + " | ".join(row))

    for key, label in (("gt", "GT count"), ("oracle_dense_frac", "dense-zone share")):
        edges = np.percentile(d0[key], [20, 40, 60, 80])
        b = np.digitize(d0[key], edges)
        print(f"\n  by quintile of the frame's {label} (edges {', '.join(f'{e:.3g}' for e in edges)}): MAE (bias)")
        print("    " + f"{'model':<16}" + "".join(f"{f'Q{q + 1}':>16}" for q in range(5)))
        res[f"quintiles_{key}"] = {"edges": edges.tolist()}
        for m, d in data.items():
            cells = [(np.abs(d["err"][b == q]).mean(), d["err"][b == q].mean()) for q in range(5)]
            res[f"quintiles_{key}"][m] = cells
            print("    " + f"{m:<16}" + "".join(f"{a:>8.1f} ({s:+5.0f})" for a, s in cells))
    out["zones"] = res


def e3(data, U, out):
    print("\n=== E3: weather and night ===")
    d0 = data["baseline_ot"]
    w = d0["weather"]
    night = d0["brightness"] < NIGHT
    groups = {g: np.isin(w, labels) for g, labels in WEATHER.items()}
    groups["night"] = night
    groups["day"] = ~night
    print("  frames per scene (one draw per frame):")
    draw0 = d0["draw"] == 0
    print("    " + f"{'scene':<10}" + "".join(f"{g:>9}" for g in groups))
    table = {}
    for s, name in enumerate(U.scenes):
        j = draw0 & (U.scene_of == s)
        table[name] = {g: int((m & j).sum()) for g, m in groups.items()}
        print("    " + f"{name:<10}" + "".join(f"{table[name][g]:>9}" for g in groups))
    res = {"frames_by_scene": table}
    one = np.ones(len(U.scene_of))
    for g, msk in groups.items():
        res[g] = {}
        line = f"  {g:<8} ({int((msk & draw0).sum())} frames)"
        for m, d in data.items():
            if m.endswith("_oracle"):
                continue
            mae = U.boot(mae_stat, {"abs_err": np.abs(d["err"]) * msk, "n": one * msk}, "frame")
            p = d["tp_0.2"][msk].sum() / max(d["tp_0.2"][msk].sum() + d["fp_0.2"][msk].sum(), 1)
            r = d["tp_0.2"][msk].sum() / d["gt"][msk].sum()
            res[g][m] = {"MAE_frame_boot": mae, "bias": float(d["err"][msk].mean()), "F1": 100 * 2 * p * r / (p + r)}
            line += f" | {m} {mae[0]:.1f} [{mae[1]:.1f}, {mae[2]:.1f}] F1 {res[g][m]['F1']:.1f}"
        print(line)
    print("  paired MAE difference to baseline_ot per group (frame bootstrap):")
    for m, d in data.items():
        if m.endswith("_oracle") or m == "baseline_ot":
            continue
        da = np.abs(d["err"]) - np.abs(d0["err"])
        row = []
        for g, msk in groups.items():
            v = U.boot(mae_stat, {"abs_err": da * msk, "n": one * msk}, "frame")
            res[g][m]["diff_vs_baseline_ot"] = v
            row.append(f"{g} {fmt(v)}")
        print(f"    {m:<14} " + " | ".join(row))
    out["weather"] = res


def e4(jsons, out):
    print("\n=== E4: detection threshold sweep (MODA / F1 / precision / recall) ===")
    res = {}
    for m, j in jsons.items():
        sw = {float(t): v for t, v in j["threshold_sweep"].items()}
        ts = sorted(sw)
        best = max(ts, key=lambda t: sw[t]["MODA"])
        res[m] = {"sweep": {str(t): sw[t] for t in ts}, "best_moda_threshold_chosen_on_test": best}
        print(
            f"  {m:<16} "
            + "  ".join(
                f"{t:g}: {sw[t]['MODA']:7.1f}/{sw[t]['F1']:4.1f}/{sw[t]['Precision']:4.1f}/{sw[t]['Recall']:4.1f}"
                for t in ts
            )
            + f"  | best MODA at {best:g} (chosen on test)"
        )
    for metric in ("MODA", "F1"):
        for t in sorted(float(x) for x in next(iter(jsons.values()))["threshold_sweep"]):
            rank = sorted(jsons, key=lambda m: -jsons[m]["threshold_sweep"][str(t)][metric])
            res.setdefault(f"ranking_{metric}", {})[str(t)] = rank
        print(
            f"  ranking by {metric}: " + " | ".join(f"{t}: {', '.join(r)}" for t, r in res[f"ranking_{metric}"].items())
        )
    out["thresholds"] = res


def e5(data, U, out):
    print("\n=== E5: router of h1_hom / h1_casa vs the oracle zone, and the routing cost ===")
    res = {}
    one = np.ones(len(U.scene_of))
    for m, orc in ROUTER.items():
        d, o = data[m], data[orc]
        cost = np.abs(d["err"]) - np.abs(o["err"])
        r = {
            "router_dense_frac": float(d["router_dense_frac"].mean()),
            "oracle_dense_frac": float(d["oracle_dense_frac"].mean()),
            "iou": float(np.nanmean(d["router_iou"])),
            "precision": float(np.nanmean(d["router_prec"])),
            "recall": float(np.nanmean(d["router_rec"])),
            "soft_dice": float(np.nanmean(d["soft_dice"])),
            "routing_cost_MAE": U.boot(mae_stat, {"abs_err": cost, "n": one}, "scene"),
            "cost_vs_iou_spearman": float(spearmanr(cost, d["router_iou"], nan_policy="omit")[0]),
        }
        for z in ("dense", "sparse"):
            cz = np.abs(d[f"pred_{z}"] - d[f"gt_{z}"]) - np.abs(o[f"pred_{z}"] - o[f"gt_{z}"])
            r[f"cost_{z}_zone"] = U.boot(mae_stat, {"abs_err": cz, "n": one}, "scene")
            r[f"bias_{z}_learned"] = float((d[f"pred_{z}"] - d[f"gt_{z}"]).mean())
            r[f"bias_{z}_oracle"] = float((o[f"pred_{z}"] - o[f"gt_{z}"]).mean())
        per_scene = {}
        for s, name in enumerate(U.scenes):
            j = U.scene_of == s
            per_scene[name] = {
                "iou": float(np.nanmean(d["router_iou"][j])),
                "router_frac": float(d["router_dense_frac"][j].mean()),
                "oracle_frac": float(d["oracle_dense_frac"][j].mean()),
                "cost": float(cost[j].mean()),
            }
        r["per_scene"] = per_scene
        res[m] = r
        print(
            f"  {m}: dense share router {100 * r['router_dense_frac']:.1f} % vs oracle {100 * r['oracle_dense_frac']:.1f} % | "
            f"IoU {r['iou']:.2f}, precision {r['precision']:.2f}, recall {r['recall']:.2f}, soft Dice {r['soft_dice']:.2f}"
        )
        print(
            f"    routing cost (MAE learned - oracle) {fmt(r['routing_cost_MAE'])}; dense zone {fmt(r['cost_dense_zone'])}, "
            f"sparse zone {fmt(r['cost_sparse_zone'])}; Spearman(cost, IoU) {r['cost_vs_iou_spearman']:+.2f}"
        )
        print(
            f"    zone bias learned / oracle: dense {r['bias_dense_learned']:+.1f} / {r['bias_dense_oracle']:+.1f}, "
            f"sparse {r['bias_sparse_learned']:+.1f} / {r['bias_sparse_oracle']:+.1f}"
        )
        print(
            "    per scene (IoU, router %, oracle %, cost): "
            + "; ".join(
                f"{k} {v['iou']:.2f} {100 * v['router_frac']:.1f}/{100 * v['oracle_frac']:.1f} {v['cost']:+.1f}"
                for k, v in per_scene.items()
            )
        )
    out["router"] = res


def main():
    ap = argparse.ArgumentParser(description="Analysis of the per-sample paper-protocol results (E1-E5).")
    ap.add_argument("--dir", default="runs/final_evals")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data = {m: load(os.path.join(args.dir, f"{m}.samples.csv")) for m in MODELS}
    jsons = {m: json.load(open(os.path.join(args.dir, f"{m}.json"))) for m in MODELS}
    d0 = data["baseline_ot"]
    for m, d in data.items():  # identical samples: same scenes, frames, draws and GT counts
        assert len(d["gt"]) == len(d0["gt"]) and (d["gt"] == d0["gt"]).all() and (d["scene"] == d0["scene"]).all(), m
        assert abs(np.abs(d["err"]).mean() - jsons[m]["MAE"]) < 1e-6, m
    U = Units(d0, np.random.default_rng(args.seed), args.bootstrap)
    print(f"{len(d0['gt'])} samples, {len(U.frames)} frames, {len(U.scenes)} scenes; {args.bootstrap} resamples")
    out = {"samples": len(d0["gt"]), "frames": len(U.frames), "scenes": U.scenes, "bootstrap": args.bootstrap}
    e1(data, U, out)
    e2(data, U, out)
    e3(data, U, out)
    e4(jsons, out)
    e5(data, U, out)
    with open(os.path.join(args.dir, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved {os.path.join(args.dir, 'analysis.json')}")


if __name__ == "__main__":
    main()
