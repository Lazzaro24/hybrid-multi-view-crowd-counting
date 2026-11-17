"""
evaluation/summarize_camera_ablation.py

Tables of the camera ablation (RQ3, doc/camera_ablation.md §7) from the result JSONs:
- Tier B: camera_ablation/<model>_<N>v[_vnorm].json (evaluate_paper_protocol.py --num_views N [--view_norm]); the
  5-view column is <model>_5v.json (the models' existing paper-protocol evaluation, copied in; vnorm is a no-op at 5).
- Tier A: runs/camera_ablation/<run>/<N>v/<variant>/result.json (trained at N) and transfer/<variant>_5to<N>v.json.
Intervals: per-scene MAE difference, mean over the test scenes, bootstrap over scenes (as compare_map_quality.py).

Usage (from src/):
    python evaluation/summarize_camera_ablation.py                       # prints the §7 tables (Markdown)
    python evaluation/summarize_camera_ablation.py --out camera_ablation/summary.json
"""

import os
import sys
import json
import argparse

import numpy as np

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = ["baseline_ot", "h1_hom", "h1_casa", "hybrid_hom"]
VARIANTS = ["homography_256", "ca_sa"]
VIEWS = [3, 5, 7, 10]


def bootstrap(diff, n_boot=10000, seed=0):
    """Mean of the per-scene differences and its 95 % bootstrap interval over scenes."""
    diff = np.asarray(diff, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(diff), (n_boot, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(1), [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def scene_key(s):
    return int(s.split("_")[1])


def paired(a, b):
    """b - a over the scenes both per-scene dicts have: (mean, lo, hi, scenes where b is lower, number of scenes)."""
    scenes = sorted(set(a) & set(b), key=scene_key)
    d = np.array([b[s] - a[s] for s in scenes])
    return (*bootstrap(d), int((d < 0).sum()), len(scenes))


def fmt_ci(c, better=True):
    s = f"{c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]"
    return s + (f", {c[3]}/{c[4]}" if better else "")


def fmt(x, spec=".1f", signed=False):
    if x is None:
        return "n/a"
    return format(x, ("+" if signed else "") + spec)


# ==========================================
# Tier B
# ==========================================


def load_tier_b(folder):
    res = {}
    for m in MODELS:
        for n in VIEWS:
            for mode in ("vnorm", "raw"):
                name = f"{m}_{n}v{'_vnorm' if mode == 'vnorm' and n != 5 else ''}.json"
                path = os.path.join(folder, name)
                if os.path.isfile(path):
                    with open(path) as f:
                        res[(m, mode, n)] = json.load(f)
    return res


def tier_b_tables(res):
    out = {}
    for mode in ("vnorm", "raw"):
        print(f"\n### Tier B, {mode}\n")
        print("| model | views | MAE | MSE | NAE | bias | MODA | Precision | Recall | F1 | dense zone |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for m in MODELS:
            for n in VIEWS:
                d = res.get((m, mode, n))
                if d is None:
                    continue
                print(
                    f"| {m if n == VIEWS[0] else ''} | {n} | {d['MAE']:.1f} | {d['MSE']:.1f} | {d['NAE']:.3f} | "
                    f"{fmt(d.get('bias'), signed=True)} | {d['MODA']:.1f} | {d['Precision']:.1f} | {d['Recall']:.1f} | "
                    f"{d['F1']:.1f} | {fmt(d.get('dense_zone_frac'), '.2f')} |"
                )
                out.setdefault(mode, {}).setdefault(m, {})[n] = {
                    k: d.get(k)
                    for k in (
                        "MAE",
                        "MSE",
                        "NAE",
                        "bias",
                        "MODA",
                        "MODP",
                        "Precision",
                        "Recall",
                        "F1",
                        "dense_zone_frac",
                        "samples",
                    )
                }
                out[mode][m][n]["GT_per_sample"] = d["GT"] / d["samples"]

        print(f"\nSteps in N ({mode}): per-scene MAE difference [95 % CI], scenes where the second count is lower\n")
        print("| model | 3 - 5 | 7 - 5 | 10 - 5 | 10 - 7 |")
        print("|---|---|---|---|---|")
        for m in MODELS:
            cells = []
            for a, b in ((5, 3), (5, 7), (5, 10), (7, 10)):
                da, db = res.get((m, mode, a)), res.get((m, mode, b))
                if da and db:
                    c = paired(da["per_scene_MAE"], db["per_scene_MAE"])
                    out.setdefault("steps", {}).setdefault(mode, {}).setdefault(m, {})[f"{b}-{a}"] = c
                    cells.append(fmt_ci(c))
                else:
                    cells.append("–")
            print(f"| {m} | " + " | ".join(cells) + " |")

    print("\nModels against each other at each N (vnorm, per-scene MAE difference)\n")
    print("| | " + " | ".join(f"{n} views" for n in VIEWS) + " |")
    print("|---|" + "---|" * len(VIEWS))
    pairs = [("baseline_ot", m) for m in MODELS[1:]] + [("h1_hom", "h1_casa")]
    for a, b in pairs:
        cells = []
        for n in VIEWS:
            da, db = res.get((a, "vnorm", n)), res.get((b, "vnorm", n))
            if da and db:
                c = paired(da["per_scene_MAE"], db["per_scene_MAE"])
                out.setdefault("models", {}).setdefault(f"{b}-{a}", {})[n] = c
                cells.append(fmt_ci(c, better=False))
            else:
                cells.append("–")
        print(f"| {b} - {a} | " + " | ".join(cells) + " |")

    for m in MODELS:
        if not all((m, "vnorm", n) in res for n in VIEWS):
            continue
        scenes = sorted(res[(m, "vnorm", 5)]["per_scene_MAE"], key=scene_key)
        print(f"\n{m}, vnorm, scene MAE (bias)\n")
        print("| views | " + " | ".join(s.split("_")[1] for s in scenes) + " |")
        print("|---|" + "---|" * len(scenes))
        for n in VIEWS:
            d = res[(m, "vnorm", n)]
            bias = d.get("per_scene_bias", {})
            print(
                f"| {n} | "
                + " | ".join(f"{d['per_scene_MAE'][s]:.0f} ({fmt(bias.get(s), '.0f', True)})" for s in scenes)
                + " |"
            )
    return out


# ==========================================
# Tier A
# ==========================================


def load_tier_a(root):
    trained, transfer = {}, {}
    for v in VARIANTS:
        for n in VIEWS:
            p = os.path.join(root, f"{n}v", v, "result.json")
            if os.path.isfile(p):
                with open(p) as f:
                    trained[(v, n)] = json.load(f)
            p = os.path.join(root, "transfer", f"{v}_5to{n}v.json")
            if os.path.isfile(p):
                with open(p) as f:
                    transfer[(v, n)] = json.load(f)
    return trained, transfer


def tier_a_tables(trained, transfer):
    out = {}

    def cell(m):
        return f"{m['mae']:.1f} ({m['bias']:+.1f})" if m else "–"

    print("\n### Tier A\n")
    print(
        "| fusion | views | trained at N: final (bias) | best-val (epoch) | val final (bias) | transfer raw (bias) | "
        "transfer vnorm (bias) |"
    )
    print("|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        for n in VIEWS:
            r, t = trained.get((v, n)), transfer.get((v, n), {})
            if r is None:
                continue
            print(
                f"| {v if n == VIEWS[0] else ''} | {n} | {cell(r['test_final'])} | {r['test']['mae']:.1f} "
                f"({r['best_epoch']}) | {cell(r['final_val'])} | {cell(t.get('raw'))} | {cell(t.get('vnorm'))} |"
            )
            out.setdefault(v, {})[n] = {
                "test_final": r["test_final"],
                "test_best": r["test"],
                "best_epoch": r["best_epoch"],
                "val_final": r["final_val"],
                "transfer": {k: t[k] for k in ("raw", "vnorm") if k in t},
            }

    print("\nPer-scene MAE differences (trained-at-N final; transfer = 5-view model with vnorm)\n")
    print("| fusion | trained 3 - 5 | trained 7 - 5 | trained 10 - 5 | transfer - trained, 3 | 7 | 10 |")
    print("|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        cells = []
        for n in (3, 7, 10):
            a, b = trained.get((v, 5)), trained.get((v, n))
            cells.append(
                fmt_ci(paired(a["test_final"]["per_scene_mae"], b["test_final"]["per_scene_mae"])) if a and b else "–"
            )
        for n in (3, 7, 10):
            a, t = trained.get((v, n)), transfer.get((v, n), {}).get("vnorm")
            cells.append(fmt_ci(paired(a["test_final"]["per_scene_mae"], t["per_scene_mae"])) if a and t else "–")
        print(f"| {v} | " + " | ".join(cells) + " |")

    print("\nca_sa - homography_256 at each N (per-scene MAE difference, scenes where ca_sa is lower)\n")
    print("| | " + " | ".join(f"{n} views" for n in VIEWS) + " |")
    print("|---|" + "---|" * len(VIEWS))
    for label, get in (
        ("trained at N", lambda v, n: (trained.get((v, n)) or {}).get("test_final")),
        (
            "5-view model, vnorm",
            lambda v, n: (trained.get((v, 5)) or {}).get("test_final")
            if n == 5
            else (transfer.get((v, n)) or {}).get("vnorm"),
        ),
    ):
        cells = []
        for n in VIEWS:
            h, c = get("homography_256", n), get("ca_sa", n)
            if h and c:
                ci = paired(h["per_scene_mae"], c["per_scene_mae"])
                out.setdefault("ca_sa-homography_256", {}).setdefault(label, {})[n] = ci
                cells.append(fmt_ci(ci))
            else:
                cells.append("–")
        print(f"| {label} | " + " | ".join(cells) + " |")
    return out


def main():
    ap = argparse.ArgumentParser(description="Camera ablation tables (doc/camera_ablation.md §7)")
    ap.add_argument("--tier_b", default=os.path.join(src_dir, "camera_ablation"))
    ap.add_argument("--tier_a", default=os.path.join(src_dir, "runs", "camera_ablation", "v1"))
    ap.add_argument("--out", default=None, help="also write the numbers as JSON")
    args = ap.parse_args()
    res_b = load_tier_b(args.tier_b)
    trained, transfer = load_tier_a(args.tier_a)
    if not res_b and not trained:
        sys.exit(f"no results in {args.tier_b} / {args.tier_a}")
    summary = {
        "tier_b": tier_b_tables(res_b) if res_b else None,
        "tier_a": tier_a_tables(trained, transfer) if trained else None,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
