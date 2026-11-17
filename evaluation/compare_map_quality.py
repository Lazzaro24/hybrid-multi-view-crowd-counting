"""
compare_map_quality.py

Paired comparison of map-quality CSVs (eval_map_quality.py, diagnostics/map_quality/) against a reference model. All
models are scored on the same held-out frames, so the comparison is paired per scene: per split, the pooled metrics of
every model (MAE, bias, F1, MODA, GAME3, far mass > 1 m / 2 m) and, against the reference, the per-scene difference of
MAE and far > 2 m: per scene the mean |pred - gt| (or far2m) over that scene's frames, d = model - reference per scene,
95% interval = 2.5 / 97.5 percentiles of the means of 5000 bootstrap resamples of d (numpy default_rng(0)), and the
number of scenes where the model is better.

Names: a CSV stem ('proposed__casa_dense'), a summary name ('ablation_v2:ca_sa', 'proposed:casa_dense'), a bare
proposed name ('casa_dense') or a path.

Usage (from src):
    uv run python evaluation/compare_map_quality.py --ref ablation_v2:ca_sa casa_dense casa_D1_s42 casa_D1_s42_final
"""

import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
# src/ and every script folder, so scripts can import each other by module name
for _d in ("", "training_scripts", "evaluation", "scripts/diagnosis", "scripts/visualisation"):
    sys.path.append(os.path.normpath(os.path.join(src_dir, _d)))

from eval_map_quality import OUT, pooled


def resolve(name):
    candidates = [
        name,
        os.path.join(OUT, name),
        os.path.join(OUT, name + ".csv"),
        os.path.join(OUT, name.replace(":", "__") + ".csv"),
        os.path.join(OUT, f"proposed__{name}.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"no map-quality CSV for {name!r} (tried {candidates[1:]})")


def load(name):
    with open(resolve(name)) as f:
        return list(csv.DictReader(f))


def per_scene(rows, key):
    by = defaultdict(list)
    for r in rows:
        v = abs(float(r["pred"]) - float(r["gt"])) if key == "mae" else float(r[key])
        by[r["scene"]].append(v)
    return {s: float(np.mean(v)) for s, v in by.items()}


def paired(rows, ref_rows, key, n_boot=5000):
    a, b = per_scene(rows, key), per_scene(ref_rows, key)
    scenes = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in scenes])
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi, int((d < 0).sum()), len(d)


def frame_keys(rows):
    return {(r["scene"], r["snapshot"]) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--splits", default="validate,test")
    args = ap.parse_args()

    ref_all = load(args.ref)
    models = [(n, load(n)) for n in args.names]
    for split in args.splits.split(","):
        ref = [r for r in ref_all if r["split"] == split]
        print(f"\n== {split} (reference {args.ref}, {len(ref)} frames, {len({r['scene'] for r in ref})} scenes)")
        print(
            f"{'model':28s} {'MAE':>6s} {'bias':>7s} {'F1':>5s} {'MODA':>6s} {'GAME3':>6s} {'far1m':>6s} {'far2m':>6s}"
            f" | {'dMAE [95% CI]':>24s} {'better':>6s} | {'dfar2m [95% CI]':>24s} {'better':>6s}"
        )
        for name, rows_all in [(args.ref, ref_all)] + models:
            rows = [r for r in rows_all if r["split"] == split]
            if frame_keys(rows) != frame_keys(ref):
                print(f"{name:28s} WARNING: frames differ from the reference ({len(rows)} vs {len(ref)}); skipped")
                continue
            p = pooled(rows, "")
            bias = float(np.mean([float(r["pred"]) - float(r["gt"]) for r in rows]))
            line = (
                f"{name:28s} {p['mae']:6.1f} {bias:+7.1f} {p['f1']:5.1f} {p['moda']:6.1f} {p['game3']:6.1f} "
                f"{p['far1m']:6.1f} {p['far2m']:6.1f}"
            )
            if name != args.ref:
                m, lo, hi, nb, n = paired(rows, ref, "mae")
                fm, flo, fhi, fnb, _ = paired(rows, ref, "far2m")
                line += (
                    f" | {m:+7.1f} [{lo:+6.1f}, {hi:+6.1f}] {nb:3d}/{n:<2d} | "
                    f"{100 * fm:+7.1f} [{100 * flo:+6.1f}, {100 * fhi:+6.1f}] {fnb:3d}/{n:<2d}"
                )
            print(line)


if __name__ == "__main__":
    main()
