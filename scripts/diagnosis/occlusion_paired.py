"""Occlusion E6 analysis: consistency with §8.21, recovery per model, paired within-scene differences."""

import sys
import numpy as np
import csv


def load(p):
    rows = list(csv.DictReader(open(p)))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])
    return out


new, old = load(sys.argv[1]), load(sys.argv[2])
kn = list(zip(new["scene"], new["snapshot"], new["idx"]))
ko = {k: i for i, k in enumerate(zip(old["scene"], old["snapshot"], old["idx"]))}
pairs = [(i, ko[k]) for i, k in enumerate(kn) if k in ko]
a_i, b_i = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
print(f"rows new {len(kn)}, old {len(ko)}, matched {len(pairs)}")
for a, b in (
    ("recovery_homography_256", "recovery_bud200_h0ctrl_homography_256"),
    ("recovery_hybrid_hom", "recovery_fb_hyb10"),
):
    print(f"  {a} vs old {b}: max |diff| {np.abs(new[a][a_i] - old[b][b_i]).max():.2e}")

models = ["homography_256", "h1_hom", "h1_casa", "hybrid_hom", "baseline_mse", "baseline_ot"]
rec = {k: new[f"recovery_{k}"] for k in models}
scenes = new["scene"]
us = list(dict.fromkeys(scenes))
groups = {u: np.where(scenes == u)[0] for u in us}
rng = np.random.default_rng(0)
blocked = new["n_block_people"] + new["n_block_static"]
fully = (new["n_vis_all"] == new["n_fov"]) & (new["n_fov"] > 0)
print(f"\nperson-frames {len(kn)}, frames {len(set(zip(new['scene'], new['snapshot'])))}, scenes {len(us)}")
print(f"{'model':<16}{'recovery':>9}{'miss<0.5':>10}{'rec. all views clear':>22}{'rec. >=2 views blocked':>24}")
for k in models:
    r = rec[k]
    print(f"{k:<16}{r.mean():>9.3f}{(r < 0.5).mean():>10.1%}{r[fully].mean():>22.3f}{r[blocked >= 2].mean():>24.3f}")


def scene_boot(y, n=2000):
    means = np.array([y[groups[u]].mean() for u in us])
    boots = [means[rng.integers(0, len(us), len(us))].mean() for _ in range(n)]
    return means.mean(), np.percentile(boots, [2.5, 97.5]), (means > 0).sum()


nf = new["n_fov"]
bp = new["n_block_people"]
bs = new["n_block_static"]
loc = new["local_2m"]
D = np.column_stack([(scenes == u).astype(float) for u in us])
X = np.column_stack([D, nf, bp, bs, loc])
names = ["n_in_view", "n_blocked_people", "n_blocked_static", "local_2m"]


def fe(idx, y):
    cols = [j for j in range(len(us)) if X[idx, j].any()] + list(range(len(us), X.shape[1]))
    return np.linalg.lstsq(X[idx][:, cols], y[idx], rcond=None)[0][-4:]


print("\npaired difference per person (model - reference): mean recovery diff [scene bootstrap 95%], scenes > 0;")
print("within-scene regression slopes of the difference [scene bootstrap 95%]")
for ref in ("baseline_ot", "baseline_mse", "homography_256"):
    for k in [x for x in models if x != ref and not (ref != "baseline_ot" and x.startswith("baseline"))]:
        y = rec[k] - rec[ref]
        mu, ci, pos = scene_boot(y)
        coef = fe(np.arange(len(y)), y)
        boot = np.array([fe(np.concatenate([groups[u] for u in rng.choice(us, len(us))]), y) for _ in range(300)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        sl = "  ".join(f"{n} {c:+.3f} [{a:+.3f}, {b:+.3f}]" for n, c, a, b in zip(names[:3], coef, lo, hi))
        print(f"  {k} - {ref}: {mu:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] {pos}/{len(us)} | {sl}")


# groups of head_ablation.md §8.21, paired per scene against a reference
grp = {
    "all": np.ones(len(nf), bool),
    "clearly visible (no view blocked, >= 3 in view)": (bp + bs == 0) & (nf >= 3),
    ">= 2 views behind static occluders": bs >= 2,
    ">= 1 view blocked by people": bp >= 1,
}


def paired(y_a, y_b, msk, n=2000):
    d = np.array(
        [
            y_a[groups[u]][msk[groups[u]]].mean() - y_b[groups[u]][msk[groups[u]]].mean()
            for u in us
            if msk[groups[u]].any()
        ]
    )
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {(d > 0).sum()}/{len(d)}"


for ref in ("baseline_ot", "homography_256"):
    print(f"\n=== groups, mean recovery (missed), paired per scene vs {ref} ===")
    for g, msk in grp.items():
        print(f"[{g}] n={msk.sum()}")
        for k in models:
            r = rec[k][msk]
            extra = "" if k == ref else "  " + paired(rec[k], rec[ref], msk)
            print(f"   {k:<16}{r.mean():.3f} ({(r < 0.5).mean():.0%}){extra}")
