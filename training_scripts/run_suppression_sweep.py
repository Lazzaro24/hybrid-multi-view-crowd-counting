"""
run_suppression_sweep.py

Sweeps the dense expert's background-suppression settings over the WHOLE multi-scene setup: every arm is a
dense-only Stage 2 fine-tune (--dense_only; neck, backbone and sparse expert frozen) on all 28 train scenes,
validated on the 10 fixed val scenes (same seeded val patches as every other run). It runs the arms one after
another, parses each arm's log and prints one comparison table.

What it measures per arm (mean of the last 3 epochs; oracle routing hides leakage, so both are needed):
  leak   = dense-expert mass OUTSIDE the oracle dense mask, persons/patch  ([val-dense] line)   -> lower is better
  denseMAE = in-mask dense mass MAE                                                                -> must not get much worse
  comb   = oracle-routed combined MAE (informational)
An arm is "OK" if its denseMAE <= tol * control's denseMAE (default 1.25).

Afterwards: take the best 1-2 "OK" arms, re-run Stage 3 from their FINAL checkpoint
(src/multiscene_final_stage_2_sw_<arm>.pth) and compare the downstream mass error.
Selection here uses the val scenes; keep the test split for the final report.

Usage (repo root):
    uv run src/training_scripts/run_suppression_sweep.py --dry_run          # print the commands only
    uv run src/training_scripts/run_suppression_sweep.py --epochs 10        # full sweep
    uv run src/training_scripts/run_suppression_sweep.py --epochs 5 --only control,ring_w8_a0.25
Do NOT run anything else on the GPU / with large RAM use at the same time.
"""

import os
import re
import sys
import csv
import json
import argparse
import datetime
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
TRAIN_SCRIPT = os.path.join(current_dir, "train_proposed_multiscene.py")

# name, mode, weight, ring width.  Weights are starting points (the single-scene validated ring setting is
# ring / 0.25 / width 8); edit freely.
ARMS = [
    dict(name="control", mode="ring", weight=0.0, ring=5),
    dict(name="ring_w8_a0.05", mode="ring", weight=0.05, ring=8),
    dict(name="ring_w8_a0.25", mode="ring", weight=0.25, ring=8),
    dict(name="ring_w8_a1.0", mode="ring", weight=1.0, ring=8),
    dict(name="ring_w16_a0.25", mode="ring", weight=0.25, ring=16),
    dict(name="full_a0.005", mode="full", weight=0.005, ring=5),
    dict(name="full_a0.02", mode="full", weight=0.02, ring=5),
    dict(name="full_a0.1", mode="full", weight=0.1, ring=5),
    # dense outside-mask supervision toward the TRUE density (suppression off); see DENSE_OUT_WEIGHT in the training script
    dict(name="dout_0.02", mode="ring", weight=0.0, ring=5, dense_out=0.02),
    dict(name="dout_0.1", mode="ring", weight=0.0, ring=5, dense_out=0.1),
    dict(name="dout_0.5", mode="ring", weight=0.0, ring=5, dense_out=0.5),
]

EPOCH_RE = re.compile(
    r"Stage 2 Epoch (\d+) \|.*?VAL MAE.*?\(Combined: ([\d.]+) \| Dense\(mass\): ([\d.]+) \| Sparse\(peak\): ([\d.]+)\)"
)
LEAK_RE = re.compile(r"\[val-dense\].*?: ([\d.]+) persons/patch")
EXC_RE = re.compile(
    r"\[val-dense-excess\] true mass outside the mask ([\d.]+) \| dense mass above the true density outside ([\d.]+)"
)


def parse_line(line, state, rows):
    """Collects (epoch, combined, dense_mae, leak) rows. The [val-dense] line follows its epoch line."""
    m = EPOCH_RE.search(line)
    if m:
        state["cur"] = dict(epoch=int(m.group(1)), comb=float(m.group(2)), dense=float(m.group(3)), leak=None)
        rows.append(state["cur"])
        return
    m = LEAK_RE.search(line)
    if m and state.get("cur") is not None and state["cur"]["leak"] is None:
        state["cur"]["leak"] = float(m.group(1))
        return
    m = EXC_RE.search(line)
    if m and state.get("cur") is not None:
        state["cur"]["true_out"] = float(m.group(1))
        state["cur"]["excess"] = float(m.group(2))


def summarize_rows(rows, last_n=3):
    rows = [r for r in rows if r["leak"] is not None]
    if not rows:
        return None
    tail = rows[-last_n:]

    def mean(key):
        return sum(r[key] for r in tail) / len(tail)

    ex = [r for r in tail if "excess" in r]
    return dict(
        epochs=len(rows),
        leak=mean("leak"),
        dense=mean("dense"),
        comb=mean("comb"),
        excess=(sum(r["excess"] for r in ex) / len(ex)) if ex else None,
        true_out=(sum(r["true_out"] for r in ex) / len(ex)) if ex else None,
    )


def build_command(arm, args, log_path):
    return [
        sys.executable,
        TRAIN_SCRIPT,
        "--stage",
        "2",
        "--epochs",
        str(args.epochs),
        "--patience2",
        str(args.epochs),
        "--resume",
        args.resume,
        "--dense_only",
        "--lr",
        str(args.lr),
        "--bg_suppression_mode",
        arm["mode"],
        "--bg_suppression_weight",
        str(arm["weight"]),
        "--ring_width",
        str(arm["ring"]),
        "--dense_out_weight",
        str(arm.get("dense_out", 0.0)),
        "--ckpt_tag",
        f"_sw_{arm['name']}",
        "--k",
        "5",
        "--frames_per_scene",
        "5",
        "--val_frames_per_scene",
        "10",
        "--val_seed",
        "1234",
        "--skip_camera_validation",
        "--log_file",
        log_path,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=10, help=">=5 so periodic checkpoints exist; the final-epoch file is always saved"
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", default=os.path.join(src_dir, "multiscene_checkpoint_stage_2_p0_epoch_10.pth"))
    parser.add_argument("--only", default=None, help="comma-separated arm names")
    parser.add_argument("--tol", type=float, default=1.25, help="max allowed denseMAE relative to the control arm")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    arms = [a for a in ARMS if args.only is None or a["name"] in args.only.split(",")]
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(src_dir, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    print(
        f"Sweep {tag}: {len(arms)} arms x {args.epochs} epochs (est. ~{2 * args.epochs} min per arm), resume from {args.resume}"
    )

    results = {}
    for arm in arms:
        log_path = os.path.join(out_dir, f"sweep_{tag}_{arm['name']}.log")
        cmd = build_command(arm, args, log_path)
        print(
            "\n"
            + "=" * 78
            + f"\nARM {arm['name']}: mode={arm['mode']} weight={arm['weight']} ring={arm['ring']} dense_out={arm.get('dense_out', 0.0)}\n"
            + " ".join(cmd)
        )
        if args.dry_run:
            continue
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf8",
            errors="replace",
            env=env,
            cwd=os.path.dirname(src_dir),
        )
        rows, state = [], {}
        for line in proc.stdout:
            parse_line(line, state, rows)
            if ">>> Stage 2 Epoch" in line or "[val-dense" in line or "Traceback" in line or "Error" in line:
                print("   " + line.rstrip()[:230])
        proc.wait()
        results[arm["name"]] = dict(arm=arm, returncode=proc.returncode, summary=summarize_rows(rows), rows=rows)
        s = results[arm["name"]]["summary"]
        print(
            f"   -> exit {proc.returncode}; "
            + (
                f"last-3 mean: leak {s['leak']:.1f}, excess-over-truth {s['excess'] if s['excess'] is None else round(s['excess'], 1)}, denseMAE {s['dense']:.1f}, comb {s['comb']:.1f}"
                if s
                else "no parsable epochs"
            )
        )

    if args.dry_run:
        return
    base = results.get("control", {}).get("summary")
    print(
        "\n"
        + "=" * 78
        + "\nSUPPRESSION SWEEP SUMMARY (mean of last 3 epochs; oracle-routed comb is informational)\n"
        + "=" * 78
    )
    print(
        f"{'arm':<16}{'mode':<6}{'weight':>7}{'ring':>5}{'dout':>6}{'leak':>9}{'excess':>8}{'denseMAE':>10}{'comb':>8}   verdict"
    )
    table = []
    for name, r in results.items():
        s, a = r["summary"], r["arm"]
        if not s:
            print(f"{name:<16}{a['mode']:<6}{a['weight']:>7}{a['ring']:>5}   (failed: exit {r['returncode']})")
            continue
        ok = (base is None) or (s["dense"] <= args.tol * base["dense"])
        table.append((s["leak"], name, a, s, ok))
    for leak, name, a, s, ok in sorted(table, key=lambda t: t[0]):
        rel = (
            f"leak {100 * (1 - s['leak'] / base['leak']):+.0f}% vs control"
            if base and name != "control" and base["leak"]
            else ""
        )
        exc = f"{s['excess']:>8.1f}" if s.get("excess") is not None else f"{'n/a':>8}"
        print(
            f"{name:<16}{a['mode']:<6}{a['weight']:>7}{a['ring']:>5}{a.get('dense_out', 0.0):>6}{s['leak']:>9.1f}{exc}{s['dense']:>10.1f}{s['comb']:>8.1f}   "
            f"{'OK' if ok else 'denseMAE too high'}  {rel}"
        )
    good = [t for t in sorted(table, key=lambda t: t[0]) if t[4] and t[1] != "control"]
    if good:
        print("\nCandidates for a Stage 3 re-run (lowest leak with acceptable denseMAE):")
        for leak, name, a, s, ok in good[:2]:
            print(f"   {name}: --resume src/multiscene_final_stage_2_sw_{name}.pth")

    with open(os.path.join(out_dir, f"suppression_sweep_{tag}.json"), "w") as f:
        json.dump(
            {
                k: dict(arm=v["arm"], returncode=v["returncode"], summary=v["summary"], rows=v["rows"])
                for k, v in results.items()
            },
            f,
            indent=1,
        )
    with open(os.path.join(out_dir, f"suppression_sweep_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "mode", "weight", "ring", "epoch", "combined_mae", "dense_mae", "dense_leak"])
        for k, v in results.items():
            for r in v["rows"]:
                w.writerow(
                    [
                        k,
                        v["arm"]["mode"],
                        v["arm"]["weight"],
                        v["arm"]["ring"],
                        r["epoch"],
                        r["comb"],
                        r["dense"],
                        r["leak"],
                    ]
                )
    print(f"\nWrote sweep results to {out_dir}")


if __name__ == "__main__":
    main()
