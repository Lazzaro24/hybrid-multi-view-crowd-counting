"""Console tee, timing instrumentation (--enable_timing) and the per-scene validation tables."""

import csv
import json
import os
import time

import numpy as np
import torch

from . import config as cfg


class Tee:
    """Duplicates every write to several streams (console + log file), flushing each time so the log stays current
    even if the process is killed."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def format_duration(seconds):
    """Human-readable duration: 12.3s / 4m 05s / 2h 03m 07s."""
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"


def tick(sync):
    """perf_counter, optionally after a CUDA synchronize so queued GPU work is attributed to the phase that launched it."""
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class PhaseTimer:
    """Accumulates wall-clock seconds per named phase (no-op when disabled)."""

    ORDER = ["data_wait", "h2d", "forward", "gt_and_loss", "backward_step", "bookkeeping"]

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.totals = {}

    def add(self, name, seconds):
        if self.enabled:
            self.totals[name] = self.totals.get(name, 0.0) + seconds

    def get(self, name):
        return self.totals.get(name, 0.0)

    def breakdown_line(self, n_iters):
        total = sum(self.totals.values())
        if not self.enabled or total <= 0 or n_iters <= 0:
            return None
        parts = [
            f"{name} {self.totals[name] / n_iters:.3f}s ({100 * self.totals[name] / total:.0f}%)"
            for name in self.ORDER
            if name in self.totals
        ]
        return " | ".join(parts)


def timing_csv_path(kind):
    return os.path.join(cfg.TIMING_DIR, f"timing_{kind}_{cfg.TIMING_RUN_TAG}.csv")


def append_csv_row(path, header, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)


def describe_hardware():
    info = {"torch": torch.__version__, "cpu_count": os.cpu_count(), "gpu": None}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {"gpu": props.name, "gpu_mem_gb": round(props.total_memory / (1024**3), 1), "cuda": torch.version.cuda}
        )
    return info


def print_and_save_timing_report():
    """Prints a per-stage timing table and writes timing_summary_<tag>.json. No-op unless --enable_timing was passed."""
    if not cfg.TIMING_ENABLED or not cfg.STAGE_TIMING_SUMMARIES:
        return
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print(f"{'Stage':<7}{'Epochs':<10}{'Total':<13}{'Mean epoch':<13}{'s/it':<8}{'Best ep.':<10}{'Time to best':<14}")
    for s_ in cfg.STAGE_TIMING_SUMMARIES:
        epochs_txt = f"{s_['epochs_run']}/{s_['epochs_budget']}" + ("*" if s_["stopped_early"] else "")
        print(
            f"{s_['stage']:<7}{epochs_txt:<10}{format_duration(s_['stage_total_s']):<13}"
            f"{format_duration(s_['mean_epoch_s']):<13}{s_['mean_s_per_it']:<8.2f}"
            f"{str(s_['best_epoch']):<10}{format_duration(s_['time_to_best_s']):<14}"
        )
    training_total = sum(s_["stage_total_s"] for s_ in cfg.STAGE_TIMING_SUMMARIES)
    startup_total = sum(cfg.STARTUP_TIMING.values())
    print(f"\nTotal training time (all stages): {format_duration(training_total)}")
    if cfg.STARTUP_TIMING:
        print(
            f"Startup (registries etc.): {format_duration(startup_total)} "
            f"-> grand total {format_duration(training_total + startup_total)}"
        )
    print("(* = stopped early by patience)")
    print("=" * 70)

    os.makedirs(cfg.TIMING_DIR, exist_ok=True)
    json_path = os.path.join(cfg.TIMING_DIR, f"timing_summary_{cfg.TIMING_RUN_TAG}.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "run_tag": cfg.TIMING_RUN_TAG,
                "hardware": describe_hardware(),
                "startup_s": cfg.STARTUP_TIMING,
                "phase_timing_enabled": cfg.PHASE_TIMING,
                "stages": cfg.STAGE_TIMING_SUMMARIES,
                "total_training_s": training_total,
            },
            f,
            indent=2,
        )
    print(f"[timing] Summary written to {json_path}")


def print_epoch_timing(
    stage, epoch, epochs, t, train_timer, scene_timing_rows, n_val_samples, n_train_scenes, scene_ref_shape
):
    """The per-epoch "[timing]" lines and the epoch / scene timing CSV rows (only with --enable_timing).
    t: dict with train_s, val_s, epoch_s, iters, resample_s, empty_cache_s, stage_elapsed_s, epoch_times."""
    s_per_it = (t["train_s"] - t["resample_s"] - t["empty_cache_s"]) / max(t["iters"], 1)
    mean_epoch_s = sum(t["epoch_times"]) / len(t["epoch_times"])
    eta_s = mean_epoch_s * (epochs - epoch - 1)
    print(
        f"    [timing] Epoch {epoch + 1}/{epochs} | train {format_duration(t['train_s'])} "
        f"({t['iters']} it, {s_per_it:.2f} s/it) | val {format_duration(t['val_s'])} "
        f"({n_val_samples} samples, {t['val_s'] / max(n_val_samples, 1):.2f} s/sample) | "
        f"epoch {format_duration(t['epoch_s'])} | stage elapsed {format_duration(t['stage_elapsed_s'])} | "
        f"ETA (no early stop) {format_duration(eta_s)}"
    )
    breakdown = train_timer.breakdown_line(t["iters"])
    print(f"    [timing]   per-iteration phases: {breakdown if breakdown else 'disabled (--no_phase_timing)'}")
    print(f"    [timing]   per-epoch overhead: resample {t['resample_s']:.1f}s | empty_cache {t['empty_cache_s']:.1f}s")
    slowest = sorted(((n, sec / max(it, 1)) for n, it, sec in scene_timing_rows), key=lambda x: -x[1])[:3]
    print("    [timing]   slowest scenes (s/it): " + ", ".join(f"{n} {v:.2f}" for n, v in slowest))

    append_csv_row(
        timing_csv_path("epochs"),
        [
            "run_tag",
            "stage",
            "epoch",
            "train_s",
            "val_s",
            "epoch_s",
            "iterations",
            "s_per_it",
            "resample_s",
            "empty_cache_s",
            "data_wait_s",
            "h2d_s",
            "forward_s",
            "gt_and_loss_s",
            "backward_step_s",
            "bookkeeping_s",
            "stage_elapsed_s",
            "n_train_scenes",
            "n_val_samples",
        ],
        [
            cfg.TIMING_RUN_TAG,
            stage,
            epoch + 1,
            f"{t['train_s']:.3f}",
            f"{t['val_s']:.3f}",
            f"{t['epoch_s']:.3f}",
            t["iters"],
            f"{s_per_it:.4f}",
            f"{t['resample_s']:.3f}",
            f"{t['empty_cache_s']:.3f}",
        ]
        + [(f"{train_timer.get(n_):.3f}" if cfg.PHASE_TIMING else "") for n_ in PhaseTimer.ORDER]
        + [f"{t['stage_elapsed_s']:.3f}", n_train_scenes, n_val_samples],
    )
    for n_, it_, sec_ in scene_timing_rows:
        ref_h, ref_w = scene_ref_shape.get(n_, ("", ""))
        append_csv_row(
            timing_csv_path("scenes"),
            ["run_tag", "stage", "epoch", "scene", "iterations", "seconds", "s_per_it", "ref_h", "ref_w"],
            [cfg.TIMING_RUN_TAG, stage, epoch + 1, n_, it_, f"{sec_:.3f}", f"{sec_ / max(it_, 1):.4f}", ref_h, ref_w],
        )


def record_stage_timing(stage, epochs, start_epoch, t, best_epoch, time_to_best_s, best_metric):
    """Stage summary for the end-of-run timing report (and, with --enable_timing, its console line and CSV row).
    t: dict with stage_start, setup_s, val_cache_s, train_s, val_s, iters, iter_s, epoch_times."""
    stage_total_s = time.perf_counter() - t["stage_start"]
    n_epochs_run = len(t["epoch_times"])
    summary = {
        "stage": stage,
        "epochs_run": n_epochs_run,
        "epochs_budget": epochs,
        "stopped_early": n_epochs_run < (epochs - start_epoch),
        "stage_total_s": stage_total_s,
        "setup_s": t["setup_s"],
        "val_cache_s": t["val_cache_s"],
        "train_s": t["train_s"],
        "val_s": t["val_s"],
        "mean_epoch_s": (sum(t["epoch_times"]) / n_epochs_run) if n_epochs_run else 0.0,
        "iterations": t["iters"],
        "mean_s_per_it": t["iter_s"] / max(t["iters"], 1),
        "best_epoch": best_epoch,
        "time_to_best_s": time_to_best_s,
        "best_metric": None if best_metric == float("inf") else float(best_metric),
    }
    cfg.STAGE_TIMING_SUMMARIES.append(summary)
    if not cfg.TIMING_ENABLED:
        return
    print(
        f"[timing] Stage {stage} finished: {n_epochs_run}/{epochs} epochs in {format_duration(stage_total_s)} "
        f"(setup {format_duration(t['setup_s'])}, train {format_duration(t['train_s'])}, val {format_duration(t['val_s'])}); "
        f"mean {summary['mean_s_per_it']:.2f} s/it over {t['iters']} iterations; "
        f"best epoch {best_epoch} reached after {format_duration(time_to_best_s)}."
    )
    append_csv_row(
        timing_csv_path("stages"),
        [
            "run_tag",
            "stage",
            "epochs_run",
            "epochs_budget",
            "stopped_early",
            "stage_total_s",
            "setup_s",
            "val_cache_s",
            "train_s",
            "val_s",
            "mean_epoch_s",
            "iterations",
            "mean_s_per_it",
            "best_epoch",
            "time_to_best_s",
            "best_metric",
        ],
        [
            cfg.TIMING_RUN_TAG,
            stage,
            n_epochs_run,
            epochs,
            summary["stopped_early"],
            f"{stage_total_s:.1f}",
            f"{t['setup_s']:.1f}",
            f"{t['val_cache_s']:.1f}",
            f"{t['train_s']:.1f}",
            f"{t['val_s']:.1f}",
            f"{summary['mean_epoch_s']:.1f}",
            t["iters"],
            f"{summary['mean_s_per_it']:.4f}",
            best_epoch,
            "" if time_to_best_s is None else f"{time_to_best_s:.1f}",
            summary["best_metric"],
        ],
    )


def report_joint_train_terms(epoch, stats, optimizer):
    """--joint: the epoch's mean unweighted loss terms per patch and the learning rates -> stdout and
    train_joint_<run tag>.csv (the w_r / w_f balance of doc/joint_training.md is read off it)."""
    terms = {"loss": stats.loss / max(stats.n_batches, 1)}
    terms.update({k: v / max(stats.joint_patches, 1) for k, v in stats.joint_terms.items()})
    lrs = " / ".join(f"{g['lr']:.3e}" for g in optimizer.param_groups)
    print(
        f">>> Joint Epoch {epoch + 1} train | " + " | ".join(f"{k} {v:.4f}" for k, v in terms.items()) + f" | lr {lrs}"
    )
    append_csv_row(
        os.path.join(cfg.VAL_LOG_DIR, f"train_joint_{cfg.TIMING_RUN_TAG}.csv"),
        ["run_tag", "epoch", *terms, "w_router", "w_fused", "lrs"],
        [cfg.TIMING_RUN_TAG, epoch + 1, *(f"{v:.6f}" for v in terms.values())]
        + [cfg.JOINT_ROUTER_WEIGHT, cfg.JOINT_FUSED_WEIGHT, lrs],
    )


def val_scenes_csv_path():
    return os.path.join(cfg.VAL_LOG_DIR, f"val_scenes_{cfg.TIMING_RUN_TAG}.csv")


def summarize_scene_stats(scene_stats, scene_order):
    """Collapses {scene: {'abs_err': [...], 'gt': [...], 'pred': [...], optional extras}} into one row per scene."""
    rows = []
    for scene in scene_order:
        b = scene_stats.get(scene)
        if not b or not b.get("abs_err"):
            continue
        abs_err = np.array(b["abs_err"], dtype=np.float64)
        gt = np.array(b["gt"], dtype=np.float64)
        pred = np.array(b["pred"], dtype=np.float64)
        row = {
            "scene": scene,
            "n_patches": int(len(abs_err)),
            "mean_gt": float(gt.mean()),
            "mean_pred": float(pred.mean()),
            "mae": float(abs_err.mean()),
            "bias": float((pred - gt).mean()),
            "nae": float(abs_err.sum() / max(gt.sum(), 1e-6)),
            "abs_sum": float(abs_err.sum()),
            "dense_mae": float(np.mean(b["dense_err"])) if b.get("dense_err") else None,
            "sparse_mae": float(np.mean(b["sparse_err"])) if b.get("sparse_err") else None,
            "mean_rel_err": float(np.mean(b["rel_err"])) if b.get("rel_err") else None,
        }
        rows.append(row)
    total_abs = sum(r["abs_sum"] for r in rows)
    for r in rows:
        r["share"] = r["abs_sum"] / total_abs if total_abs > 0 else 0.0
    return rows


def report_scene_breakdown(stage, epoch, rows, full_table):
    """One summary line every epoch; the full per-scene table when full_table=True."""
    if not rows:
        return
    total_n = sum(r["n_patches"] for r in rows)
    total_abs = sum(r["abs_sum"] for r in rows)
    total_gt = sum(r["mean_gt"] * r["n_patches"] for r in rows)
    median_mae = float(np.median([r["mae"] for r in rows]))
    worst = max(rows, key=lambda r: r["mae"])
    print(
        f"    [val-scene] {len(rows)} scenes, {total_n} patches | pooled MAE {total_abs / total_n:.1f} | "
        f"mean GT/patch {total_gt / total_n:.1f} | pooled NAE {100 * total_abs / max(total_gt, 1e-6):.0f}% | "
        f"median scene MAE {median_mae:.1f} | worst {worst['scene']} MAE {worst['mae']:.1f} "
        f"({100 * worst['share']:.0f}% of total error)"
    )
    if not full_table:
        return
    extra_hdr = ""
    if stage == 2:
        extra_hdr = f"{'denseMAE':>9}{'sparseMAE':>10}"
    elif stage == 3:
        extra_hdr = f"{'relErr':>8}"
    print(
        f"    [val-scene] {'scene':<11}{'patches':>8}{'meanGT':>9}{'meanPred':>10}{'MAE':>9}{'bias':>9}{'NAE':>7}{'share':>7}{extra_hdr}"
    )
    for r in sorted(rows, key=lambda r: -r["mae"]):
        extra = ""
        if stage == 2:
            extra = f"{r['dense_mae']:>9.1f}{r['sparse_mae']:>10.1f}"
        elif stage == 3:
            extra = f"{100 * r['mean_rel_err']:>7.0f}%"
        print(
            f"    [val-scene] {r['scene']:<11}{r['n_patches']:>8}{r['mean_gt']:>9.1f}{r['mean_pred']:>10.1f}"
            f"{r['mae']:>9.1f}{r['bias']:>+9.1f}{100 * r['nae']:>6.0f}%{100 * r['share']:>6.0f}%{extra}"
        )


def write_scene_rows_csv(stage, epoch, rows, path=None, label="scene"):
    path = path or val_scenes_csv_path()
    header = [
        "run_tag",
        "stage",
        "epoch",
        label,
        "n_patches",
        "mean_gt",
        "mean_pred",
        "mae",
        "bias",
        "nae",
        "error_share",
        "dense_mae",
        "sparse_mae",
        "mean_rel_err",
    ]
    for r in rows:

        def fmt(v):
            return "" if v is None else f"{v:.4f}"

        append_csv_row(
            path,
            header,
            [
                cfg.TIMING_RUN_TAG,
                stage,
                epoch,
                r["scene"],
                r["n_patches"],
                fmt(r["mean_gt"]),
                fmt(r["mean_pred"]),
                fmt(r["mae"]),
                fmt(r["bias"]),
                fmt(r["nae"]),
                fmt(r["share"]),
                fmt(r["dense_mae"]),
                fmt(r["sparse_mae"]),
                fmt(r["mean_rel_err"]),
            ],
        )
