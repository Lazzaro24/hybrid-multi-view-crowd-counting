"""
evaluation/model_cost.py

Cost table of the final models (doc/final_evaluations.md E7): parameters per part, inference time per sample, peak GPU
memory and FLOPs, each with exactly the prediction path of evaluate_paper_protocol.py (whole scene, 5 views, batch 1;
the proposed models in 200 x 200 tiles under autocast, the baseline in one pass without autocast, as the eval runs
them; the density rule predicts the dense map twice: once for its zone, once for the routed map).

Samples: --samples test samples (seeded frames over all test scenes, seeded 5-view draws, the same for every model),
after --warmup samples that are not timed. Data loading is outside the timing; every timed call is wrapped in
torch.cuda.synchronize(). The backbone (frozen, the same ResNet-18 in every model) is also timed on its own, so the
cost of fusion + head can be read as total - backbone. FLOPs (torch.utils.flop_counter) on the first timed sample;
the map size changes per scene, so they are reported with that sample's map size.

Usage (from src):
    uv run python evaluation/model_cost.py
    uv run python evaluation/model_cost.py --models "baseline_ot=baseline:learned:runs/baseline/ot/best.pth"
Writes diagnostics/model_cost/cost.json and prints the table.
"""

import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)
sys.path.append(current_dir)

import evaluate_paper_protocol as EP
from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
from utils.exclusions import load_excluded_scenes

# The final models, local paths (doc/final_evaluations.md §1)
DEFAULT_MODELS = ",".join(
    [
        "baseline_mse=baseline:learned:runs/baseline/run_1/best.pth",
        "baseline_ot=baseline:learned:runs/baseline/ot/best.pth",
        "h1_hom=proposed:learned:runs/full_length_dense_mask_homography/4_router_anchor/"
        "multiscene_final_stage_4_bud200_homography_256_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth",
        "h1_casa=proposed:learned:runs/full_length_dense_mask_final/4_router_anchor/"
        "multiscene_final_stage_4_bud200_ca_sa_dout0.7_cw0.01_genloss_a0.7_ro_anchor0.1_fs.pth",
        "hybrid_hom=proposed:dzone0.1p:runs/full_length_hybrid_results/"
        "multiscene_final_stage_2_bud200_homography_256_S4D2_s42.pth",
    ]
)


def parameter_table(model, routing):
    """Parameters per top-level part, the head split into its children (dense / sparse expert, router). Parts the
    prediction never runs are listed but not counted: the Stage 1 decoder (throwaway_head) and, with the density rule,
    the router (head.gating_net)."""
    parts = {}
    for name, child in model.named_children():
        if name == "head" and len(list(child.named_children())) > 1:
            for sub, mod in child.named_children():
                parts[f"head.{sub}"] = sum(p.numel() for p in mod.parameters())
        else:
            parts[name] = sum(p.numel() for p in child.parameters())
    unused = [p for p in ("throwaway_head", "head.gating_net" if routing.startswith("dzone") else None) if p in parts]
    total = sum(v for k, v in parts.items() if k not in unused)
    return {
        "total": total,
        "without_backbone": total - parts.get("backbone", 0),
        "parts": parts,
        "unused_at_inference": unused,
    }


def make_predict(model_type, routing, model, device):
    """The prediction call of evaluate_paper_protocol.run_samples for this model and routing."""
    if model_type == "baseline":
        return lambda sample: EP.predict_baseline(model, sample, device)
    if routing == "learned":
        return lambda sample: EP.predict_proposed(model, sample, device)
    if routing.startswith("dzone"):
        thr = float(routing[len("dzone") :])

        def predict(sample):
            shape = (int(sample["wld_map_paras"][0, 3].item()), int(sample["wld_map_paras"][0, 4].item()))
            return EP.predict_proposed(model, sample, device, EP.density_zone(model, sample, device, thr, shape))

        return predict
    raise SystemExit(f"routing {routing!r}: model_cost.py times learned | dzone<thr>[p]")


def timed(fn, device):
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize(device)
    return out, time.perf_counter() - t0


def load_samples(args):
    """--warmup + --samples test samples: seeded frames, seeded 5-view draws (identical for every model)."""
    split_dir = os.path.join(src_dir, "datasets", "SynMVCrowd", "data", args.split)
    excluded = load_excluded_scenes()
    scenes = [
        s for s in sorted(os.listdir(split_dir)) if os.path.isdir(os.path.join(split_dir, s)) and s not in excluded
    ]
    datasets = [
        SynMVCrowdDataset(scene_dir=os.path.join(split_dir, s), camera_ids=None, num_sampled_cams=5, load_refs=True)
        for s in scenes
    ]
    rng = np.random.RandomState(args.seed)
    picks = [(k, int(rng.randint(len(datasets[k])))) for k in rng.randint(len(scenes), size=args.warmup + args.samples)]
    return scenes, datasets, picks


def batch(s):
    """One unbatched dataset sample -> the batch-of-1 layout the eval's DataLoader gives."""
    keys = ("images", "camera_paras", "wld_map_paras", "ref_points_full")
    return {k: s[k][None] for k in keys if k in s}


def main():
    ap = argparse.ArgumentParser(description="Parameters, time, memory and FLOPs of the final models.")
    ap.add_argument("--models", default=DEFAULT_MODELS, help="comma-separated name=model:routing:weights")
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(src_dir, "diagnostics", "model_cost", "cost.json"))
    args = ap.parse_args()
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(device)

    scenes, datasets, picks = load_samples(args)
    results = {"gpu": gpu, "samples": args.samples, "warmup": args.warmup, "seed": args.seed, "models": {}}
    for spec in args.models.split(","):
        name, rest = spec.split("=", 1)
        model_type, routing, weights = rest.split(":", 2)
        model, epoch = EP.load_model(model_type, os.path.join(src_dir, weights), device)
        model.requires_grad_(False)  # eval only; FlopCounterMode's module tracker fails on grad-requiring leaves
        base, peaks = EP.parse_routing(routing)
        if peaks:
            model.T.apply_sparse_render(model, {"mode": "peaks", "kernel": 3, "threshold": 0.064})
            model.head.render_kernel = model.head.render_kernel.to(device)
        predict = make_predict(model_type, base, model, device)
        autocast = model_type != "baseline"  # as the eval: predict_proposed runs under autocast, the baseline doesn't
        print(f"[load] {name}: {weights} (epoch {epoch}, {routing})")

        times, bb_times, flops = [], [], None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for j, (k, f) in enumerate(picks):
                random.seed(args.seed * 100003 + j)  # the same camera draw for every model
                sample = batch(datasets[k][f])
                _, t = timed(lambda: predict(sample), device)
                images = sample["images"].to(device)
                with torch.amp.autocast("cuda", enabled=autocast):
                    _, tb = timed(lambda: model.backbone(images.flatten(0, 1)), device)
                if j < args.warmup:
                    continue
                times.append(t)
                bb_times.append(tb)
                if flops is None:
                    with FlopCounterMode(display=False) as fc:
                        predict(sample)
                    shape = tuple(int(sample["wld_map_paras"][0, i].item()) for i in (3, 4))
                    flops = {"total": fc.get_total_flops(), "map_shape": shape, "scene": scenes[k], "frame": f}
        q = lambda v: [float(x) for x in np.percentile(v, [25, 50, 75])]  # noqa: E731
        results["models"][name] = {
            "weights": weights,
            "routing": routing,
            "parameters": parameter_table(model, routing),
            "seconds_per_sample_q25_median_q75": q(times),
            "backbone_seconds_q25_median_q75": q(bb_times),
            "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
            "flops_one_sample": flops,
        }
        model = predict = None  # free the GPU before the next model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGPU {gpu}; {args.samples} samples after {args.warmup} warm-up, whole scene, 5 views, batch 1")
    print(
        f"{'model':<14}{'params [M]':>11}{'w/o backbone':>13}{'s/sample':>10}{'IQR':>15}{'backbone s':>12}"
        f"{'peak GB':>9}{'GFLOPs':>9}"
    )
    for name, r in results["models"].items():
        p, t, b = r["parameters"], r["seconds_per_sample_q25_median_q75"], r["backbone_seconds_q25_median_q75"]
        print(
            f"{name:<14}{p['total'] / 1e6:>11.2f}{p['without_backbone'] / 1e6:>13.2f}{t[1]:>10.3f}"
            f"{f'{t[0]:.3f}-{t[2]:.3f}':>15}{b[1]:>12.3f}{r['peak_memory_gb']:>9.2f}"
            f"{r['flops_one_sample']['total'] / 1e9:>9.0f}"
        )
        print(
            "    "
            + ", ".join(
                f"{k} {v / 1e6:.2f} M" + (" (unused)" if k in p["unused_at_inference"] else "")
                for k, v in p["parts"].items()
            )
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
