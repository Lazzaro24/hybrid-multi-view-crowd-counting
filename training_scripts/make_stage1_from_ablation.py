"""
make_stage1_from_ablation.py

Turns a finished neck-ablation variant (runs/neck_ablation/<run>/<variant>/last.pth, final EMA weights) into a
"Stage 1 done" checkpoint for train_proposed_multiscene.py, so the proposed head (dense + sparse expert, router) can
be trained on that fusion with the usual Stage 2 / Stage 3 commands (--resume <this file>).

  backbone  <- the ablation's backbone (the same frozen 2D-pretrained ResNet-18 as the proposed lineage)
  neck      <- AblationNeck: the ablation's fusion module (weights copied) + a fresh head_processor (256 -> 64 ch)
  head      <- not saved: experts and router start fresh, as after a real Stage 1 (whose throwaway head is dropped)
The ablation's own counting head (BaselineDecoder) is dropped.

Usage (from src):
    uv run python training_scripts/make_stage1_from_ablation.py --run ablation_v2 --variant ca_sa
    -> multiscene_best_stage_1_abl_v2_ca_sa.pth
"""

import os
import sys
import argparse

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)
sys.path.append(current_dir)

import train_proposed_multiscene as TP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ablation_v2")
    ap.add_argument("--variant", required=True)
    ap.add_argument(
        "--run_dir", default=None, help="neck_ablation.py output folder (default: runs/neck_ablation/<run>)"
    )
    ap.add_argument(
        "--out", default=None, help="default: multiscene_best_stage_1_abl_<run suffix>_<variant>.pth in src"
    )
    args = ap.parse_args()

    run_dir = args.run_dir or os.path.join(src_dir, "runs", "neck_ablation", args.run)
    ck = torch.load(os.path.join(run_dir, args.variant, "last.pth"), map_location="cpu", weights_only=False)
    state = ck["ema_state_dict"]["model"] if ck.get("ema_state_dict") else ck["model_state_dict"]
    z_anchors = ck["args"].get("z_anchors", 4)
    assert ck["args"].get("refs_file", "master_refs.pt") == "master_refs.pt" and z_anchors == 4, (
        "the proposed training reads master_refs.pt (4 anchors); this variant was trained on another cache"
    )
    neck_config = {
        "type": "ablation",
        "variant": args.variant,
        "z_anchors": z_anchors,
        "source": f"{args.run}/{args.variant}/last.pth (epoch {ck['epoch']}, "
        f"{'EMA' if ck.get('ema_state_dict') else 'raw'})",
    }

    model = TP.ProposedModel(num_cam=TP.NUM_CAMS_TO_SAMPLE, neck_config=neck_config)
    mapped = {k: v for k, v in state.items() if k.startswith("backbone.")}
    mapped.update({"neck.fusion." + k[len("fusion.") :]: v for k, v in state.items() if k.startswith("fusion.")})
    target = model.state_dict()
    bad = [k for k in mapped if k not in target or target[k].shape != mapped[k].shape]
    assert not bad, f"keys that do not fit the proposed model: {bad[:5]}"
    covered = [k for k in target if k.startswith(("backbone.", "neck.fusion."))]
    missing = [k for k in covered if k not in mapped]
    assert not missing, f"backbone / fusion keys without ablation weights: {missing[:5]}"
    target.update(mapped)
    model.load_state_dict(target)
    # only what a Stage 2 resume should inherit: the head (experts, router, throwaway head) starts fresh
    out_state = {k: v for k, v in model.state_dict().items() if k.startswith(("backbone.", "neck."))}

    out = args.out or os.path.join(
        src_dir, f"multiscene_best_stage_1_abl_{args.run.replace('ablation_', '')}_{args.variant}.pth"
    )
    torch.save(
        {
            "model_state_dict": out_state,
            "neck_config": neck_config,
            "epoch": ck["epoch"],
            "stage": 1,
            "note": "made by make_stage1_from_ablation.py; neck.head_processor is freshly initialised",
        },
        out,
    )
    n_fusion = sum(v.numel() for k, v in out_state.items() if k.startswith("neck.fusion."))
    n_hp = sum(v.numel() for k, v in out_state.items() if k.startswith("neck.head_processor."))
    print(
        f"[saved] {out}\n  neck_config {neck_config}\n  {len(mapped)} tensors copied (fusion {n_fusion:,} values); "
        f"fresh head_processor {n_hp:,} values; maxpool layout {TP.checkpoint_uses_maxpool(out_state)}"
    )


if __name__ == "__main__":
    main()
