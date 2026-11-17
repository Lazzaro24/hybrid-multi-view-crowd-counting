"""
visualize_router_mask.py

Qualitative companion to diagnose_routing.py: shows the LEARNED router mask (gating_net's sigmoid
output) against the oracle (ground-truth-density-threshold) mask, spatially, for a Stage>=3
checkpoint -- something none of the existing visualization scripts do (visualize_experts.py always
uses the oracle mask for its "MASKED" panels and discards the model's own learned mask entirely).

Panels per patch: oracle mask | learned mask m | mask error (m - oracle, diverging: red = router
over-covers, blue = under-covers) | true density T | fused prediction under the LEARNED mask | fused
prediction under the ORACLE mask (upper-bound reference).

Do NOT run while training is running.
Usage (from src):
    uv run scripts/visualisation/visualize_router_mask.py --checkpoint multiscene_best_stage_3_dout0.7_cw0.01_f10.pth --val_scenes scene_49,scene_18
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
from visualize_experts import draw_panels, build_filtered_samples


def main():
    import torch
    import torch.nn.functional as F
    import train_proposed_multiscene as T

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val_scenes", default="scene_49,scene_18")
    ap.add_argument("--frames_per_scene", type=int, default=2)
    ap.add_argument("--val_seed", type=int, default=1234)
    ap.add_argument("--patch", default="auto", help="'auto' = the patch with most oracle-dense pixels, or 0/1/2")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sparse_arch", choices=["shallow", "deep"], default="shallow")
    ap.add_argument("--out_dir", default=os.path.join(src_dir, "diagnostics", "router_mask_viz"))
    args = ap.parse_args()
    T.SPARSE_ARCH = args.sparse_arch
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(T.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json"), "r") as f:
        cams_cfg = json.load(f)
    val_all = T.build_scene_registry("validate", args.k, cams_cfg, {}, validate_setup=False)
    wanted = [s.strip() for s in args.val_scenes.split(",") if s.strip()]
    val_sel = [e for e in val_all if e["scene_name"] in wanted]
    missing = sorted(set(wanted) - {e["scene_name"] for e in val_sel})
    if missing:
        print(f"[warning] scenes not in the val registry: {missing}")
    samples = build_filtered_samples(T, val_sel, args.frames_per_scene, args.val_seed, 0, 0, 1)

    label = os.path.splitext(os.path.basename(args.checkpoint))[0]
    model = T.ProposedModel(num_cam=args.k).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing_k, unexpected_k = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"[load] {label}: missing={len(missing_k)} unexpected={len(unexpected_k)}")
    model.eval()

    map_kernel = T.get_map_kernel(sigma=T.GAUSSIAN_SIGMA, kernel_size=T.GAUSSIAN_KERNEL_SIZE).to(device)
    KMP = float(map_kernel.sum())
    pad = map_kernel.shape[-1] // 2

    for s in samples:
        wld = s["wld_map_paras"].to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            proj, masks = model(
                s["images"].to(device), s["ref_points_full"].to(device), s["hw_random"].to(device), stage=3
            )
        proj, masks = proj.float(), masks.float()
        P = proj.shape[1]

        gms, Ts = [], []
        for p in range(P):
            hw = s["hw_random"].to(device)[:, :, p]
            tpatch = T.create_ground_truth_heatmap(s["gt_world_xy"], wld, hw, bev_shape=T.BEV_SHAPE).to(device)
            conv = F.conv2d(tpatch, map_kernel, padding=pad)
            gm = T.oracle_mask_from_conv(conv)
            gms.append(gm)
            Ts.append(conv)

        show_p = int(np.argmax([float(g.sum()) for g in gms])) if args.patch == "auto" else min(int(args.patch), P - 1)

        gm, T_conv = gms[show_p], Ts[show_p]
        feat, m = proj[:, show_p], masks[:, show_p]
        with torch.no_grad():
            D = torch.relu(model.head.dense_expert(feat)).float()
            S = torch.relu(model.head.sparse_expert(feat)).float()
        fused_learned = m * D + (1 - m) * S
        fused_oracle = gm * D + (1 - gm) * S

        gm_np, m_np = gm[0, 0].cpu().numpy(), m[0, 0].cpu().numpy()
        T_np = T_conv[0, 0].cpu().numpy()
        vmax = max(float(T_np.max()), 1.0)
        learned_mass = float(fused_learned.sum() / KMP)
        oracle_mass = float(fused_oracle.sum() / KMP)
        true_mass = float(T_conv.sum() / KMP)

        panels = [
            (f"oracle mask (coverage {100 * gm_np.mean():.1f}%)", gm_np, "gray", 0, 1),
            (
                f"learned mask m (mean-in {(m_np * gm_np).sum() / max(gm_np.sum(), 1):.2f}, "
                f"mean-out {(m_np * (1 - gm_np)).sum() / max((1 - gm_np).sum(), 1):.2f})",
                m_np,
                "gray",
                0,
                1,
            ),
            ("mask error (m - oracle)\nred=over-covers, blue=under-covers", m_np - gm_np, "RdBu_r", -1, 1),
            (f"true density T (max {T_np.max():.2f})", T_np, "hot", 0, vmax),
            (
                f"fused, LEARNED mask\ntotal {learned_mass:.1f} vs true {true_mass:.1f}",
                fused_learned[0, 0].cpu().numpy(),
                "hot",
                0,
                vmax,
            ),
            (
                f"fused, ORACLE mask (upper bound)\ntotal {oracle_mass:.1f} vs true {true_mass:.1f}",
                fused_oracle[0, 0].cpu().numpy(),
                "hot",
                0,
                vmax,
            ),
        ]
        snap = str(s["snapshot_id"][0])
        fname = f"router_{label}_{s['scene_name']}_snap{snap}_p{show_p}.png"
        draw_panels(
            os.path.join(args.out_dir, fname),
            f"{label} | {s['scene_name']} | snapshot {snap} | patch {show_p} | shared colour scale (0..{vmax:.2f})",
            panels,
        )
        print(f"[wrote] {fname}  learned={learned_mass:.1f} oracle={oracle_mass:.1f} true={true_mass:.1f}")

    print(f"\nFigures written to {args.out_dir}")


if __name__ == "__main__":
    main()
