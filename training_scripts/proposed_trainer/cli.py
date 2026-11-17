"""Command line of train_proposed_multiscene.py. Defaults that are settings live in config."""

import argparse
import os

from . import config as cfg


def build_parser():
    parser = argparse.ArgumentParser(description="Multi-Scene Curriculum Training")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Run a single stage. Omit this (and use --full_curriculum) to run all four in sequence.",
    )
    parser.add_argument(
        "--full_curriculum",
        action="store_true",
        help="Run Stages 1->4 in sequence, automatically chaining each stage from the "
        "previous stage's multiscene_best_stage_N.pth.",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help="No curriculum (doc/joint_training.md): neck, experts and router train together for --epochs on the "
        "Stage 2 + w_r x router + w_f x Stage 4 losses; router at the Stage 3 lr, no early stopping.",
    )
    parser.add_argument("--joint_router_weight", type=float, default=1.0, help="--joint: w_r (router Tversky)")
    parser.add_argument("--joint_fused_weight", type=float, default=1.0, help="--joint: w_f (fused-map loss)")
    parser.add_argument("--val_every", type=int, default=1, help="Validate every N epochs and after the last one.")
    parser.add_argument("--epochs", type=int, default=500, help="Used when --stage or --joint is given.")
    parser.add_argument("--epochs1", type=int, default=100)
    parser.add_argument("--epochs2", type=int, default=100)
    parser.add_argument("--epochs3", type=int, default=100)
    parser.add_argument("--epochs4", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from (single-stage mode only)")
    parser.add_argument(
        "--backbone",
        type=str,
        default=os.path.join(cfg.src_dir, "runs", "train_2d", "new_best", "resnet18_2d_pretrained_ep4.pth"),
        help="2D-pretrained stride-8 backbone, loaded when starting without --resume.",
    )
    parser.add_argument(
        "--no_router_contrast",
        action="store_true",
        help="GatingNet without the contrast branch (fine + context, the tversky_alpha=0.7 reference router). "
        "Use the same setting in every stage of a lineage and when evaluating it.",
    )
    parser.add_argument(
        "--neck",
        choices=["legacy", "bevformer", "ablation"],
        default="legacy",
        help="Neck of a fresh (Stage 1 or --joint) model: 'legacy' = 1x deformable self-attn + 1x cross-attn; "
        "'bevformer' = stack of BEVFormer encoder layers (models/neck/proposed/bev_encoder.py); 'ablation' = a "
        "randomly initialised neck-ablation fusion (--neck_variant). "
        "Ignored with --resume: the checkpoint's stored neck is used.",
    )
    parser.add_argument(
        "--neck_variant", default="homography_256", help="--neck ablation: the fusion (neck_ablation.py variants)"
    )
    parser.add_argument("--neck_layers", type=int, default=3)
    parser.add_argument("--neck_heads", type=int, default=8)
    parser.add_argument("--neck_points_per_anchor", type=int, default=1, help="SCA sampling points per pillar anchor")
    parser.add_argument("--neck_sa_points", type=int, default=4, help="self-attention sampling points per head")
    parser.add_argument("--neck_ffn_dim", type=int, default=512)
    parser.add_argument("--neck_dropout", type=float, default=0.1)
    parser.add_argument(
        "--neck_grad_checkpoint",
        action="store_true",
        help="bevformer neck: recompute encoder layers in the backward pass (3 layers: 9.7 -> 2.1 GiB peak, ~12%% slower)",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument(
        "--k",
        type=int,
        default=cfg.NUM_CAMS_TO_SAMPLE,
        help="Number of cameras per scene (matches optimal_cameras_filtered.json keys). "
        "Fixed per run -- the planned k=3..10 ablation means separate runs, not a "
        "single run varying k.",
    )
    parser.add_argument(
        "--frames_per_scene",
        type=int,
        default=cfg.FRAMES_PER_SCENE_PER_EPOCH,
        help="Frames sampled per TRAIN scene per epoch (resampled -- see --resample_every_n_epochs).",
    )
    parser.add_argument(
        "--val_frames_per_scene",
        type=int,
        default=cfg.VAL_FRAMES_PER_SCENE,
        help="Frames sampled per VAL scene, ONCE per stage (fixed for that stage's entire run).",
    )
    parser.add_argument(
        "--resample_every_n_epochs",
        type=int,
        default=cfg.RESAMPLE_EVERY_N_EPOCHS,
        help="Redraw each train scene's frame subset every N epochs. 1 = every epoch.",
    )
    parser.add_argument("--patience1", type=int, default=cfg.PATIENCE_STAGE_1)
    parser.add_argument("--patience2", type=int, default=cfg.PATIENCE_STAGE_2)
    parser.add_argument("--patience3", type=int, default=cfg.PATIENCE_STAGE_3)
    parser.add_argument("--patience4", type=int, default=cfg.PATIENCE_STAGE_4)
    parser.add_argument(
        "--skip_camera_validation",
        action="store_true",
        help="Skip log_and_validate_camera_setup at startup (faster iteration once "
        "a camera config has already been validated once -- NOT recommended for "
        "a run whose numbers will go in the thesis).",
    )

    # Loss and oracle-mask overrides
    parser.add_argument("--bg_suppression_weight", type=float, default=None)
    parser.add_argument(
        "--dense_out_weight",
        type=float,
        default=0.0,
        help="Stage 2: weight of the dense outside-mask term toward the TRUE density (0 = original loss). "
        "Try 0.02-0.5; the term is ~1.3x weight at the current plateau, vs ~0.013 for the in-mask MSE.",
    )
    parser.add_argument(
        "--count_loss_weight",
        type=float,
        default=None,
        help="Stage 2: weight of the per-patch L1 total-mass-matching term (dense_count_loss + "
        "sparse_count_loss), default COUNT_LOSS_WEIGHT=0.01. Raising this pushes the model harder "
        "toward matching TOTAL count even where the per-pixel shape stays imperfect -- a direct "
        "lever on the systematic undercounting seen in dense/sparse masked totals, orthogonal to "
        "--dense_out_weight (which only targets the background pedestal, not in-zone mass).",
    )
    parser.add_argument("--ot_weight", type=float, default=None)
    parser.add_argument("--stage4_mask_anchor_weight", type=float, default=None)
    parser.add_argument("--bg_suppression_mode", type=str, default=None, choices=["full", "ring"])
    parser.add_argument("--ring_width", type=int, default=None)
    parser.add_argument(
        "--oracle_min_extent",
        type=int,
        default=None,
        help="Oracle dense-zone mask: morphological opening with a k x k square (odd k, BEV px; "
        "5 px = 1 m). Only regions the k x k window fits inside stay dense. Default "
        "ORACLE_MIN_EXTENT=1 (off). See utils/oracle_mask.py.",
    )
    parser.add_argument(
        "--oracle_min_area",
        type=int,
        default=None,
        help="Oracle dense-zone mask: drop connected components smaller than this many BEV px. "
        "Default ORACLE_MIN_AREA=0 (off).",
    )
    parser.add_argument(
        "--oracle_density_window",
        type=int,
        default=None,
        help="Oracle dense-zone mask: odd window (BEV px) over which density is averaged; with > 1 the "
        'mask is "mean density > --oracle_min_density persons/m^2" instead of the pointwise '
        "peak threshold. Default ORACLE_DENSITY_WINDOW=1 (off).",
    )
    parser.add_argument(
        "--oracle_min_density", type=float, default=None, help="persons/m^2 threshold for --oracle_density_window."
    )
    parser.add_argument(
        "--oracle_occupied_only",
        action="store_true",
        help="With --oracle_density_window: also require the pixel to be within ~1 m of a person.",
    )
    parser.add_argument(
        "--stage2_unmasked",
        action="store_true",
        help="Stage 2: train BOTH experts on the full patch (no oracle-mask split); checkpoint selection "
        "becomes the per-cell routing upper bound unless --stage2_select sparse_f1. Also applies to "
        "--sparse_only fine-tunes (e.g. genloss on the full patch).",
    )
    parser.add_argument(
        "--stage3_target",
        type=str,
        default="oracle",
        choices=["oracle", "performance"],
        help="Stage 3 router target: 'oracle' (density-defined mask, Tversky/Focal) or 'performance' "
        "(per --route_cell cell, which frozen expert's count is closer to the GT; weighted BCE).",
    )
    parser.add_argument(
        "--route_cell",
        type=int,
        default=None,
        help="Cell size in BEV px for performance routing / the Stage 2 upper bound (default ROUTE_CELL=10, "
        "= 2 m; must divide the 200 px patch).",
    )
    parser.add_argument("--mass_normalize_dense", action="store_true")
    parser.add_argument("--mass_normalize_sparse", action="store_true")
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--stage3_oracle_weight", type=float, default=None)
    parser.add_argument("--stage3_perf_weight", type=float, default=None)
    parser.add_argument("--focal_alpha", type=float, default=None)
    parser.add_argument("--focal_gamma", type=float, default=None)
    parser.add_argument("--stage3_loss_type", type=str, default=None, choices=["dice", "focal"])
    parser.add_argument("--tversky_alpha", type=float, default=None)
    parser.add_argument("--tversky_beta", type=float, default=None)
    parser.add_argument("--log_file", type=str, default=None)

    # Fixed validation set + per-scene validation logs
    parser.add_argument(
        "--val_seed",
        type=int,
        default=cfg.VAL_SEED_DEFAULT,
        help='Seed for building the cached validation set (per-scene: "<seed>:<scene>"). Makes every '
        "stage validate on identical frames/cameras/crops. Negative = disable (old behaviour).",
    )
    parser.add_argument(
        "--val_log_dir",
        type=str,
        default=None,
        help="Directory for the val manifest and per-scene validation CSV (default: src/val_logs).",
    )

    # Mid-stage resume (single-stage mode)
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Epochs already completed for this stage (with --resume <checkpoint at that epoch>). "
        "Fast-forwards the cosine LR schedule and epoch numbering. Approximate: optimizer state "
        "is not restored.",
    )
    parser.add_argument(
        "--best_metric_so_far",
        type=float,
        default=None,
        help="Best selection metric already achieved in the stopped run (from its log), so a worse "
        "epoch does not overwrite the saved best checkpoint.",
    )

    # Sparse-expert detection experiments (Stage 2)
    parser.add_argument(
        "--sparse_only", action="store_true", help="Train only the sparse expert (single --stage 2 run)."
    )
    parser.add_argument(
        "--precise_bn_frames",
        type=int,
        default=0,
        help=">0: before each validation, re-estimate BatchNorm running statistics of the layers training in this stage "
        "as a cumulative average over this many random frames of every train scene (~15 s/epoch for 1). Recommended: 1.",
    )
    parser.add_argument(
        "--router_only",
        action="store_true",
        help="Stage 4 only: train just gating_net against the fused-map loss (neck and experts frozen).",
    )
    parser.add_argument(
        "--dense_only",
        action="store_true",
        help="Train only the dense expert (single --stage 2 run); pair with --bg_suppression_weight.",
    )
    parser.add_argument(
        "--sparse_loss",
        choices=["mse", "focal", "genloss"],
        default="mse",
        help="'focal' = penalty-reduced focal loss on a sigmoid heatmap; 'genloss' = Wan et al.'s "
        "Generalized (UOT) loss restricted to the sparse-gated region (single --stage 2 run).",
    )
    parser.add_argument("--sparse_focal_weight", type=float, default=1.0)
    parser.add_argument(
        "--sparse_genloss_weight",
        type=float,
        default=1.0,
        help="--sparse_loss genloss: weight on the GeneralizedLoss term (replaces sparse_mse).",
    )
    parser.add_argument(
        "--sparse_genloss_topk",
        type=int,
        default=1000,
        help="--sparse_loss genloss: cap on predicted 'mass' points fed to the OT solve, to bound "
        "cost-matrix memory when the sparse region covers most of the patch.",
    )
    parser.add_argument(
        "--sparse_genloss_bg_mode",
        choices=["mse", "mass"],
        default="mse",
        help="'mass' = predicted mass on empty ground in count units (sum / GAUSSIAN_SCALE) instead of "
        "the pixel-mean MSE, which is too small to matter next to the OT term",
    )
    parser.add_argument(
        "--sparse_bg_empty_thr",
        type=float,
        default=1e-6,
        help="empty-ground threshold on the target map (peak 1 per person): 1e-6 ~ 2.4 m, 1e-3 ~ 1.7 m from anyone",
    )
    parser.add_argument(
        "--sparse_genloss_bg_weight",
        type=float,
        default=0.0,
        help="--sparse_loss genloss: weight on an MSE-to-zero penalty for sparse_pred on truly-empty "
        "sparse-region pixels, to counteract genloss's tendency to over-predict peaks everywhere "
        "(no elementwise false-positive penalty in the OT term itself).",
    )
    parser.add_argument(
        "--sparse_genloss_count_weight",
        type=float,
        default=0.0,
        help="--sparse_loss genloss: extra weight on sparse_count_loss (mass calibration), on top of "
        "the much smaller shared COUNT_LOSS_WEIGHT -- fixes amplitude collapse (model predicting "
        "near-zero density almost everywhere since genloss's OT cost is scale-invariant).",
    )
    parser.add_argument("--sparse_cell_loss_weight", type=float, default=0.0)
    parser.add_argument("--sparse_cell_sizes", type=str, default="8,16")
    parser.add_argument(
        "--stage2_select",
        choices=["combined", "sparse_f1", "experts", "dense"],
        default="combined",
        help="experts = mean of dense-alone and sparse-alone count MAE (needs --stage2_unmasked)",
    )
    parser.add_argument(
        "--stage3_select",
        choices=["mass", "soft_mae"],
        default="mass",
        help="soft_mae = learned soft-routing count MAE (needs --stage3_target performance)",
    )
    parser.add_argument(
        "--dense_loss",
        choices=["mse", "ot"],
        default="mse",
        help="Stage 2 dense expert loss: 'ot' = the SynMVCrowd baseline's OT loss (genloss_synmvcrowd, "
        "cost ed) on its map in count x 100 units, replacing the dense MSE",
    )
    parser.add_argument(
        "--freeze_adapter",
        action="store_true",
        help="--dense_only: keep the ablation neck's adapter frozen (a sparse expert trained on top stays valid)",
    )
    parser.add_argument(
        "--freeze_fusion",
        action="store_true",
        help="Stage 2 with an ablation neck: keep the fusion frozen (eval mode); the adapter and both "
        "experts train (--dense_only already freezes it)",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help=">0: validate, select and save an exponential moving average of the trainable weights "
        "(per step; the ablation / H0 recipe uses 0.999). 0 = raw weights (default)",
    )
    parser.add_argument(
        "--select_window",
        type=int,
        default=1,
        help="checkpoint selection and early stopping on the mean of the last N validated epochs",
    )
    parser.add_argument(
        "--seed_metric_history",
        type=str,
        default="",
        help="comma-separated raw selection metrics of the last epochs before a --start_epoch resume, "
        "so the --select_window average continues across it",
    )
    parser.add_argument("--sparse_nms_mult", type=float, default=None)
    parser.add_argument("--sparse_nms_kernel", type=int, default=None)
    parser.add_argument("--sparse_match_px", type=float, default=5.0)
    parser.add_argument(
        "--keep_sparse_bias",
        action="store_true",
        help="With --sparse_loss focal: do not re-initialise the final-conv bias (use when resuming "
        "from a checkpoint that was already trained in heatmap mode).",
    )
    parser.add_argument("--ckpt_tag", type=str, default="", help="Suffix for checkpoint names, e.g. _pA.")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory for the checkpoints (default: src).")
    parser.add_argument(
        "--sparse_render",
        choices=["raw", "peaks"],
        default=None,
        help="How the sparse expert enters the fusion (Stages 3/4, eval): 'peaks' = one map kernel per NMS "
        "peak (doc/head_ablation.md 6b). Default: the resumed checkpoint's, else raw. Stored in checkpoints.",
    )
    parser.add_argument("--sparse_render_kernel", type=int, default=3, help="NMS kernel for --sparse_render peaks")
    parser.add_argument(
        "--sparse_render_threshold",
        type=float,
        default=0.064,
        help="Peak threshold for --sparse_render peaks (map units, one person peaks at 1.0)",
    )
    parser.add_argument(
        "--neck_adapter",
        choices=["conv64", "none"],
        default=None,
        help="ablation neck: 'none' drops the head_processor (256 -> 64 ch), so the head reads the "
        "fusion's channels directly. Overrides a resumed checkpoint's neck config; stored in new "
        "checkpoints.",
    )
    parser.add_argument(
        "--dense_arch",
        choices=["legacy", "wide512"],
        default=None,
        help="'wide512' = the baseline decoder's 512-channel stack as the dense expert (fresh init). "
        "Overrides a resumed checkpoint's config; stored in new checkpoints (neck_config).",
    )
    parser.add_argument(
        "--sparse_arch",
        choices=["shallow", "deep"],
        default="shallow",
        help="'deep' = 5-conv sparse expert with 15 px receptive field (fresh init when loading a "
        "shallow checkpoint). Later stages / evaluation of such a checkpoint need the same flag.",
    )
    parser.add_argument(
        "--full_scene_val_frames",
        type=int,
        default=0,
        help="Stages 2-4: each epoch also score this many frames per val scene on the WHOLE scene "
        "(paper protocol: random 5-view draws, tiled, learned routing; oracle in Stage 2) and "
        "select the checkpoint on that MAE instead of the patch metric. 0 = off.",
    )
    parser.add_argument(
        "--full_scene_val_draws", type=int, default=2, help="Random camera draws per frame for --full_scene_val_frames."
    )
    parser.add_argument(
        "--train_backbone",
        choices=["none", "layer4", "all"],
        default="none",
        help="Backbone layers that train in Stages 1/2/4: 'none' (frozen, default), 'layer4' (the "
        "regime every pre-stride-8 checkpoint was trained in) or 'all' (as in the SynMVCrowd paper).",
    )
    parser.add_argument(
        "--backbone_lr_mult",
        type=float,
        default=1.0,
        help="Backbone learning rate as a multiple of the stage learning rate (with --train_backbone).",
    )
    parser.add_argument(
        "--reinit_sparse",
        action="store_true",
        help="Train the sparse expert from a fresh initialisation (fair control for --sparse_arch deep).",
    )
    parser.add_argument("--lr", type=float, default=None, help="Override the stage learning rate (single-stage runs).")

    # Weather
    parser.add_argument(
        "--train_weather",
        type=str,
        default=None,
        help="Comma-separated weather values to KEEP for training frames (e.g. EXTRASUNNY,CLEAR). "
        "Values seen in the data: CLEAR, EXTRASUNNY, CLOUDS, OVERCAST, RAIN, FOGGY, THUNDER, XMAS, EMPTY "
        "(EMPTY = weather field is an empty string). Needs utils/weather_index.json. Default: all.",
    )
    parser.add_argument(
        "--val_weather",
        type=str,
        default=None,
        help="Same, for the validation registry. Changes the validation set (new fingerprint) -- "
        "leave unset for all-weather validation.",
    )

    # Timing (off by default)
    parser.add_argument(
        "--enable_timing",
        action="store_true",
        help='Turn on all "[timing]" console output, CSVs (timing_logs/) and the end-of-run JSON '
        "summary. Off by default to keep logs focused on training signal.",
    )
    parser.add_argument(
        "--timing_dir",
        type=str,
        default=None,
        help="Directory for timing CSV/JSON output (default: src/timing_logs). Only used with --enable_timing.",
    )
    parser.add_argument(
        "--no_phase_timing",
        action="store_true",
        help="With --enable_timing: disable per-phase timing inside the train loop (removes the CUDA "
        "synchronizes between phases). Per-scene/epoch/stage timing stays on.",
    )
    return parser
