"""Hyperparameters and run settings of the proposed-model trainer.

Every part of the trainer reads these at call time (cfg.NAME), so a value set here applies everywhere.
train_proposed_multiscene.py forwards reads and writes of these names to this module, which keeps scripts that
do `T.SPARSE_ARCH = 'deep'` working. The defaults reproduce the recipe of the existing checkpoints; apply_args()
sets them from the command line.
"""

import datetime
import os

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Data sampling -------------------------------------------------------------------------------------------------
NUM_CAMS_TO_SAMPLE = 5  # k cameras per scene, from optimal_cameras_filtered.json (--k)
FRAMES_PER_SCENE_PER_EPOCH = 15  # train frames per scene, redrawn every RESAMPLE_EVERY_N_EPOCHS
VAL_FRAMES_PER_SCENE = 5  # val frames per scene, drawn once per stage
RESAMPLE_EVERY_N_EPOCHS = 1
BATCH_SIZE = 1  # one frame per step: scenes differ in grid size, so frames of two scenes cannot batch
NUM_WORKERS = 0
BEV_SHAPE = (200, 200)  # training patch (tile) size in BEV px

# --- Optimisation --------------------------------------------------------------------------------------------------
LR_STAGE_1 = 1e-4
LR_STAGE_2 = 1e-4
LR_STAGE_3 = 5e-4
LR_STAGE_4 = 1e-5
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 5.0
PATIENCE_STAGE_1 = 15
PATIENCE_STAGE_2 = 15
PATIENCE_STAGE_3 = 15
PATIENCE_STAGE_4 = 15

# --- Target maps and the oracle dense-zone mask (utils/oracle_mask.py) -----------------------------------------------
GAUSSIAN_SCALE = 100
GAUSSIAN_SIGMA = 5
GAUSSIAN_KERNEL_SIZE = 20
ROUTE_LOSS_WEIGHT = 1.0
DENSITY_PEAK_MULTIPLIER = 2.0
MASK_SMOOTHING_KERNEL = 3
ORACLE_MIN_EXTENT = 1  # odd k: morphological opening with a k x k square (1 = off)
ORACLE_MIN_AREA = 0  # drop dense components smaller than this many px (0 = off)
ORACLE_DENSITY_WINDOW = 1  # > 1: dense where the mean density over this odd window > ORACLE_MIN_DENSITY p/m^2
ORACLE_MIN_DENSITY = None
ORACLE_OCCUPIED_ONLY = False  # also require the pixel to be within ~1 m of a person
ORACLE_OCCUPIED_PEAKS = 0.1  # "occupied" = map_gt_conv above this many single-person peaks
BEV_PX_PER_M = 5.0  # SynMVCrowdDataset rasterises every scene at 5 px/m
MASS_REL_ERROR_FLOOR = 10.0  # Stage 3 relative mass error: denominator floor, so near-empty patches cannot dominate

# --- Losses --------------------------------------------------------------------------------------------------------
OT_SCALE = 10.0
OT_WEIGHT = 0.0  # Stage 4 OT term; at 0 the (slow) OT loss is not even called
STAGE4_MASK_ANCHOR_WEIGHT = 0.0  # Stage 4: keep the router close to the oracle mask
COUNT_MAE_WEIGHT = 0.01
LOCALIZATION_WEIGHT = 0.1
PEAK_THRESHOLD_FRACTION = 0.1
COUNT_LOSS_WEIGHT = 0.01  # Stage 2: L1 between predicted and true mass per patch, both experts
SPARSE_PEAK_THRESHOLD_MULTIPLIER = 5
SPARSE_NMS_KERNEL_SIZE = 3
BG_SUPPRESSION_WEIGHT = 0.0  # Stage 2: push the dense output to 0 outside its region ('ring' = a band around it)
BG_SUPPRESSION_MODE = "ring"
RING_WIDTH = 5
DENSE_OUT_WEIGHT = 0.0  # Stage 2: supervise the dense output toward the true density outside its region
STAGE3_ORACLE_WEIGHT = 1.0
STAGE3_PERF_WEIGHT = 0.0  # legacy soft-label term (RoutingLabelGenerator); unused by every recipe
STAGE3_LOSS_TYPE = "dice"  # router vs oracle mask: 'dice' (Tversky) | 'focal'
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0
TVERSKY_ALPHA = 0.5
TVERSKY_BETA = 0.5

# --- Performance routing (doc/oracle_mask_redesign.md §9) ------------------------------------------------------------
STAGE2_UNMASKED = False  # Stage 2: both experts on the whole patch (no oracle split)
STAGE3_TARGET = "oracle"  # 'oracle' mask | 'performance': per cell, which expert's count is closer to the GT
ROUTE_CELL = 10  # performance-routing cell in BEV px (10 px = 2 m); must divide BEV_SHAPE

# --- Stage 2 expert experiments (doc/head_ablation.md) -------------------------------------------------------------
SPARSE_ONLY = False  # train only the sparse expert
DENSE_ONLY = False  # train only the dense expert
ROUTER_ONLY = False  # Stage 4: train only the router, against the fused-map loss
PRECISE_BN_FRAMES = 0  # > 0: re-estimate BatchNorm statistics over this many frames per scene before validation
SPARSE_LOSS = "mse"  # 'mse' | 'focal' (sigmoid heatmap) | 'genloss' (Wan et al.'s UOT loss)
SPARSE_ACTIVATION = "relu"  # 'sigmoid' with --sparse_loss focal
SPARSE_FOCAL_WEIGHT = 1.0
SPARSE_GENLOSS_WEIGHT = 1.0
SPARSE_GENLOSS_TOPK = 1000  # cap on predicted mass points in the OT solve
SPARSE_GENLOSS_BG_WEIGHT = 0.0  # genloss has no per-pixel false-positive penalty: penalise output on empty ground
SPARSE_GENLOSS_BG_MODE = "mse"  # 'mse' (pixel mean) | 'mass' (empty-ground mass in count units)
SPARSE_BG_EMPTY_THR = 1e-6  # "empty" = target below this (1e-6 ~ 2.4 m, 1e-3 ~ 1.7 m from anyone)
SPARSE_GENLOSS_COUNT_WEIGHT = (
    0.0  # genloss is scale-invariant: weight on the sparse count term (doc/sparse_ot_diagnose.md)
)
FOCAL_HEATMAP_ALPHA = 2.0
FOCAL_HEATMAP_BETA = 4.0
SPARSE_HEATMAP_BIAS_INIT = -5.0  # final-conv bias when switching to sigmoid (p ~ 0.007), unless SPARSE_KEEP_BIAS
SPARSE_KEEP_BIAS = False
SPARSE_CELL_LOSS_WEIGHT = 0.0
SPARSE_CELL_SIZES = (8, 16)
SPARSE_MATCH_PX = 5.0  # matching radius (BEV px) of the validation F1
SPARSE_RENDER = None  # how the sparse expert enters the fusion: None = relu(sparse), {'mode': 'peaks', ...}
SPARSE_ARCH = "shallow"  # 'shallow' (2 convs, 5 px receptive field) | 'deep' (5 convs, 15 px)
REINIT_SPARSE = False  # drop the checkpoint's sparse expert and train it from a fresh init
DENSE_LOSS = "mse"  # 'mse' | 'ot' (the SynMVCrowd baseline's OT loss)
FREEZE_ADAPTER = False  # with DENSE_ONLY: keep the ablation neck's 256 -> 64 adapter frozen
FREEZE_FUSION = False  # Stage 2 on an ablation neck: keep the fusion frozen

# --- Joint run (no curriculum, doc/joint_training.md) --------------------------------------------------------------
JOINT_ROUTER_WEIGHT = 1.0  # w_r: weight of the router's Tversky loss to the oracle mask
JOINT_FUSED_WEIGHT = 1.0  # w_f: weight of the fused-map loss (Stage 4's)
VAL_EVERY = 1  # validate every N epochs (and after the last); patience counts validated epochs

# --- Checkpoint selection ------------------------------------------------------------------------------------------
STAGE2_SELECT = "combined"  # 'combined' | 'sparse_f1' | 'experts' | 'dense'
STAGE3_SELECT = "mass"  # 'mass' (relative mass error) | 'soft_mae' (learned soft-routing count MAE)
SELECT_WINDOW = 1  # select / stop on the mean of the last N validated epochs
SEED_METRIC_HISTORY = ()  # selection metrics of the epochs before a --start_epoch resume (oldest first)
EMA_DECAY = 0.0  # > 0: validate, select and save an EMA of the trainable weights
FULL_SCENE_VAL_FRAMES = 0  # > 0: also score whole val scenes (paper protocol) and select on that MAE
FULL_SCENE_VAL_DRAWS = 2
CKPT_TAG = ""  # suffix of the checkpoint names
CKPT_DIR = src_dir

# --- Architecture --------------------------------------------------------------------------------------------------
TRAIN_BACKBONE = "none"  # 'none' (frozen) | 'layer4' (the pre-stride-8 regime) | 'all'
BACKBONE_LR_MULT = 1.0
BACKBONE_USE_MAXPOOL = True  # stride 8; a legacy stride-4 checkpoint switches it off on resume
ROUTER_CONTRAST = True  # GatingNet contrast branch (--no_router_contrast = fine + context router)
NECK_CONFIG = {"type": "legacy"}  # neck of a fresh model; a resumed checkpoint's neck_config takes precedence
NECK_OVERRIDES = {}  # --neck_adapter / --dense_arch, applied on top of the (resumed) neck config
NECK_GRAD_CHECKPOINT = False  # bevformer neck: recompute encoder layers in the backward pass

# --- Logs ----------------------------------------------------------------------------------------------------------
TIMING_DIR = os.path.join(src_dir, "timing_logs")
TIMING_RUN_TAG = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
TIMING_ENABLED = False  # --enable_timing: all "[timing]" output, CSVs and the JSON summary
PHASE_TIMING = False  # CUDA-synchronised per-phase timing (only with TIMING_ENABLED)
STAGE_TIMING_SUMMARIES = []  # one dict per finished stage
STARTUP_TIMING = {}
VAL_LOG_DIR = os.path.join(src_dir, "val_logs")
VAL_SEED_DEFAULT = 1234
VAL_CACHE_INFO = {}  # {'fingerprint': ...} of the first stage's validation set


# Settings taken from the argument of the same (lower-case) name, always or only when it is given.
_ALWAYS = [
    "SPARSE_ONLY",
    "DENSE_ONLY",
    "ROUTER_ONLY",
    "PRECISE_BN_FRAMES",
    "SPARSE_LOSS",
    "SPARSE_FOCAL_WEIGHT",
    "SPARSE_GENLOSS_WEIGHT",
    "SPARSE_GENLOSS_TOPK",
    "SPARSE_GENLOSS_BG_WEIGHT",
    "SPARSE_GENLOSS_BG_MODE",
    "SPARSE_BG_EMPTY_THR",
    "SPARSE_GENLOSS_COUNT_WEIGHT",
    "SPARSE_CELL_LOSS_WEIGHT",
    "STAGE2_SELECT",
    "STAGE3_SELECT",
    "EMA_DECAY",
    "FREEZE_FUSION",
    "DENSE_LOSS",
    "FREEZE_ADAPTER",
    "SPARSE_MATCH_PX",
    "CKPT_TAG",
    "SPARSE_ARCH",
    "REINIT_SPARSE",
    "TRAIN_BACKBONE",
    "FULL_SCENE_VAL_FRAMES",
    "FULL_SCENE_VAL_DRAWS",
    "NECK_GRAD_CHECKPOINT",
    "BACKBONE_LR_MULT",
    "DENSE_OUT_WEIGHT",
    "STAGE2_UNMASKED",
    "STAGE3_TARGET",
    "JOINT_ROUTER_WEIGHT",
    "JOINT_FUSED_WEIGHT",
    "VAL_EVERY",
]
_IF_GIVEN = [
    "BG_SUPPRESSION_WEIGHT",
    "COUNT_LOSS_WEIGHT",
    "OT_WEIGHT",
    "STAGE4_MASK_ANCHOR_WEIGHT",
    "BG_SUPPRESSION_MODE",
    "RING_WIDTH",
    "ORACLE_MIN_EXTENT",
    "ORACLE_MIN_AREA",
    "ORACLE_DENSITY_WINDOW",
    "ORACLE_MIN_DENSITY",
    "ROUTE_CELL",
    "STAGE3_ORACLE_WEIGHT",
    "STAGE3_PERF_WEIGHT",
    "FOCAL_ALPHA",
    "FOCAL_GAMMA",
    "STAGE3_LOSS_TYPE",
    "TVERSKY_ALPHA",
    "TVERSKY_BETA",
]


def apply_args(args, parser):
    """Set the module's settings from the parsed command line (cli.build_parser) and check their combinations."""
    from .reporting import describe_hardware
    from .targets import oracle_mask_config

    g = globals()
    for name in _ALWAYS:
        g[name] = getattr(args, name.lower())
    for name in _IF_GIVEN:
        if getattr(args, name.lower()) is not None:
            g[name] = getattr(args, name.lower())
    if args.sparse_nms_mult is not None:
        g["SPARSE_PEAK_THRESHOLD_MULTIPLIER"] = args.sparse_nms_mult
    if args.sparse_nms_kernel is not None:
        g["SPARSE_NMS_KERNEL_SIZE"] = args.sparse_nms_kernel
    g.update(
        SPARSE_ACTIVATION="sigmoid" if args.sparse_loss == "focal" else "relu",
        SPARSE_CELL_SIZES=tuple(int(x) for x in args.sparse_cell_sizes.split(",") if x.strip()),
        SELECT_WINDOW=max(1, args.select_window),
        SEED_METRIC_HISTORY=tuple(float(x) for x in args.seed_metric_history.split(",") if x.strip()),
        SPARSE_KEEP_BIAS=args.keep_sparse_bias,
        ROUTER_CONTRAST=not args.no_router_contrast,
        TIMING_ENABLED=args.enable_timing,
        PHASE_TIMING=args.enable_timing and not args.no_phase_timing,
        ORACLE_OCCUPIED_ONLY=ORACLE_OCCUPIED_ONLY or args.oracle_occupied_only,
    )
    if args.ckpt_dir:
        g["CKPT_DIR"] = os.path.abspath(args.ckpt_dir)
        os.makedirs(CKPT_DIR, exist_ok=True)
    if args.timing_dir:
        g["TIMING_DIR"] = args.timing_dir
    if args.val_log_dir:
        g["VAL_LOG_DIR"] = args.val_log_dir
    if args.neck_adapter is not None:
        NECK_OVERRIDES["adapter"] = args.neck_adapter == "conv64"
    if args.dense_arch is not None:
        NECK_OVERRIDES["dense_arch"] = args.dense_arch
    if args.sparse_render is not None:
        g["SPARSE_RENDER"] = (
            {"mode": "peaks", "kernel": args.sparse_render_kernel, "threshold": args.sparse_render_threshold}
            if args.sparse_render == "peaks"
            else {}
        )
    if args.neck == "bevformer":
        g["NECK_CONFIG"] = {
            "type": "bevformer",
            "num_layers": args.neck_layers,
            "num_heads": args.neck_heads,
            "points_per_anchor": args.neck_points_per_anchor,
            "sa_points": args.neck_sa_points,
            "ffn_dim": args.neck_ffn_dim,
            "dropout": args.neck_dropout,
        }
    elif args.neck == "ablation":  # a fresh neck-ablation fusion (the curriculum gets a trained one from Stage 1)
        g["NECK_CONFIG"] = {"type": "ablation", "variant": args.neck_variant, "z_anchors": 4}

    if TIMING_ENABLED:
        print(f"[timing] Hardware: {describe_hardware()}")
        print(
            f"[timing] Timing logs -> {TIMING_DIR} (run tag {TIMING_RUN_TAG}, phase timing "
            f"{'ON' if PHASE_TIMING else 'OFF'})"
        )
    if ORACLE_DENSITY_WINDOW > 1 and ORACLE_MIN_DENSITY is None:
        parser.error("--oracle_density_window > 1 needs --oracle_min_density (persons/m^2)")
    print(f"[oracle mask] {oracle_mask_config()}")
    if BEV_SHAPE[0] % ROUTE_CELL or BEV_SHAPE[1] % ROUTE_CELL:
        parser.error(f"--route_cell {ROUTE_CELL} must divide the patch size {BEV_SHAPE}")
    if STAGE2_UNMASKED and args.full_scene_val_frames > 0 and args.stage == 2:
        parser.error(
            "--full_scene_val_frames scores Stage 2 with the density oracle mask, meaningless for --stage2_unmasked"
        )
    if STAGE2_UNMASKED or STAGE3_TARGET == "performance":
        print(
            f"[routing] stage2_unmasked={STAGE2_UNMASKED} | stage3_target={STAGE3_TARGET} | route_cell={ROUTE_CELL} px "
            f"({ROUTE_CELL / 5:g} m)"
        )

    if args.joint:
        if args.stage is not None or args.full_curriculum:
            parser.error("--joint is its own schedule; leave out --stage and --full_curriculum.")
        if args.sparse_only or args.dense_only or args.router_only or args.freeze_fusion:
            parser.error("--joint trains every part; the single-part probes do not apply.")
        if args.sparse_loss == "focal" or STAGE3_TARGET == "performance":
            parser.error("--joint supports --sparse_loss mse / genloss and the oracle router target only.")
        if STAGE4_MASK_ANCHOR_WEIGHT > 0.0:
            parser.error("--joint already holds the router to the oracle mask (--joint_router_weight); no anchor.")
    if args.val_every < 1:
        parser.error("--val_every must be >= 1")
    if args.full_curriculum and (
        args.sparse_only or args.dense_only or args.sparse_loss != "mse" or args.lr is not None
    ):
        parser.error(
            "--sparse_only / --dense_only / --sparse_loss focal / --lr are for a single --stage 2 run, not --full_curriculum."
        )
    stage2_flags = args.sparse_only or args.dense_only or args.sparse_loss != "mse" or args.freeze_fusion
    if stage2_flags and args.stage != 2 and not args.joint:
        parser.error("--sparse_only, --dense_only and --sparse_loss focal only apply to --stage 2.")
    if args.router_only and (args.full_curriculum or args.stage != 4):
        parser.error("--router_only only applies to a single --stage 4 run.")
    if args.sparse_only and args.dense_only:
        parser.error("Use either --sparse_only or --dense_only, not both.")
    if args.full_curriculum and args.start_epoch:
        parser.error("--start_epoch only works with a single --stage run, not --full_curriculum.")
    if not args.full_curriculum and not args.joint and args.stage is None:
        parser.error("Provide either --stage N (single stage), --full_curriculum (all 4 stages) or --joint.")
