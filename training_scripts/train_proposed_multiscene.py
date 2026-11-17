"""
Multi-scene curriculum training of the proposed model (backbone + neck + hybrid routing head).

    Stage 1  neck (with a throwaway head) on the density map
    Stage 2  dense and sparse experts, each against its region of the oracle dense-zone mask
             (--stage2_unmasked: both on the whole patch)
    Stage 3  router on the frozen neck and experts: against the oracle mask, or per-cell performance
             labels (--stage3_target performance)
    Stage 4  everything jointly on the fused map (--router_only: only the router)
    --joint  no curriculum: neck, experts and router together on the Stage 2 + router + Stage 4 losses

Each stage resumes from the previous stage's best checkpoint; --full_curriculum chains all four:

    uv run python training_scripts/train_proposed_multiscene.py --stage 2 --resume <stage-1 checkpoint> --ckpt_tag _x

Every epoch draws fresh frames from every train scene (random 200 x 200 patches). Validation uses a fixed set of
patches from the held-out val scenes (--val_seed), so all stages and runs are scored on the same patches; the
checkpoint is selected per stage on the metric in proposed_trainer/validation.py. Checkpoints:
multiscene_{best,final}_stage_N<tag>.pth and every 5th epoch, in --ckpt_dir (default src/).

The building blocks live in proposed_trainer/ (overview in its __init__.py), every setting in
proposed_trainer/config.py. Scripts importing this module (`import train_proposed_multiscene as T`) keep working:
the old names are re-exported, and reads and writes of settings (T.SPARSE_ARCH = 'deep') go to the config.
"""

import datetime
import gc
import math
import os
import random
import sys
import time
import types
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)

# re-exported for the scripts that use them through this module
from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset, NoFramesAfterFilter  # noqa: E402,F401
from losses.gaussian_mse import GaussianMSE_norm, get_map_kernel  # noqa: E402,F401
from losses.ot_loss.genloss import GeneralizedLoss  # noqa: E402,F401
from losses.ot_loss.genloss_synmvcrowd import SynMVCrowdGeneralizedLoss, patch_points  # noqa: E402,F401
from models.backbone.feature_extractor import DilatedResNet18  # noqa: E402,F401
from models.head.proposed.routing import HybridRoutingHead  # noqa: E402,F401
from models.neck.proposed.bev_encoder import BEVFormerEncoderNeck  # noqa: E402,F401
from models.neck.proposed.spatial_cross_attention import SpatialCrossAttentionNeck  # noqa: E402,F401
from utils.exclusions import load_excluded_scenes, load_excluded_frames  # noqa: E402,F401
from utils.oracle_mask import compute_oracle_mask  # noqa: E402,F401
from training_scripts.proposed_trainer import config as cfg  # noqa: E402
from training_scripts.proposed_trainer.cli import build_parser  # noqa: E402
from training_scripts.proposed_trainer.data import (  # noqa: E402,F401
    build_cached_val_samples,
    build_full_scene_val_set,
    build_registries,
    build_scene_registry,
    build_stage_val_sets,
    full_scene_validation,
    load_camera_selection,
    log_and_validate_camera_setup,
    sample_train_subset_loader,
    validate_excluded_frame_keys,
)
from training_scripts.proposed_trainer.losses import (  # noqa: E402,F401
    RoutingLabelGenerator,
    StageLosses,
    binary_focal_loss,
    mask_loss,
    normalized_mse_loss,
    penalty_reduced_focal_loss,
    relative_cell_loss,
    tversky_loss,
    weighted_routing_bce,
)
from training_scripts.proposed_trainer.model import (  # noqa: E402,F401
    AblationNeck,
    ProposedModel,
    apply_sparse_render,
    build_model_for_checkpoint,
    build_neck,
    build_sparse_expert,
    checkpoint_neck_config,
    checkpoint_sparse_arch,
    checkpoint_uses_maxpool,
    load_resume_weights,
    neck_ctx,
)
from training_scripts.proposed_trainer.reporting import (  # noqa: E402,F401
    PhaseTimer,
    Tee,
    append_csv_row,
    describe_hardware,
    format_duration,
    print_and_save_timing_report,
    print_epoch_timing,
    record_stage_timing,
    report_joint_train_terms,
    report_scene_breakdown,
    summarize_scene_stats,
    tick,
    timing_csv_path,
    val_scenes_csv_path,
    write_scene_rows_csv,
)
from training_scripts.proposed_trainer.targets import (  # noqa: E402,F401
    PatchTargets,
    cell_routing_targets,
    count_matches_np,
    create_ground_truth_heatmap,
    extract_camera_locations,
    extract_peaks_via_nms,
    extract_target_coordinates,
    filter_points_outside_mask,
    oracle_density_threshold,
    oracle_mask_config,
    oracle_mask_from_conv,
    project_cameras_to_patch,
    region_points,
)
from training_scripts.proposed_trainer.training import (  # noqa: E402,F401
    Selector,
    WeightEMA,
    build_optimizer,
    ckpt_file,
    fast_forward_schedule,
    recalibrate_batchnorm,
    save_checkpoint,
    set_train_modes,
)
from training_scripts.proposed_trainer.validation import Validation  # noqa: E402


class _SettingsModule(types.ModuleType):
    """Makes the settings in proposed_trainer.config readable and writable as attributes of this module."""

    def __getattr__(self, name):
        try:
            return getattr(cfg, name)
        except AttributeError:
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}") from None

    def __setattr__(self, name, value):
        if (name.isupper() or name == "src_dir") and hasattr(cfg, name) and name not in vars(self):
            setattr(cfg, name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _SettingsModule


class TrainStats:
    """What one training epoch logs."""

    def __init__(self):
        self.timer = PhaseTimer(enabled=cfg.PHASE_TIMING)
        self.loss = self.dense_loss = self.sparse_loss = 0.0
        self.batch_dense_loss = self.batch_sparse_loss = 0.0  # Stage 2, summed over the frame's patches
        self.train_dense_errors, self.train_sparse_errors = [], []
        self.n_batches = self.iters = 0
        self.scene_rows = []  # (scene, iterations, seconds)
        self.empty_cache_s = 0.0
        self.joint_terms, self.joint_patches = defaultdict(float), 0  # --joint: unweighted loss terms, summed


def build_stage_model(stage, k, checkpoint_path, pretrained_backbone_path, device):
    """ProposedModel for this stage. A resumed checkpoint fixes the backbone layout (stride 4 for every checkpoint
    before the stride-8 switch), the neck and the sparse rendering; the 2D-pretrained backbone is only loaded for a
    fresh model (the checkpoint's backbone replaces it anyway)."""
    use_maxpool, neck_config, resume_sparse_render = cfg.BACKBONE_USE_MAXPOOL, cfg.NECK_CONFIG, None
    if checkpoint_path and os.path.isfile(checkpoint_path):
        resume_ckpt = torch.load(checkpoint_path, map_location="cpu")
        use_maxpool = checkpoint_uses_maxpool(resume_ckpt["model_state_dict"])
        neck_config = checkpoint_neck_config(resume_ckpt)
        if neck_config != cfg.NECK_CONFIG:
            print(
                f"[neck] --resume checkpoint holds neck {neck_config}; using it instead of the command line's {cfg.NECK_CONFIG}"
            )
        resume_sparse_render = resume_ckpt.get("sparse_render")
        del resume_ckpt
        pretrained_backbone_path = None
    print(
        f"[backbone] layout: {'stride 8 (MaxPool)' if use_maxpool else 'legacy stride 4 (no MaxPool)'} | "
        f"trainable: {cfg.TRAIN_BACKBONE} | LR mult: {cfg.BACKBONE_LR_MULT}"
    )
    if cfg.NECK_OVERRIDES:
        neck_config = {**neck_config, **cfg.NECK_OVERRIDES}
        print(f"[neck] command-line overrides {cfg.NECK_OVERRIDES} -> {neck_config}")
    model = ProposedModel(
        num_cam=k, pretrained_backbone_path=pretrained_backbone_path, use_maxpool=use_maxpool, neck_config=neck_config
    ).to(device)
    sparse_render = cfg.SPARSE_RENDER if cfg.SPARSE_RENDER is not None else resume_sparse_render  # flag wins
    apply_sparse_render(model, sparse_render)
    print(f"[sparse render] {sparse_render or 'raw relu(sparse) map'}")
    print(
        f"[neck] {getattr(model.neck, 'describe', lambda: 'legacy: 1x deformable self-attention + 1x cross-attention')()} | "
        f"{sum(p.numel() for p in model.neck.parameters()):,} params"
        f"{' | gradient checkpointing' if getattr(model.neck, 'grad_checkpoint', False) else ''}"
    )
    model.set_stage_trainability(stage)
    return model


def prepare_probes(model, stage, lr):
    """Sigmoid-heatmap bias init (--sparse_loss focal) and a summary line for the single-part probes."""

    def n_trainable():
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if stage == 2 and cfg.SPARSE_ACTIVATION == "sigmoid" and not cfg.SPARSE_KEEP_BIAS:
        leaves = [m for m in model.head.sparse_expert.modules() if len(list(m.children())) == 0]
        if leaves and isinstance(leaves[-1], nn.Conv2d) and leaves[-1].bias is not None:
            leaves[-1].bias.data.fill_(cfg.SPARSE_HEATMAP_BIAS_INIT)
            print(
                f"[sparse] Sigmoid-heatmap mode: final conv bias set to {cfg.SPARSE_HEATMAP_BIAS_INIT} "
                f"(initial p ~ {1 / (1 + math.exp(-cfg.SPARSE_HEATMAP_BIAS_INIT)):.3f})."
            )
        else:
            print(
                "WARNING: [sparse] could not find a final Conv2d with bias in sparse_expert; "
                "heatmap bias NOT initialised (the expert may end in another layer type)."
            )
    if cfg.ROUTER_ONLY and stage == 4:
        print(
            f"[router] --router_only: {n_trainable():,} trainable parameters (gating_net only); loss = Stage 4 fused-map MSE; "
            f"lr={lr:g}; tag='{cfg.CKPT_TAG}'"
        )
    if cfg.FREEZE_FUSION and stage == 2:
        assert isinstance(model.neck, AblationNeck), (
            "--freeze_fusion needs an ablation neck (--resume an _abl_ Stage 1)"
        )
        n_fusion = sum(p.numel() for p in model.neck.fusion.parameters())
        print(
            f"[neck] --freeze_fusion: fusion ({n_fusion:,} parameters) frozen in eval mode; "
            f"{n_trainable():,} trainable parameters (adapter + experts)"
        )
    if cfg.DENSE_ONLY and stage == 2:
        n_adapter = (
            sum(p.numel() for p in model.neck.head_processor.parameters() if p.requires_grad)
            if isinstance(model.neck, AblationNeck)
            else 0
        )
        bg = (
            "OFF"
            if cfg.BG_SUPPRESSION_WEIGHT <= 0
            else f"{cfg.BG_SUPPRESSION_MODE}, weight {cfg.BG_SUPPRESSION_WEIGHT}, ring width {cfg.RING_WIDTH}"
        )
        print(
            f"[dense] --dense_only: {n_trainable():,} trainable parameters (dense expert"
            f"{f' + {n_adapter:,} adapter' if n_adapter else ' only'}); dense loss: {cfg.DENSE_LOSS}; background suppression: "
            f"{bg}; tag='{cfg.CKPT_TAG}'"
        )
    if cfg.SPARSE_ONLY and stage == 2:
        print(
            f"[sparse] --sparse_only: {n_trainable():,} trainable parameters (sparse expert only); "
            f"loss={cfg.SPARSE_LOSS}, cell weight={cfg.SPARSE_CELL_LOSS_WEIGHT}, select={cfg.STAGE2_SELECT}, tag='{cfg.CKPT_TAG}'"
        )


def train_one_epoch(model, stage, epoch, epochs, loaders, optimizer, scaler, ema, losses, device):
    """One pass over every train scene in random order, one optimiser step per frame."""
    stats = TrainStats()
    timer = stats.timer
    scene_order = list(range(len(loaders)))
    random.shuffle(scene_order)
    for scene_idx in scene_order:
        scene_name, loader = loaders[scene_idx]
        pbar = tqdm(loader, desc=f"Stage {stage} | Epoch {epoch + 1}/{epochs} | {scene_name}")
        scene_start = t_prev = time.perf_counter()  # t_prev: end of the previous step (until the batch = data wait)
        scene_iters = 0
        for sample in pbar:
            t_batch = time.perf_counter()
            timer.add("data_wait", t_batch - t_prev)  # NUM_WORKERS=0: the full synchronous __getitem__ cost
            wld_paras = sample["wld_map_paras"].to(device)
            hw_random = sample["hw_random"].to(device)
            patch_num = hw_random.shape[-1]
            cam_locations = extract_camera_locations(sample["camera_paras"].to(device)).squeeze(0)
            t_h2d = tick(cfg.PHASE_TIMING)
            timer.add("h2d", t_h2d - t_batch)

            with torch.amp.autocast("cuda"):
                predicted_patches, predicted_masks = model(
                    sample["images"].to(device),
                    sample["ref_points_full"].to(device),
                    hw_random,
                    stage=stage,
                    camera_paras=sample["camera_paras"].to(device),
                    wld_map_paras=wld_paras,
                )
            predicted_patches, predicted_masks = predicted_patches.float(), predicted_masks.float()
            t_fwd = tick(cfg.PHASE_TIMING)
            timer.add("forward", t_fwd - t_h2d)

            optimizer.zero_grad()
            total_loss = 0.0
            stats.batch_dense_loss = stats.batch_sparse_loss = 0.0
            for p in range(patch_num):
                with torch.amp.autocast("cuda", enabled=False):
                    t = PatchTargets(
                        sample["gt_world_xy"], wld_paras, hw_random[:, :, p], cam_locations, losses.map_kernel, device
                    )
                    total_loss += losses.patch_loss(
                        model, predicted_patches[:, p, ...], predicted_masks[:, p, ...], t, stats
                    )
            loss = total_loss / patch_num
            t_loss = tick(cfg.PHASE_TIMING)
            timer.add("gt_and_loss", t_loss - t_fwd)  # CPU ground-truth generation + per-patch loss math

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update()
            t_step = tick(cfg.PHASE_TIMING)
            timer.add("backward_step", t_step - t_loss)

            stats.loss += loss.item()
            if stage == 2:
                stats.dense_loss += (stats.batch_dense_loss / patch_num).item()
                stats.sparse_loss += (stats.batch_sparse_loss / patch_num).item()
            stats.n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
            scene_iters += 1
            stats.iters += 1
            t_prev = time.perf_counter()
            timer.add("bookkeeping", t_prev - t_step)

        stats.scene_rows.append((scene_name, scene_iters, time.perf_counter() - scene_start))
        # Release cached CUDA blocks after every scene: the scenes' reference-point tensors differ in size, and the
        # fragmented cache grew past the card's memory (16.9 GB reserved for 0.45 GB allocated), which Windows then
        # pages to system RAM (1 -> 5-10 s/it).
        t_ec = time.perf_counter()
        torch.cuda.empty_cache()
        stats.empty_cache_s += time.perf_counter() - t_ec
    return stats


def run_full_scene_validation(model, stage, full_scene_val, epoch, device):
    t_fs = time.perf_counter()
    model.eval()
    fs_res = full_scene_validation(model, full_scene_val, "oracle" if stage == 2 else "learned", device)
    print(
        f"    [val-fullscene] {fs_res['n']} samples | MAE {fs_res['mae']:.1f} | bias {fs_res['bias']:+.1f} | "
        f"NAE {100 * fs_res['nae']:.0f}% | "
        + " ".join(f"{k} {v:.0f}" for k, v in sorted(fs_res["per_scene"].items()))
        + f" | {time.perf_counter() - t_fs:.0f}s"
    )
    append_csv_row(
        os.path.join(cfg.VAL_LOG_DIR, f"val_fullscene_{cfg.TIMING_RUN_TAG}.csv"),
        ["run_tag", "stage", "epoch", "mae", "bias", "nae", "n"],
        [
            cfg.TIMING_RUN_TAG,
            stage,
            epoch + 1,
            f"{fs_res['mae']:.4f}",
            f"{fs_res['bias']:.4f}",
            f"{fs_res['nae']:.4f}",
            fs_res["n"],
        ],
    )
    return fs_res


def train(
    stage=1,
    epochs=100,
    checkpoint_path=None,
    pretrained_backbone_path=None,
    lr_schedule="cosine",
    mass_normalize_dense=False,
    mass_normalize_sparse=False,
    k=cfg.NUM_CAMS_TO_SAMPLE,
    frames_per_scene=cfg.FRAMES_PER_SCENE_PER_EPOCH,
    val_frames_per_scene=cfg.VAL_FRAMES_PER_SCENE,
    resample_every_n_epochs=cfg.RESAMPLE_EVERY_N_EPOCHS,
    patience=None,
    train_registry=None,
    val_registry=None,
    val_seed=cfg.VAL_SEED_DEFAULT,
    start_epoch=0,
    best_metric_so_far=None,
    lr_override=None,
):
    """Train one stage; returns the path of its best checkpoint.

    start_epoch / best_metric_so_far resume a stopped stage approximately (weights only: the LR schedule is
    fast-forwarded, Adam and the patience counter start fresh, and a worse epoch cannot overwrite the saved best)."""
    if stage not in (1, "joint"):
        assert checkpoint_path is not None, (
            f"Stage {stage} requires a checkpoint from the previous stage to resume from!"
        )
        # A missing file used to skip the load silently and train from the default init under the stage's name.
        assert os.path.isfile(checkpoint_path), (
            f"--resume checkpoint does not exist: {checkpoint_path!r}. Stage {stage} would otherwise "
            f"silently start from ProposedModel's default init (ImageNet backbone only, everything else "
            f"randomly initialised) instead of resuming -- refusing rather than running a misleading "
            f"multi-hour job. If you really want a from-scratch Stage {stage} run, that is not supported "
            f"by this script's curriculum design; run Stage 1 first, or fix the path."
        )

    stage_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"--- [MULTI-SCENE] Starting Stage {stage} Training on "
        f"{len(train_registry)} train scenes / {len(val_registry)} val scenes [{device}] ---"
    )
    if stage == "joint":  # neck + experts at the Stage 2 rate; no early stop: the final weights are the result
        lr_override, patience = lr_override or cfg.LR_STAGE_2, patience or epochs
    lr = (
        lr_override
        if lr_override is not None
        else {1: cfg.LR_STAGE_1, 2: cfg.LR_STAGE_2, 3: cfg.LR_STAGE_3, 4: cfg.LR_STAGE_4}[stage]
    )
    patience_limit = (
        patience
        if patience is not None
        else {1: cfg.PATIENCE_STAGE_1, 2: cfg.PATIENCE_STAGE_2, 3: cfg.PATIENCE_STAGE_3, 4: cfg.PATIENCE_STAGE_4}[stage]
    )

    model = build_stage_model(stage, k, checkpoint_path, pretrained_backbone_path, device)
    losses = StageLosses(stage, device, mass_normalize_dense, mass_normalize_sparse)
    router_lr = cfg.LR_STAGE_3 if stage == "joint" else None  # --joint: the router in its own group
    optimizer, scheduler, scaler, backbone_trains = build_optimizer(model, lr, lr_schedule, epochs, router_lr)
    selector = Selector(best_metric_so_far, patience_limit)
    if start_epoch > 0:
        fast_forward_schedule(scheduler, start_epoch)
        print(
            f"[resume] Continuing Stage {stage} at epoch {start_epoch + 1}/{epochs} | LR fast-forwarded to "
            f"{optimizer.param_groups[0]['lr']:.2e} | best metric so far: "
            f"{'none (will be set by the first epoch)' if best_metric_so_far is None else selector.best} | "
            f"Adam state re-initialised, patience counter reset."
        )
    if checkpoint_path and os.path.isfile(checkpoint_path):
        load_resume_weights(model, checkpoint_path, device)
    prepare_probes(model, stage, lr)

    cached_val_samples, full_scene_val, val_cache_s = build_stage_val_sets(
        stage, val_registry, val_frames_per_scene, val_seed
    )
    val_scene_order = [e["scene_name"] for e in val_registry]
    t = {
        "stage_start": stage_start,
        "setup_s": time.perf_counter() - stage_start,
        "val_cache_s": val_cache_s,
        "train_s": 0.0,
        "val_s": 0.0,
        "iters": 0,
        "iter_s": 0.0,
        "epoch_times": [],
    }
    if cfg.TIMING_ENABLED:
        print(
            f"[timing] Stage {stage} setup (model build, checkpoint load, val cache): {format_duration(t['setup_s'])}"
        )
    scene_ref_shape = {e["scene_name"]: tuple(e["dataset"].active_refs.shape[1:3]) for e in train_registry}
    time_to_best_s = None
    ema = WeightEMA(model, cfg.EMA_DECAY) if cfg.EMA_DECAY > 0 else None
    if ema is not None:
        print(
            f"[ema] validating, selecting and saving an EMA of {sum(p.numel() for p in ema.params):,} trainable "
            f"parameters (decay {cfg.EMA_DECAY:g} per step)"
        )

    train_loaders = None
    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        set_train_modes(model, stage, backbone_trains)
        losses.label_generator.train()
        resample_s = 0.0
        if epoch % resample_every_n_epochs == 0:
            if train_loaders is not None:  # drop the previous loaders first (memory growth seen in a pilot run)
                train_loaders = None
                gc.collect()
            t_resample = time.perf_counter()
            train_loaders = [(e["scene_name"], sample_train_subset_loader(e, frames_per_scene)) for e in train_registry]
            resample_s = time.perf_counter() - t_resample

        train_stats = train_one_epoch(
            model, stage, epoch, epochs, train_loaders, optimizer, scaler, ema, losses, device
        )
        train_s = time.perf_counter() - epoch_start
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.swap_in()  # precise BN, validation, selection and the checkpoints below see the EMA weights
        t["train_s"] += train_s
        t["iters"] += train_stats.iters
        t["iter_s"] += train_s - resample_s - train_stats.empty_cache_s
        if stage == "joint":
            report_joint_train_terms(epoch, train_stats, optimizer)
        if (epoch + 1) % cfg.VAL_EVERY and epoch != epochs - 1:  # --val_every: a train-only epoch
            t["epoch_times"].append(time.perf_counter() - epoch_start)
            if (epoch + 1) % 5 == 0:
                save_checkpoint(model, ckpt_file("checkpoint", stage, epoch + 1))
            if ema is not None:
                ema.swap_out()
            continue

        if cfg.PRECISE_BN_FRAMES > 0:
            recalibrate_batchnorm(model, stage, train_loaders, cfg.PRECISE_BN_FRAMES, device)
        model.eval()
        losses.label_generator.eval()
        val_start = time.perf_counter()
        val = Validation(stage, losses).run(model, cached_val_samples, epoch, epochs, len(val_registry))
        val_s = time.perf_counter() - val_start
        epoch_s = time.perf_counter() - epoch_start  # excludes the checkpoint I/O below

        fs_res = (
            run_full_scene_validation(model, stage, full_scene_val, epoch, device)
            if full_scene_val is not None
            else None
        )
        metric, metric_name, is_pct = val.selection_metric(fs_res)
        metric, metric_name = selector.smooth(metric, metric_name)
        val.report(epoch, train_stats, metric, metric_name, is_pct, len(val_registry))
        scene_rows = summarize_scene_stats(val.scene_stats, val_scene_order)
        report_scene_breakdown(
            stage, epoch + 1, scene_rows, full_table=(selector.is_best(metric) or epoch == epochs - 1)
        )
        write_scene_rows_csv(stage, epoch + 1, scene_rows)

        t["epoch_times"].append(epoch_s)
        t["val_s"] += val_s
        if cfg.TIMING_ENABLED:
            print_epoch_timing(
                stage,
                epoch,
                epochs,
                {
                    "train_s": train_s,
                    "val_s": val_s,
                    "epoch_s": epoch_s,
                    "iters": train_stats.iters,
                    "resample_s": resample_s,
                    "empty_cache_s": train_stats.empty_cache_s,
                    "stage_elapsed_s": time.perf_counter() - stage_start,
                    "epoch_times": t["epoch_times"],
                },
                train_stats.timer,
                train_stats.scene_rows,
                len(cached_val_samples),
                len(train_registry),
                scene_ref_shape,
            )

        result = selector.update(metric, epoch + 1)
        if result == "best":
            time_to_best_s = time.perf_counter() - stage_start
            save_checkpoint(model, ckpt_file("best", stage))
            shown = f"{selector.best * 100:.1f}%" if is_pct else f"{selector.best:.4f}"
            print(f"*** New best {metric_name} ({shown}). Saved checkpoint. ***")
        elif result == "stop":
            print(
                f"!!! {metric_name} failed to improve for {patience_limit} consecutive epochs. Triggering Early Stopping !!!"
            )
            break
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, ckpt_file("checkpoint", stage, epoch + 1))
        if ema is not None:
            ema.swap_out()  # back to the raw weights for the next epoch's training

    if ema is not None:
        ema.swap_in()  # the final checkpoint holds the EMA weights too (no-op after an early stop)
    # the final-epoch weights too: the best-epoch file is a minimum over noisy epochs
    final_path = ckpt_file("final", stage)
    save_checkpoint(model, final_path)
    print(f"[checkpoint] Final-epoch weights saved to {final_path}")
    record_stage_timing(stage, epochs, start_epoch, t, selector.best_epoch, time_to_best_s, selector.best)
    return ckpt_file("best", stage)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.log_file != "":
        log_path = args.log_file or f"multiscene_training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        log_path = (
            log_path if os.path.isabs(log_path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), log_path)
        )
        sys.stdout = Tee(sys.__stdout__, open(log_path, "a", buffering=1))
        print(f"Logging all output to: {log_path}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cfg.apply_args(args, parser)

    def weather(s):
        return [w.strip().upper() for w in s.split(",")] if s else None

    train_registry, val_registry = build_registries(
        args.k, not args.skip_camera_validation, weather(args.train_weather), weather(args.val_weather)
    )
    common = dict(
        k=args.k,
        frames_per_scene=args.frames_per_scene,
        val_frames_per_scene=args.val_frames_per_scene,
        resample_every_n_epochs=args.resample_every_n_epochs,
        train_registry=train_registry,
        val_registry=val_registry,
        val_seed=(args.val_seed if args.val_seed >= 0 else None),
        lr_schedule=args.lr_schedule,
        mass_normalize_dense=args.mass_normalize_dense,
        mass_normalize_sparse=args.mass_normalize_sparse,
    )
    patience = {1: args.patience1, 2: args.patience2, 3: args.patience3, 4: args.patience4}

    if args.full_curriculum:
        epochs = {1: args.epochs1, 2: args.epochs2, 3: args.epochs3, 4: args.epochs4}
        print("=" * 70)
        print("FULL MULTI-SCENE CURRICULUM RUN: Stages 1 -> 2 -> 3 -> 4")
        print(f"Epochs per stage: {args.epochs1} / {args.epochs2} / {args.epochs3} / {args.epochs4}")
        print(f"Patience per stage: {args.patience1} / {args.patience2} / {args.patience3} / {args.patience4}")
        print("=" * 70)
        checkpoint_path = None
        for stage in [1, 2, 3, 4]:
            print(f"\n{'#' * 70}\n# STAGE {stage} START\n{'#' * 70}")
            checkpoint_path = train(
                stage=stage,
                epochs=epochs[stage],
                checkpoint_path=checkpoint_path,
                pretrained_backbone_path=args.backbone if stage == 1 else None,
                patience=patience[stage],
                **common,
            )
            print(f"# STAGE {stage} COMPLETE -- best checkpoint: {checkpoint_path}")
        print("\n" + "=" * 70)
        print("FULL CURRICULUM COMPLETE.")
        print(f"Final checkpoint (Stage 4): {checkpoint_path}")
        print("=" * 70)
    else:  # a single stage, or --joint
        train(
            stage="joint" if args.joint else args.stage,
            epochs=args.epochs,
            checkpoint_path=args.resume,
            start_epoch=args.start_epoch,
            best_metric_so_far=args.best_metric_so_far,
            lr_override=args.lr,
            pretrained_backbone_path=args.backbone,
            patience=patience.get(args.stage),
            **common,
        )
    print_and_save_timing_report()


if __name__ == "__main__":
    main()
