"""Optimiser, EMA, BatchNorm handling and checkpoint bookkeeping of one training stage."""

import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import config as cfg
from .model import AblationNeck
from .targets import oracle_mask_config


class WeightEMA:
    """Exponential moving average of the trainable weights, as in neck_ablation.py (H0's recipe): updated after
    every optimiser step, decay warm-up min(decay, (1 + n) / (10 + n)). Instead of a model copy it keeps fp32 shadows and
    swaps them into the model for validation, selection and saving (swap_in) and the raw weights back for training
    (swap_out). BatchNorm buffers are not averaged: the model's own are used, and --precise_bn_frames re-estimates them
    on the swapped-in EMA weights before every validation."""

    def __init__(self, model, decay):
        self.decay, self.n, self.active = decay, 0, False
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [p.detach().float().clone() for p in self.params]
        self.raw = None

    @torch.no_grad()
    def update(self):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, q in zip(self.shadow, self.params):
            e.lerp_(q.detach().float(), 1.0 - d)

    @torch.no_grad()
    def swap_in(self):
        if not self.active:
            self.raw = [p.detach().clone() for p in self.params]
            for p, e in zip(self.params, self.shadow):
                p.copy_(e)
            self.active = True

    @torch.no_grad()
    def swap_out(self):
        if self.active:
            for p, r in zip(self.params, self.raw):
                p.copy_(r)
            self.raw, self.active = None, False


def build_optimizer(model, lr, lr_schedule, epochs, router_lr=None):
    """Adam over the trainable parameters (backbone at lr * BACKBONE_LR_MULT; with router_lr, the router in its own
    group at that rate), cosine schedule unless 'constant' (every group from its own rate to 1e-6), AMP grad
    scaler. Returns (optimizer, scheduler or None, scaler, backbone trains)."""
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    router_params = [p for p in model.head.gating_net.parameters() if p.requires_grad] if router_lr is not None else []
    own_group = {id(p) for p in backbone_params + router_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in own_group]
    param_groups = [{"params": other_params, "lr": lr}]
    if router_params:
        param_groups.append({"params": router_params, "lr": router_lr})
        print(
            f"[optimizer] router: {sum(p.numel() for p in router_params):,} parameters in its own group (lr {router_lr:g})"
        )
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * cfg.BACKBONE_LR_MULT})
        print(
            f"[backbone] {sum(p.numel() for p in backbone_params):,} trainable backbone parameters "
            f"(lr {lr * cfg.BACKBONE_LR_MULT:g}) + {sum(p.numel() for p in other_params):,} others (lr {lr:g})"
        )
    optimizer = optim.Adam(param_groups, lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = None if lr_schedule == "constant" else CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return optimizer, scheduler, torch.amp.GradScaler("cuda"), bool(backbone_params)


def fast_forward_schedule(scheduler, start_epoch):
    """--start_epoch resume: step the LR schedule past the epochs already done. Approximate continuation: only the
    weights are in the checkpoint, so Adam moments, the grad scaler and the patience counter start fresh."""
    if scheduler is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "scheduler.step() before optimizer.step()" -- intended here
            for _ in range(start_epoch):
                scheduler.step()


def set_train_modes(model, stage, backbone_trains):
    """train()/eval() per submodule for this stage. model.train() alone would let the BatchNorm statistics of frozen
    parts drift, so every part that does not learn in this stage is put back into eval mode."""
    model.train()
    model.backbone.eval()
    if backbone_trains:
        # as in the pre-stride-8 regime, only layer4's BN uses batch statistics (a batch is 5 views of one frame)
        model.backbone.feature_extractor[-1].train()
    sparse_probe = cfg.SPARSE_ONLY and stage == 2
    dense_probe = cfg.DENSE_ONLY and stage == 2
    router_probe = cfg.ROUTER_ONLY and stage == 4
    if stage in (1, 2, 4, "joint") and not (sparse_probe or dense_probe or router_probe):
        model.neck.train()
    else:
        model.neck.eval()
    if dense_probe and isinstance(model.neck, AblationNeck) and not cfg.FREEZE_ADAPTER:
        model.neck.head_processor.train()  # the adapter trains with the dense expert (fusion stays in eval)
    if cfg.FREEZE_FUSION and stage == 2 and isinstance(model.neck, AblationNeck):
        model.neck.fusion.eval()
    if stage in (2, 4, "joint"):
        model.head.dense_expert.train(not (sparse_probe or router_probe))
        model.head.sparse_expert.train(not (dense_probe or router_probe))
    else:
        model.head.dense_expert.eval()
        model.head.sparse_expert.eval()
    model.head.gating_net.train(stage in (3, 4, "joint"))
    model.throwaway_head.train(stage == 1)


def recalibrate_batchnorm(model, stage, train_loaders, frames_per_scene, device):
    """Precise BN: with one patch per batch, the running statistics follow the last scene or two of the epoch
    (bn_recalibration_test.py: epoch-to-epoch MAE spread 142 -> 58 after re-estimating). Re-estimate them as a
    cumulative average over `frames_per_scene` frames of every train scene, for the BN layers training this stage."""
    bn_mods = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training]
    if not bn_mods:
        return
    old_momentum = [m.momentum for m in bn_mods]
    for m in bn_mods:
        m.reset_running_stats()
        m.momentum = None
    with torch.no_grad():
        for _scene_name, loader in train_loaders:
            it = iter(loader)
            for _ in range(frames_per_scene):
                try:
                    smp = next(it)
                except StopIteration:
                    break
                with torch.amp.autocast("cuda"):
                    model(
                        smp["images"].to(device),
                        smp["ref_points_full"].to(device),
                        smp["hw_random"].to(device),
                        stage=stage,
                        camera_paras=smp["camera_paras"].to(device),
                        wld_map_paras=smp["wld_map_paras"].to(device),
                    )
    for m, mom in zip(bn_mods, old_momentum):
        m.momentum = mom


def ckpt_file(kind, stage, epoch=None):
    """multiscene_{best|final}_stage_N<tag>.pth, or multiscene_checkpoint_stage_N<tag>_epoch_E.pth."""
    suffix = f"_epoch_{epoch}" if epoch is not None else ""
    return os.path.join(cfg.CKPT_DIR, f"multiscene_{kind}_stage_{stage}{cfg.CKPT_TAG}{suffix}.pth")


def save_checkpoint(model, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "neck_config": model.neck_config,
            "oracle_mask_config": oracle_mask_config(),
            "sparse_render": model.head.sparse_render,
        },
        path,
    )


class Selector:
    """Checkpoint selection and early stopping on the validation metric (lower is better), optionally on the mean
    of the last SELECT_WINDOW epochs (SEED_METRIC_HISTORY continues that mean across a --start_epoch resume)."""

    def __init__(self, best_metric_so_far, patience):
        self.best = float(best_metric_so_far) if best_metric_so_far is not None else float("inf")
        self.patience = patience
        self.epochs_without_improvement = 0
        self.history = list(cfg.SEED_METRIC_HISTORY)
        self.best_epoch = None

    def smooth(self, metric, name):
        """Record this epoch's raw metric; return the one to select on (window mean) and its name."""
        self.history.append(metric)
        if cfg.SELECT_WINDOW > 1:
            metric = float(np.mean(self.history[-cfg.SELECT_WINDOW :]))
            name = f"{name}, mean of last {min(cfg.SELECT_WINDOW, len(self.history))} epochs"
        return metric, name

    def is_best(self, metric):
        return metric < self.best

    def update(self, metric, epoch):
        """Returns 'best', 'worse' or 'stop'."""
        if metric < self.best:
            self.best, self.epochs_without_improvement, self.best_epoch = metric, 0, epoch
            return "best"
        self.epochs_without_improvement += 1
        print(f"--- No improvement for {self.epochs_without_improvement}/{self.patience} epochs ---")
        return "stop" if self.epochs_without_improvement >= self.patience else "worse"
