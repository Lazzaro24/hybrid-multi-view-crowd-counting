"""The proposed model (backbone + neck + hybrid routing head), its neck variants and checkpoint helpers."""

import argparse
import os
import sys

import torch
import torch.nn as nn

from models.backbone.feature_extractor import DilatedResNet18
from models.neck.proposed.spatial_cross_attention import SpatialCrossAttentionNeck
from models.neck.proposed.bev_encoder import BEVFormerEncoderNeck
from models.head.proposed.routing import HybridRoutingHead
from losses.gaussian_mse import get_map_kernel

from . import config as cfg


def checkpoint_uses_maxpool(state_dict):
    """Stride-8 (MaxPool kept) vs. legacy stride-4 backbone layout. Without MaxPool, layer1 sits at
    feature_extractor index 3, so its keys only exist in a legacy checkpoint."""
    return "backbone.feature_extractor.3.0.conv1.weight" not in state_dict


def checkpoint_neck_config(checkpoint):
    """Neck config stored with a checkpoint; checkpoints saved before the encoder neck existed hold the legacy neck."""
    return dict(checkpoint.get("neck_config") or {"type": "legacy"})


def checkpoint_sparse_arch(state_dict):
    """'deep' if the saved sparse expert is build_sparse_expert('deep') (5 convs: indices 0..8), else 'shallow'."""
    return "deep" if any(k.startswith("head.sparse_expert.8.") for k in state_dict) else "shallow"


def build_sparse_expert(arch, in_channels=64, hidden_channels=64):
    """'deep': four 3x3 conv+ReLU layers with dilations (1, 2, 2, 1) + a 3x3 output conv (linear, like the original).
    Receptive field 15 px (3 m at 5 px/m) vs 5 px for the original shallow expert; ~148 k parameters. Same
    interface: [B, in_channels, H, W] -> [B, 1, H, W] raw output."""
    if arch != "deep":
        raise ValueError(f"unknown sparse architecture {arch!r}")
    layers, c = [], in_channels
    for d in (1, 2, 2, 1):
        layers += [nn.Conv2d(c, hidden_channels, kernel_size=3, padding=d, dilation=d), nn.ReLU(inplace=True)]
        c = hidden_channels
    layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1))
    return nn.Sequential(*layers)


class AblationNeck(nn.Module):
    """The fusion of a neck-ablation variant (training_scripts/neck_ablation.py: 'ca_sa', 'homography_256', ...) as
    the proposed model's neck, so the proposed head can be trained on a fusion already trained in the ablation
    ("Stage 1 done", training_scripts/make_stage1_from_ablation.py). A fresh head_processor (the legacy neck's
    2 x 3x3 conv, 256 -> 64) maps the fusion's 256 channels to the head's 64 and trains in Stage 2. The homography
    also needs the cameras, the world grid and the tile's offset (needs_ctx); attention fusions only use the tile's
    reference points."""

    needs_ctx = True

    def __init__(self, variant, num_cam, num_z_anchors=4, out_channels=64, adapter=True):
        super().__init__()
        sys.path.append(os.path.join(cfg.src_dir, "training_scripts"))
        import neck_ablation as NA

        assert num_cam == NA.NUM_CAMS, f"ablation fusions are built for {NA.NUM_CAMS} cameras, got {num_cam}"
        self.variant = variant
        self.fusion = NA.build_fusion(
            variant, argparse.Namespace(z_anchors=num_z_anchors, bev_layers=3, no_grad_checkpoint=True)
        )
        self.uses_geometry = isinstance(self.fusion, NA.HomographyFusion)
        c = self.fusion.out_channels
        if adapter:
            self.head_processor = nn.Sequential(
                nn.Conv2d(c, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            )
            self.out_channels = out_channels
        else:  # --neck_adapter none: the head reads the fusion's own channels (256 for homography_256)
            self.head_processor = nn.Identity()
            self.out_channels = c

    def describe(self):
        return f"ablation fusion '{self.variant}' ({self.fusion.out_channels} ch)" + (
            f" + head_processor -> {self.out_channels} ch"
            if self.out_channels != self.fusion.out_channels
            else " (no adapter)"
        )

    def forward(self, features_2d, reference_points, ctx=None, hw=(0, 0)):
        if self.uses_geometry:
            bev = self.fusion(features_2d, ctx, int(hw[0]), int(hw[1]))
        else:  # LegacyAttentionFusion slices [h0:h0+200, w0:w0+200] of the refs it gets: pass the tile's own
            bev = self.fusion(features_2d, {"ref_points_full": reference_points}, 0, 0)
        return self.head_processor(bev)


def neck_ctx(neck, camera_paras, wld_map_paras, h0, w0):
    """Extra arguments for necks that need the cameras / world grid (AblationNeck), {} for all others."""
    if not getattr(neck, "needs_ctx", False):
        return {}
    assert camera_paras is not None and wld_map_paras is not None, "this neck needs camera_paras and wld_map_paras"
    return {"ctx": {"camera_paras": camera_paras, "wld_map_paras": wld_map_paras}, "hw": (h0, w0)}


def build_neck(neck_config, num_cam):
    params = dict(neck_config)
    neck_type = params.pop("type")
    if neck_type == "ablation":
        return AblationNeck(
            params["variant"], num_cam, num_z_anchors=params.get("z_anchors", 4), adapter=params.get("adapter", True)
        )
    if neck_type == "legacy":
        return SpatialCrossAttentionNeck(
            in_channels=512, out_channels=64, full_bev_shape=cfg.BEV_SHAPE, num_cameras=num_cam, num_z_anchors=4
        )
    if neck_type == "bevformer":
        neck = BEVFormerEncoderNeck(
            in_channels=512,
            out_channels=64,
            full_bev_shape=cfg.BEV_SHAPE,
            num_cameras=num_cam,
            num_z_anchors=4,
            **params,
        )
        neck.grad_checkpoint = cfg.NECK_GRAD_CHECKPOINT
        return neck
    raise ValueError(f"unknown neck type in {neck_config!r}")


def apply_sparse_render(model, config):
    """Set how the sparse expert enters the fusion (None = relu(sparse), or rendered NMS peaks)."""
    kernel = get_map_kernel(sigma=cfg.GAUSSIAN_SIGMA, kernel_size=cfg.GAUSSIAN_KERNEL_SIZE) if config else None
    model.head.set_sparse_render(config, kernel)


def build_model_for_checkpoint(checkpoint, num_cam=cfg.NUM_CAMS_TO_SAMPLE):
    """ProposedModel with the backbone layout, neck, sparse architecture and sparse rendering of a saved checkpoint
    (weights not loaded)."""
    state = checkpoint.get("model_state_dict", checkpoint)
    previous, cfg.SPARSE_ARCH = cfg.SPARSE_ARCH, checkpoint_sparse_arch(state)
    try:
        model = ProposedModel(
            num_cam=num_cam, use_maxpool=checkpoint_uses_maxpool(state), neck_config=checkpoint_neck_config(checkpoint)
        )
    finally:
        cfg.SPARSE_ARCH = previous
    apply_sparse_render(model, checkpoint.get("sparse_render"))
    return model


class ProposedModel(nn.Module):
    def __init__(
        self, num_cam=cfg.NUM_CAMS_TO_SAMPLE, pretrained_backbone_path=None, use_maxpool=None, neck_config=None
    ):
        super().__init__()
        self.neck_config = dict(neck_config or cfg.NECK_CONFIG)
        self.backbone = DilatedResNet18(
            pretrained=True, use_maxpool=cfg.BACKBONE_USE_MAXPOOL if use_maxpool is None else use_maxpool
        )

        if pretrained_backbone_path:
            # The backbone is frozen in every stage, so a missing file would silently train on
            # frozen ImageNet features -- fail instead.
            if not os.path.isfile(pretrained_backbone_path):
                raise FileNotFoundError(f"2D-pretrained backbone not found: {pretrained_backbone_path!r}")
            checkpoint = torch.load(pretrained_backbone_path, map_location="cpu", weights_only=False)
            backbone_weights = {
                k.replace("backbone.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
                if k.startswith("backbone.")
            }
            self.backbone.load_state_dict(backbone_weights)

        self.neck = build_neck(self.neck_config, num_cam)
        # the head reads the neck's output width (64, or the fusion's own with --neck_adapter none); the dense
        # expert's architecture is stored with the neck config ('dense_arch', --dense_arch)
        head_in = getattr(self.neck, "out_channels", 64)
        self.head = HybridRoutingHead(
            in_channels=head_in,
            hidden_channels=64,
            router_contrast=cfg.ROUTER_CONTRAST,
            dense_arch=self.neck_config.get("dense_arch", "legacy"),
        )
        if cfg.SPARSE_ARCH != "shallow":  # swap the sparse expert (same interface); default keeps the original
            self.head.sparse_expert = build_sparse_expert(cfg.SPARSE_ARCH, in_channels=head_in, hidden_channels=64)
        self.throwaway_head = nn.Sequential(
            nn.Conv2d(head_in, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def set_stage_trainability(self, stage):
        # The whole backbone (incl. layer4) stays frozen in every stage, identical to the
        # baseline, so the fusion mechanism is the only variable in the comparison.
        for param in self.parameters():
            param.requires_grad = False

        if stage == 1:
            for param in self.neck.parameters():
                param.requires_grad = True
            for param in self.throwaway_head.parameters():
                param.requires_grad = True
        elif stage == 2:
            for param in self.neck.parameters():
                param.requires_grad = True
            for param in self.head.dense_expert.parameters():
                param.requires_grad = True
            for param in self.head.sparse_expert.parameters():
                param.requires_grad = True
        elif stage == 3:
            for param in self.head.gating_net.parameters():
                param.requires_grad = True
        elif stage in (4, "joint"):
            for param in self.neck.parameters():
                param.requires_grad = True
            for param in self.head.dense_expert.parameters():
                param.requires_grad = True
            for param in self.head.sparse_expert.parameters():
                param.requires_grad = True
            for param in self.head.gating_net.parameters():
                param.requires_grad = True

        if cfg.SPARSE_ONLY and stage == 2:  # probe: only the sparse expert learns
            for param in self.parameters():
                param.requires_grad = False
            for param in self.head.sparse_expert.parameters():
                param.requires_grad = True
        if cfg.DENSE_ONLY and stage == 2:  # probe: only the dense expert learns
            for param in self.parameters():
                param.requires_grad = False
            for param in self.head.dense_expert.parameters():
                param.requires_grad = True
            # an AblationNeck's head_processor (256 -> 64 adapter) is fresh after make_stage1_from_ablation.py:
            # it trains with the dense expert; the fusion itself stays frozen
            if isinstance(self.neck, AblationNeck) and not cfg.FREEZE_ADAPTER:
                for param in self.neck.head_processor.parameters():
                    param.requires_grad = True
        if cfg.FREEZE_FUSION and stage == 2 and isinstance(self.neck, AblationNeck):
            # joint Stage 2 on the ablation's own (frozen) fusion: the sparse loss cannot reshape the features
            for param in self.neck.fusion.parameters():
                param.requires_grad = False
        if cfg.ROUTER_ONLY and stage == 4:  # probe: only the router learns, against the fused-map MSE
            for param in self.parameters():
                param.requires_grad = False
            for param in self.head.gating_net.parameters():
                param.requires_grad = True

        # --train_backbone: 'layer4' restores the pre-stride-8 regime (layer4 trained in Stages 1/2/4),
        # 'all' trains the whole backbone with the neck, as the SynMVCrowd paper's baseline does.
        probe = (cfg.SPARSE_ONLY or cfg.DENSE_ONLY) and stage == 2 or cfg.ROUTER_ONLY and stage == 4
        if cfg.TRAIN_BACKBONE != "none" and stage in (1, 2, 4, "joint") and not probe:
            part = self.backbone.feature_extractor[-1] if cfg.TRAIN_BACKBONE == "layer4" else self.backbone
            for param in part.parameters():
                param.requires_grad = True

    def forward(self, images, ref_points_full, hw_random=None, stage=4, camera_paras=None, wld_map_paras=None):
        B, N, C, H, W = images.shape
        images_flat = images.view(B * N, C, H, W)

        features_flat = self.backbone(images_flat)
        _, C_f, H_f, W_f = features_flat.shape
        features = features_flat.view(B, N, C_f, H_f, W_f)

        if hw_random is not None:
            patch_num = hw_random.shape[-1]
            out_patches, out_masks = [], []

            for p in range(patch_num):
                hw_offset = hw_random[:, :, p]
                h_start, w_start = hw_offset[0, 0].item(), hw_offset[0, 1].item()
                patch_ref_points = ref_points_full[
                    :, :, h_start : h_start + cfg.BEV_SHAPE[0], w_start : w_start + cfg.BEV_SHAPE[1], :, :
                ]

                proj_bev = self.neck(
                    features, patch_ref_points, **neck_ctx(self.neck, camera_paras, wld_map_paras, h_start, w_start)
                )

                if stage == 1:
                    decoded_bev = torch.relu(self.throwaway_head(proj_bev))
                    mask_bev = torch.zeros_like(decoded_bev)
                elif stage == 2:
                    dense_bev = torch.relu(self.head.dense_expert(proj_bev))
                    sparse_raw = self.head.sparse_expert(proj_bev)
                    sparse_bev = (
                        torch.sigmoid(sparse_raw.float())
                        if cfg.SPARSE_ACTIVATION == "sigmoid"
                        else torch.relu(sparse_raw)
                    )
                    decoded_bev = torch.cat([dense_bev, sparse_bev], dim=1)
                    mask_bev = torch.zeros_like(dense_bev)
                elif stage == 3:
                    decoded_bev = proj_bev
                    mask_bev = self.head.gating_net(proj_bev)
                elif stage == 4:
                    decoded_bev, mask_bev = self.head(proj_bev)
                elif stage == "joint":
                    # head.forward's fusion, keeping the experts' maps for their own losses: [fused, dense, sparse]
                    mask_bev = self.head.gating_net(proj_bev)
                    dense_bev = torch.relu(self.head.dense_expert(proj_bev))
                    sparse_bev = torch.relu(self.head.sparse_expert(proj_bev))
                    sparse_in = sparse_bev if self.head.sparse_render is None else self.head.sparse_map(proj_bev)
                    fused_bev = mask_bev * dense_bev + (1.0 - mask_bev) * sparse_in
                    decoded_bev = torch.cat([fused_bev, dense_bev, sparse_bev], dim=1)

                out_patches.append(decoded_bev)
                out_masks.append(mask_bev)

            return torch.stack(out_patches, dim=1), torch.stack(out_masks, dim=1)
        return None, None


def _drop_changed(state, model_sd, prefix):
    """True if the saved tensors under `prefix` do not fit the model (shape differs, or a key missing on one side)."""
    mismatch = any(
        k.startswith(prefix) and (k not in model_sd or model_sd[k].shape != v.shape) for k, v in state.items()
    )
    absent = any(k.startswith(prefix) and k not in state for k in model_sd)
    return mismatch or absent


def load_resume_weights(model, checkpoint_path, device):
    """Load a checkpoint into the model, starting the parts whose architecture changed from a fresh init instead
    (load_state_dict(strict=False) tolerates missing keys, not shape mismatches): the sparse expert (other
    --sparse_arch, or --reinit_sparse), the gating net (frozen until Stage 3, so its saved weights are only init) and
    any head tensor changed by --dense_arch / --neck_adapter."""
    state = torch.load(checkpoint_path, map_location=device)["model_state_dict"]
    model_sd = model.state_dict()
    if cfg.REINIT_SPARSE or _drop_changed(state, model_sd, "head.sparse_expert."):
        n_drop = sum(1 for k in state if k.startswith("head.sparse_expert."))
        state = {k: v for k, v in state.items() if not k.startswith("head.sparse_expert.")}
        print(
            f"[sparse] Dropped {n_drop} saved sparse_expert tensors "
            f"({'--reinit_sparse' if cfg.REINIT_SPARSE else 'architecture differs from checkpoint'}); "
            f"sparse expert ({cfg.SPARSE_ARCH}) starts from a fresh initialisation."
        )
    if _drop_changed(state, model_sd, "head.gating_net."):
        n_drop = sum(1 for k in state if k.startswith("head.gating_net."))
        state = {k: v for k, v in state.items() if not k.startswith("head.gating_net.")}
        print(
            f"[gating] Dropped {n_drop} saved gating_net tensors (architecture differs from "
            f"checkpoint); gating_net starts from a fresh initialisation."
        )
    mismatched = [
        k for k, v in state.items() if k.startswith("head.") and k in model_sd and model_sd[k].shape != v.shape
    ]
    dense_saved = [k for k in state if k.startswith("head.dense_expert.")]
    if any(k not in model_sd or k in mismatched for k in dense_saved):
        mismatched = sorted(set(mismatched) | set(dense_saved))
    if mismatched:
        state = {k: v for k, v in state.items() if k not in mismatched}
        print(
            f"[head] Dropped {len(mismatched)} saved head tensors whose architecture changed "
            f"({sorted({'.'.join(k.split('.')[:2]) for k in mismatched})}); they start from a fresh initialisation."
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  missing keys: {missing}")
    print(f"  unexpected keys: {unexpected}")
