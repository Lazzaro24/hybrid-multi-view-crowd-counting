"""
neck_ablation.py

Controlled comparison of BEV fusion modules ("necks"): everything except the fusion step is identical.

  shared  : frozen 2D-pretrained stride-8 backbone, the baseline's counting head (BaselineDecoder: 3x3 conv
            feat_before_merge + map_classifier + abs), GaussianMSE_norm loss (scale 100) on the dataset's
            density_patches, optimizer / schedule / epochs / seed, training frames, crops and camera order
            (seeded per epoch), validation and test samples, whole-scene evaluation on 200x200 tiles.
  varies  : the fusion module that turns the 5 camera feature maps into one BEV feature map:

    homography  SynMVCrowd baseline: ViewFeatureSelector + SpatialProjector (ground-plane homography) +
                max over cameras                                               -> 512 ch
    homography_256  the same, but the camera features are first compressed 512 -> 256 by the attention necks'
                cam_compression (1x1 conv + BN + ReLU), so its head is identical to the attention variants'
                (same width, same initial weights). homography vs homography_256 isolates width / head size;
                homography_256 vs the attention variants isolates the fusion method  -> 256 ch
    ca          legacy neck WITHOUT self-attention: learned queries -> deformable cross-attention   -> 256 ch
    sa_ca       legacy neck as before the fix: self-attention on the learned queries -> cross-attention
                (the self-attention output is the same for every image)       -> 256 ch
    ca_sa       legacy modules, one layer, self-attention AFTER cross-attention (sees image features) -> 256 ch
    bev3        BEVFormer-style encoder: 3 x [deformable SA -> LN -> SCA -> LN -> FFN -> LN], 8 heads -> 256 ch
    hca         forward-backward hybrid (FB-BEV, Li et al. ICCV 2023): the homography_256 BEV is the query of one
                BEVFormer spatial cross-attention, whose output is added back as a residual, so the sampling offsets
                and camera / height weights are predicted from this image's evidence. The cross-attention output
                projection starts at zero: at step 0 the variant computes exactly homography_256  -> 256 ch
    hca_mv      as hca, but the query is Linear([max, mean, var] over the cameras that see the cell), so the
                cross-attention also sees how much the views disagree                          -> 256 ch
    hca_static  control for hca: offsets and anchor weights are predicted from the position encoding only, so
                where and how much the cross-attention samples is the same for every image (the sampled camera
                features, residual and homography path are unchanged)                         -> 256 ch
    mvha        multi-view height attention (models/neck/proposed/mv_height_attention.py): every BEV cell weighs all
                5 cameras x Z height anchors with ONE softmax whose logits are key-based (query . sampled feature),
                plus a cross-view consistency term (cosine of a camera's sampled key with the other cameras' at the
                same 3D point) and a geometry term; no learned offsets. Unlike the deformable cross-attentions
                (weights = Linear(query), cameras summed or averaged), the weights depend on what each camera shows
                and cameras compete per 3D point                                              -> 256 ch
    mvha_res    homography_256 + mvha as a zero-initialised residual (as hca); the homography BEV is also added
                to mvha's query. At step 0 it computes exactly homography_256                 -> 256 ch

  The attention necks' own head_processor (2 extra 3x3 convs) is removed so that all conv decoding happens in the
  shared head. Only the head's first conv differs in input width (512 for homography, 256 for all others).

Deviation from the SynMVCrowd paper protocol (applies to every variant): each scene uses its fixed K-means 5-camera
set (optimal_cameras_filtered.json, as in the proposed training) instead of 5 random cameras per sample -- the
attention necks need reference points for the drawn cameras, and those for all cameras do not fit in RAM (~20 GB).

Evaluation / selection (identical for every variant):
  - validation, checkpoint selection and testing use an exponential moving average (EMA) of the trainable weights
    (--ema_decay, per step), which damps the epoch-to-epoch swings of the count scale; --no_ema uses the raw weights
  - validation: fixed val frames (--val_frames_per_scene, default 20 -> 200 whole-scene samples), identical in every
    epoch (loaders are rebuilt from their seed, precise-BN uses the same train frames every time)
  - test (whole test split sample, --test_frames_per_scene) on two weight sets: the best-val epoch ("best") and the
    final epoch ("final"); a large gap between the two means one run is not enough to rank variants
  Metrics: MAE / RMSE / NAE / bias of the count (sum of the map / 100), per-scene MAE.

Height anchors (attention variants): --refs_file / --z_anchors select the reference-point cache. The default
master_refs.pt was built assuming d_mean is the ground, but d_mean is the mean HEAD height (annotations are heads,
d_delta = 1.75 m), so its 4 anchors sit at feet, head, +1.8 m and +3.55 m above the heads. master_refs_k8.pt
(--z_anchors 8) spans d_mean - 1.75 - 2.5 .. d_mean + 2.5 m: feet to head with 2.5 m terrain margins both ways.
Results from different caches are not comparable for the attention variants (the homography ones ignore the refs).

Usage (from src):
    uv run python training_scripts/neck_ablation.py --run_name ablation_v1
    uv run python training_scripts/neck_ablation.py --run_name ablation_v1 --variants ca_sa,bev3 --epochs 30
    uv run python training_scripts/neck_ablation.py --run_name ablation_v1 --summarize      # table only
    uv run python training_scripts/neck_ablation.py --run_name ablation_k8 --variants hca --refs_file master_refs_k8.pt --z_anchors 8
    uv run python training_scripts/neck_ablation.py --run_name ablation_mv --variants mvha,mvha_res --refs_file master_refs_k8.pt --z_anchors 8
    # OT fine-tune on top of a finished variant, as the baseline's "Baseline (OT)" (train_baseline.py --loss ot):
    uv run python training_scripts/neck_ablation.py --run_name ablation_ot --variants ca_sa --loss ot --lr 1e-5 --lr_schedule constant --init_from runs/neck_ablation/ablation_v2/ca_sa/last.pth --val_every 5
Each variant writes runs/neck_ablation/<run_name>/<variant>/{best.pth,last.pth,log.csv,result.json}; finished
variants (result.json present) are skipped and an interrupted variant resumes from its last.pth (unless
--overwrite), so an interrupted run can simply be restarted with the same command.
"""

import os
import sys
import csv
import copy
import json
import math
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
from models.backbone.feature_extractor import DilatedResNet18
from models.neck.baseline.feature_selection import ViewFeatureSelector
from models.neck.baseline.projector import SpatialProjector
from models.neck.proposed.spatial_cross_attention import SpatialCrossAttentionNeck, DeformableSelfAttention
from models.neck.proposed.bev_encoder import (
    BEVFormerEncoderNeck,
    LearnedBEVPositionalEncoding,
    MultiHeadSpatialCrossAttention,
)
from models.neck.proposed.mv_height_attention import MultiViewHeightAttention
from models.head.baseline.decoder import BaselineDecoder
from losses.gaussian_mse import GaussianMSE_norm, get_map_kernel
from losses.ot_loss.genloss_synmvcrowd import SynMVCrowdGeneralizedLoss, patch_points
from utils.exclusions import load_excluded_scenes

VARIANTS = [
    "homography",
    "homography_256",
    "ca",
    "sa_ca",
    "ca_sa",
    "bev3",
    "hca",
    "hca_mv",
    "hca_static",
    "mvha",
    "mvha_res",
]
SCALE = 100  # maps sum to count * 100 (GaussianMSE_norm), as in the baseline
MAP_SIGMA, MAP_KERNEL_SIZE = 5, 20
TILE = (200, 200)
NUM_CAMS = 5
EMBED = 256


# ==========================================
# FUSION MODULES: (features [B,N,512,h,w], ctx, h0, w0) -> BEV features [B,C,200,200]
# ==========================================


class HomographyFusion(nn.Module):
    def __init__(self, compress=False):
        super().__init__()
        self.out_channels = EMBED if compress else 512
        # compress=True: the attention necks' cam_compression (same layers as SpatialCrossAttentionNeck), before selection
        self.cam_compression = (
            nn.Sequential(nn.Conv2d(512, EMBED, kernel_size=1), nn.BatchNorm2d(EMBED), nn.ReLU(inplace=True))
            if compress
            else None
        )
        self.feature_selector = ViewFeatureSelector(in_channels=self.out_channels, hidden_channels=64)
        self.projector = SpatialProjector(bev_shape=TILE, original_image_shape=(1080, 1920))

    def forward(self, features, ctx, h0, w0):
        B, N, C, H, W = features.shape
        if self.cam_compression is not None:
            features = self.cam_compression(features.reshape(B * N, C, H, W)).view(B, N, self.out_channels, H, W)
        feats = self.feature_selector(features)
        hw = torch.tensor([[h0, w0]], device=features.device, dtype=torch.float32).expand(B, 2)
        proj = self.projector(feats, ctx["camera_paras"], ctx["wld_map_paras"], hw)  # [B, N, C, H, W]
        return proj.max(dim=1).values  # baseline: max over cameras


class LegacyAttentionFusion(SpatialCrossAttentionNeck):
    """The legacy neck's modules (cam_compression, learned queries, DeformableSelfAttention,
    DeformableCrossAttention) in a chosen order, without the head_processor convs."""

    out_channels = EMBED

    def __init__(self, order, num_z_anchors=4):
        super().__init__(
            in_channels=512,
            embed_dim=EMBED,
            out_channels=64,
            full_bev_shape=TILE,
            num_cameras=NUM_CAMS,
            num_z_anchors=num_z_anchors,
        )
        self.order = order
        del self.head_processor
        if order == "ca":
            del self.self_attention
        elif num_z_anchors != 4:
            # the legacy neck ties the self-attention's sampling points to the anchor count; keep its 4 points so
            # that only the height anchors change
            self.self_attention = DeformableSelfAttention(embed_dim=EMBED, num_points=4)

    def forward(self, features, ctx, h0, w0):
        refs = ctx["ref_points_full"][:, :, h0 : h0 + TILE[0], w0 : w0 + TILE[1]]
        B, N, C, H, W = features.shape
        feats = self.cam_compression(features.reshape(B * N, C, H, W)).view(B, N, self.embed_dim, H, W)
        q = self.bev_queries.unsqueeze(0).expand(B, -1, -1)
        if self.order == "sa_ca":
            q = self.self_attention(q)
        q = self.cross_attention(bev_queries=q, features_2d=feats, reference_points=refs)
        if self.order == "ca_sa":
            q = self.self_attention(q)
        return q.transpose(1, 2).reshape(B, self.embed_dim, *TILE)


class BEVFormerFusion(nn.Module):
    out_channels = EMBED

    def __init__(self, num_layers=3, grad_checkpoint=True, num_z_anchors=4):
        super().__init__()
        self.neck = BEVFormerEncoderNeck(
            in_channels=512,
            embed_dim=EMBED,
            out_channels=64,
            full_bev_shape=TILE,
            num_cameras=NUM_CAMS,
            num_z_anchors=num_z_anchors,
            num_layers=num_layers,
        )
        self.neck.head_processor = nn.Identity()
        self.neck.grad_checkpoint = grad_checkpoint

    def forward(self, features, ctx, h0, w0):
        return self.neck(features, ctx["ref_points_full"][:, :, h0 : h0 + TILE[0], w0 : w0 + TILE[1]])


class HybridFusion(nn.Module):
    """Forward-backward fusion (FB-BEV): homography_256 as the forward step, one BEVFormer spatial cross-attention
    (8 heads, 4 height anchors per camera, hit-normalised) as the backward step, queried by the homography BEV.

    The homography module is built first, so from the same seed its initial weights equal homography_256's; with the
    zero-initialised cross-attention output projection the variant starts as exactly homography_256 and everything it
    gains is the cross-attention's residual correction.
      query='max'   : the homography BEV (max over cameras)                                       [hca]
      query='stats' : Linear([max, mean, var]); mean / var over the cameras that see the cell's homography plane
                      (projections of the other cameras are zero padding)                         [hca_mv]
      query='static': zero query, so offsets and anchor weights come from the position encoding alone and are the
                      same for every image; in MultiHeadSpatialCrossAttention the query only feeds those two
                      predictors (the values are the camera features)                             [hca_static]"""

    out_channels = EMBED

    def __init__(self, query="max", num_z_anchors=4):
        super().__init__()
        self.homog = HomographyFusion(compress=True)
        self.query = query
        self.query_proj = nn.Linear(3 * EMBED, EMBED) if query == "stats" else None
        self.pos = LearnedBEVPositionalEncoding(EMBED, TILE)
        self.cross_attention = MultiHeadSpatialCrossAttention(
            EMBED, num_heads=8, num_cameras=NUM_CAMS, num_z_anchors=num_z_anchors
        )
        nn.init.zeros_(self.cross_attention.output_proj.weight)  # bias is zero already

    def forward(self, features, ctx, h0, w0):
        base, delta = self.parts(features, ctx, h0, w0)
        return base + delta

    def parts(self, features, ctx, h0, w0):
        """(homography BEV, cross-attention correction), both [B, C, h, w]; forward() returns their sum."""
        B, N, C, H, W = features.shape
        L = TILE[0] * TILE[1]
        hom = self.homog
        feats = hom.cam_compression(features.reshape(B * N, C, H, W)).view(B, N, EMBED, H, W)
        hw = torch.tensor([[h0, w0]], device=feats.device, dtype=torch.float32).expand(B, 2)
        proj = hom.projector(hom.feature_selector(feats), ctx["camera_paras"], ctx["wld_map_paras"], hw)  # [B,N,C,h,w]
        base = proj.max(dim=1).values  # = homography_256
        refs = ctx["ref_points_full"][:, :, h0 : h0 + TILE[0], w0 : w0 + TILE[1]]
        refs = refs.reshape(B, N, L, refs.shape[-2], 2)

        if self.query == "stats":
            # visibility of the homography plane: project a map of ones (1 inside the image, 0 outside)
            seen = hom.projector(feats.new_ones(B, N, 1, H, W), ctx["camera_paras"], ctx["wld_map_paras"], hw)
            n = seen.sum(1).clamp(min=1e-3)
            mean = (proj * seen).sum(1) / n
            var = (((proj - mean.unsqueeze(1)) ** 2) * seen).sum(1) / n
            q = self.query_proj(torch.cat([base, mean, var], 1).flatten(2).transpose(1, 2))
        elif self.query == "static":
            q = torch.zeros_like(base.flatten(2).transpose(1, 2))  # position encoding only
        else:
            q = base.flatten(2).transpose(1, 2)  # [B, L, C]
        pos = self.pos().unsqueeze(0).expand(B, -1, -1).to(q.dtype)
        delta = self.cross_attention(q, pos, feats, refs)  # [B, L, C]
        return base, delta.transpose(1, 2).reshape(B, EMBED, *TILE)


class MultiViewHeightFusion(nn.Module):
    """MultiViewHeightAttention over the cached camera x height reference points.
      residual=False: own cam_compression (built first, as in the legacy necks) -> attention output   [mvha]
      residual=True : homography_256 (built first, so from the same seed its weights equal homography_256's) plus
                      the attention as a zero-initialised residual; the homography BEV also enters the query  [mvha_res]
    Set self.attention.keep_aux to keep the per-cell height / camera weight marginals (diagnostics)."""

    out_channels = EMBED

    def __init__(self, residual=False, num_z_anchors=8, grad_checkpoint=True):
        super().__init__()
        self.homog = HomographyFusion(compress=True) if residual else None
        self.cam_compression = (
            None
            if residual
            else nn.Sequential(nn.Conv2d(512, EMBED, kernel_size=1), nn.BatchNorm2d(EMBED), nn.ReLU(inplace=True))
        )
        self.attention = MultiViewHeightAttention(
            EMBED,
            num_cameras=NUM_CAMS,
            num_z_anchors=num_z_anchors,
            base_dim=EMBED if residual else None,
            grad_checkpoint=grad_checkpoint,
            zero_init_output=residual,
        )

    def forward(self, features, ctx, h0, w0):
        base, out = self.parts(features, ctx, h0, w0)
        return out if base is None else base + out

    def parts(self, features, ctx, h0, w0):
        """(homography BEV or None, attention output), both [B, C, h, w]; forward() returns their sum."""
        B, N, C, H, W = features.shape
        L = TILE[0] * TILE[1]
        refs = ctx["ref_points_full"][:, :, h0 : h0 + TILE[0], w0 : w0 + TILE[1]]
        refs = refs.reshape(B, N, L, refs.shape[-2], 2)
        if self.homog is None:
            feats = self.cam_compression(features.reshape(B * N, C, H, W)).view(B, N, EMBED, H, W)
            return None, self.attention(feats, refs).transpose(1, 2).reshape(B, EMBED, *TILE)
        hom = self.homog
        feats = hom.cam_compression(features.reshape(B * N, C, H, W)).view(B, N, EMBED, H, W)
        hw = torch.tensor([[h0, w0]], device=feats.device, dtype=torch.float32).expand(B, 2)
        base = (
            hom.projector(hom.feature_selector(feats), ctx["camera_paras"], ctx["wld_map_paras"], hw).max(dim=1).values
        )
        delta = self.attention(feats, refs, base=base.flatten(2).transpose(1, 2))
        return base, delta.transpose(1, 2).reshape(B, EMBED, *TILE)


def build_fusion(variant, args):
    if variant in ("homography", "homography_256"):
        return HomographyFusion(compress=variant == "homography_256")
    if variant in ("ca", "sa_ca", "ca_sa"):
        return LegacyAttentionFusion(variant, args.z_anchors)
    if variant == "bev3":
        return BEVFormerFusion(
            num_layers=args.bev_layers, grad_checkpoint=not args.no_grad_checkpoint, num_z_anchors=args.z_anchors
        )
    if variant in ("hca", "hca_mv", "hca_static"):
        query = {"hca": "max", "hca_mv": "stats", "hca_static": "static"}[variant]
        return HybridFusion(query=query, num_z_anchors=args.z_anchors)
    if variant in ("mvha", "mvha_res"):
        return MultiViewHeightFusion(
            residual=variant == "mvha_res", num_z_anchors=args.z_anchors, grad_checkpoint=not args.no_grad_checkpoint
        )
    raise ValueError(variant)


class AblationModel(nn.Module):
    def __init__(self, variant, args):
        super().__init__()
        self.backbone = DilatedResNet18(pretrained=True)
        ck = torch.load(args.backbone, map_location="cpu", weights_only=False)
        self.backbone.load_state_dict(
            {k.replace("backbone.", ""): v for k, v in ck["model_state_dict"].items() if k.startswith("backbone.")}
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        # head first, from the same seed, so every 256-channel variant starts from the identical head
        torch.manual_seed(args.seed)
        self.head = BaselineDecoder(num_cam=NUM_CAMS, out_channel=512 if variant == "homography" else EMBED)
        torch.manual_seed(args.seed + 1)
        self.fusion = build_fusion(variant, args)
        assert self.fusion.out_channels == self.head.feat_before_merge.in_channels

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # frozen backbone keeps its BatchNorm statistics
        return self

    def features(self, images):
        B, N = images.shape[:2]
        with torch.no_grad():
            f = self.backbone(images.flatten(0, 1))
        return f.view(B, N, *f.shape[1:])

    def predict_tile(self, features, ctx, h0, w0):
        bev = self.fusion(features, ctx, h0, w0)
        return self.head(bev.unsqueeze(1))  # BaselineDecoder max-pools over dim 1 (a single "view" here)


# ==========================================
# DATA
# ==========================================


def load_split(split, cams_cfg, excluded, refs_file="master_refs.pt", z_anchors=4):
    split_dir = os.path.join(src_dir, "datasets", "SynMVCrowd", "data", split)
    out = []
    for scene in sorted(os.listdir(split_dir), key=lambda s: int(s.split("_")[1]) if "_" in s else 0):
        if not os.path.isdir(os.path.join(split_dir, scene)):
            continue
        if scene in excluded:
            continue
        cams = cams_cfg.get(scene, {}).get(str(NUM_CAMS))
        if cams is None:
            print(f"[skip] {split}/{scene}: no K-means camera set for k={NUM_CAMS}")
            continue
        ds = SynMVCrowdDataset(
            scene_dir=os.path.join(split_dir, scene),
            camera_ids=cams,
            num_sampled_cams=NUM_CAMS,
            load_refs=True,
            refs_file=refs_file,
        )
        assert ds.active_refs.shape[-2] == z_anchors, (
            f"{scene}/{refs_file} has {ds.active_refs.shape[-2]} anchors, --z_anchors is {z_anchors}"
        )
        out.append((scene, ds))
    return out


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(scenes, frames_per_scene, seed, num_workers, shuffle=False):
    """Fixed, seeded selection of frames per scene; the generator also fixes crops and camera order in the workers,
    so every variant sees exactly the same samples for the same seed."""
    rng = random.Random(seed)
    indices, offset = [], 0
    for _, ds in scenes:
        n = len(ds) if frames_per_scene <= 0 else min(frames_per_scene, len(ds))
        for f in sorted(rng.sample(range(len(ds)), n)):
            indices.append(offset + f)
        offset += len(ds)
    if shuffle:
        rng.shuffle(indices)
    return DataLoader(
        ConcatDataset([ds for _, ds in scenes]),
        batch_size=1,
        sampler=indices,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=True,
    )


def tile_starts(full, tile):
    if full <= tile:
        return [0]
    return sorted(set(list(range(0, full - tile + 1, tile)) + [full - tile]))


def gt_count_in_map(gt_world_xy, wld, shape):
    xy = gt_world_xy.reshape(-1, 2)
    r, a, b = wld[0].item(), wld[1].item(), wld[2].item()
    w_min, h_min = wld[6].item(), wld[7].item()
    cols = torch.round((xy[:, 0] - w_min + a) * r)
    rows = torch.round((xy[:, 1] - h_min + b) * r)
    return int(((rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])).sum())


# ==========================================
# EVALUATION (whole scene, 200x200 tiles, identical for every variant)
# ==========================================


@torch.no_grad()
def evaluate(model, scenes, loader, device, desc):
    model.eval()
    names = [n for n, _ in scenes]
    bounds = np.cumsum([0] + [len(ds) for _, ds in scenes])
    errs, per_scene = [], defaultdict(list)
    for sample, idx in zip(tqdm(loader, desc=desc, mininterval=10), loader.sampler):
        scene = names[int(np.searchsorted(bounds, idx, side="right") - 1)]
        images = sample["images"].to(device)
        wld = sample["wld_map_paras"].to(device)
        ctx = {
            "ref_points_full": sample["ref_points_full"].to(device),
            "camera_paras": sample["camera_paras"].to(device),
            "wld_map_paras": wld,
        }
        H, W = int(wld[0, 3].item()), int(wld[0, 4].item())
        pred = torch.zeros((H, W), device=device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            feats = model.features(images)
            for h0 in tile_starts(H, TILE[0]):
                for w0 in tile_starts(W, TILE[1]):
                    out = model.predict_tile(feats, ctx, h0, w0)[0, 0].float()
                    hh, ww = min(TILE[0], H - h0), min(TILE[1], W - w0)
                    pred[h0 : h0 + hh, w0 : w0 + ww] = out[:hh, :ww]
        gt = gt_count_in_map(sample["gt_world_xy"][0], sample["wld_map_paras"][0], (H, W))
        err = pred.sum().item() / SCALE - gt
        errs.append((err, gt))
        per_scene[scene].append(abs(err))
    e = np.array([x for x, _ in errs])
    return {
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e**2).mean())),
        "bias": float(e.mean()),
        "nae": float(np.mean([min(abs(x) / g, 1.0) if g else 0.0 for x, g in errs])),
        "n": len(errs),
        "per_scene_mae": {k: float(np.mean(v)) for k, v in per_scene.items()},
    }


@torch.no_grad()
def precise_bn(model, loader, device):
    """Re-estimate the fusion's BatchNorm running statistics as a cumulative average over `loader`
    (one frame per train scene), as the proposed training does before validation."""
    mods = [m for m in model.fusion.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not mods:
        return
    old = [m.momentum for m in mods]
    for m in mods:
        m.reset_running_stats()
        m.momentum = None
    model.eval()
    for m in mods:
        m.train()
    for sample in loader:
        ctx = {
            "ref_points_full": sample["ref_points_full"].to(device),
            "camera_paras": sample["camera_paras"].to(device),
            "wld_map_paras": sample["wld_map_paras"].to(device),
        }
        hw = sample["hw_random"][0]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            feats = model.features(sample["images"].to(device))
            for p in range(hw.shape[-1]):
                model.fusion(feats, ctx, int(hw[0, p]), int(hw[1, p]))
    for m, mom in zip(mods, old):
        m.momentum = mom
    model.eval()


class EMA:
    """Exponential moving average of the trainable weights (fusion + head), kept in a frozen copy of the model that is
    used for validation, selection and testing. Buffers (BatchNorm statistics) are copied; precise-BN re-estimates them
    before every evaluation anyway. Decay warms up as min(decay, (1 + n) / (10 + n))."""

    def __init__(self, model, decay):
        self.decay, self.n = decay, 0
        self.model = copy.deepcopy(model).eval()
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.pairs = [(e, q) for e, q in zip(self.model.parameters(), model.parameters()) if q.requires_grad]
        self.buffers = list(zip(self.model.buffers(), model.buffers()))

    @torch.no_grad()
    def update(self):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, q in self.pairs:
            e.lerp_(q.detach(), 1.0 - d)
        for e, b in self.buffers:
            e.copy_(b)

    def state_dict(self):
        return {"model": self.model.state_dict(), "n": self.n}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.n = state["n"]


# ==========================================
# TRAINING ONE VARIANT
# ==========================================


def run_variant(variant, args, data, device):
    out_dir = os.path.join(args.out_root, variant)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isfile(os.path.join(out_dir, "result.json")) and not args.overwrite:
        print(f"[skip] {variant}: result.json exists (use --overwrite to retrain)")
        return
    train_scenes, val_scenes, test_scenes = data
    model = AblationModel(variant, args).to(device)
    if args.init_from:
        # fine-tune a finished variant (its evaluated weights: EMA if it had one), like train_baseline.py --init_from;
        # the EMA below starts from these weights. Ignored on resume (last.pth is loaded over it).
        init = torch.load(os.path.join(src_dir, args.init_from), map_location="cpu", weights_only=False)
        model.load_state_dict(
            init["ema_state_dict"]["model"] if init.get("ema_state_dict") else init["model_state_dict"]
        )
        print(
            f"[init] {variant}: weights from {args.init_from} (epoch {init['epoch']}, "
            f"{'EMA' if init.get('ema_state_dict') else 'raw'})"
        )
        del init
    if args.freeze_fusion:
        # decoder-only fine-tune on the frozen fusion (the head-budget control of doc/head_ablation.md §8.14):
        # the fusion keeps the --init_from weights (eval mode, not optimised), only the decoder trains
        assert args.init_from, "--freeze_fusion needs --init_from"
        for p in model.fusion.parameters():
            p.requires_grad = False
        print(f"[freeze] {variant}: fusion frozen (eval mode); training the decoder only")
    n_fusion = sum(p.numel() for p in model.fusion.parameters())
    n_head = sum(p.numel() for p in model.head.parameters())
    print(
        f"\n{'=' * 90}\n[{variant}] fusion {n_fusion:,} params | head {n_head:,} params | out channels "
        f"{model.fusion.out_channels}\n{'=' * 90}"
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    ema = None if args.no_ema else EMA(model, args.ema_decay)
    eval_model = model if ema is None else ema.model  # the weights that are validated, selected and tested
    if args.optimizer == "adam":
        opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.SGD(trainable, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    iters = sum(min(args.frames_per_scene, len(ds)) for _, ds in train_scenes)
    total, warm = args.epochs * iters, int(args.warmup_ratio * args.epochs * iters)
    if args.lr_schedule == "constant":  # as the baseline's OT fine-tune (no warmup, no decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 1.0)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda it: (it / max(warm, 1))
            if it < warm
            else 0.5 * (1 + math.cos(math.pi * (it - warm) / max(total - warm, 1))),
        )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = GaussianMSE_norm(scale=SCALE).to(device)
    ot_criterion = SynMVCrowdGeneralizedLoss(cost="ed", scale=SCALE) if args.loss == "ot" else None
    kernel = get_map_kernel(sigma=MAP_SIGMA, kernel_size=MAP_KERNEL_SIZE).to(device)

    # Built fresh from a fixed seed on every use: a reused DataLoader draws a new worker seed per pass, which changed
    # the camera order (val) and the crops / camera order (precise-BN) from one epoch to the next.
    def val_loader():
        return make_loader(val_scenes, args.val_frames_per_scene, args.val_seed, args.num_workers)

    def bn_loader():
        return make_loader(train_scenes, 1, args.val_seed + 7, args.num_workers)

    log_path = os.path.join(out_dir, "log.csv")
    last_path = os.path.join(out_dir, "last.pth")
    best = {"mae": float("inf"), "epoch": 0}
    start_epoch = 1
    # Resume an interrupted variant from last.pth (exact: optimizer, schedule, scaler and best-so-far are restored,
    # and every epoch's training data is seeded by its epoch number). --overwrite starts from scratch instead.
    last = (
        torch.load(last_path, map_location=device, weights_only=False)
        if os.path.isfile(last_path) and not args.overwrite
        else None
    )
    if last is not None and "optimizer_state_dict" in last:
        model.load_state_dict(last["model_state_dict"])
        opt.load_state_dict(last["optimizer_state_dict"])
        sched.load_state_dict(last["scheduler_state_dict"])
        scaler.load_state_dict(last["scaler_state_dict"])
        if ema is not None:
            ema.load_state_dict(last["ema_state_dict"])
        best, start_epoch = last["best"], last["epoch"] + 1
        print(
            f"[resume] {variant}: continuing at epoch {start_epoch} (best val MAE {best['mae']:.2f} at epoch {best['epoch']})"
        )
    else:
        if last is not None:
            print(
                f"[resume] {variant}: last.pth has no optimizer state (older script version) -- starting from scratch"
            )
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_patch_mae",
                    "val_mae",
                    "val_rmse",
                    "val_nae",
                    "val_bias",
                    "lr",
                    "train_s",
                    "val_s",
                    "oom_skipped",
                    "peak_gpu_gib",
                ]
            )
    del last
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if args.freeze_fusion:
            model.fusion.eval()
        t0 = time.time()
        loader = make_loader(
            train_scenes, args.frames_per_scene, args.seed * 1000 + epoch, args.num_workers, shuffle=True
        )
        losses, maes, skipped, oom = [], [], 0, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for sample in tqdm(loader, desc=f"[{variant}] epoch {epoch}/{args.epochs}", mininterval=10):
            try:
                ctx = {
                    "ref_points_full": sample["ref_points_full"].to(device, non_blocking=True),
                    "camera_paras": sample["camera_paras"].to(device),
                    "wld_map_paras": sample["wld_map_paras"].to(device),
                }
                hw = sample["hw_random"][0]
                target = sample["density_patches"].to(device).permute(1, 0, 2, 3).float()  # [P, 1, H, W]
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    feats = model.features(sample["images"].to(device, non_blocking=True))
                    pred = torch.cat(
                        [model.predict_tile(feats, ctx, int(hw[0, p]), int(hw[1, p])) for p in range(hw.shape[-1])]
                    )  # [P, 1, H, W]
                if ot_criterion is None:
                    loss, map_gt = criterion(pred.float(), target, kernel)
                else:
                    # as train_baseline.py --loss ot: OT loss per patch against its GT points, mean over the patches
                    loss = (
                        sum(
                            ot_criterion(pred[p : p + 1].float(), patch_points(target[p, 0]))
                            for p in range(pred.shape[0])
                        )
                        / pred.shape[0]
                    )
                    map_gt = target * SCALE  # only for the logged patch MAE (map sums are count * SCALE)
                if not torch.isfinite(loss):
                    skipped += 1
                    sched.step()
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if args.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                scaler.step(opt)
                scaler.update()
            except RuntimeError as e:  # torch.OutOfMemoryError and the driver-level "CUDA error: out of memory"
                if "out of memory" not in str(e):
                    raise
                if "CUDA error" in str(e):
                    # Driver-level error: the CUDA context is unusable afterwards (even empty_cache fails), so exit and
                    # let remote_scripts/run_neck_ablation.ps1 restart the process, which resumes from last.pth.
                    raise
                oom += 1
                print(
                    f"\n[warning] {variant} epoch {epoch}: CUDA out of memory, step skipped ({oom} this epoch); "
                    f"another program using the GPU? allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB"
                )
                opt.zero_grad(set_to_none=True)
                loss = pred = feats = map_gt = None
                torch.cuda.empty_cache()
                if oom > args.max_oom_per_epoch:
                    raise RuntimeError(
                        f"{oom} out-of-memory steps in epoch {epoch}; aborting (restart the same "
                        f"command to resume from epoch {epoch})"
                    ) from e
                sched.step()
                continue
            sched.step()
            if ema is not None:
                ema.update()
            losses.append(loss.item())
            maes.append(abs(pred.float().sum().item() - map_gt.float().sum().item()) / SCALE / pred.shape[0])
        train_s = time.time() - t0
        peak_gib = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0

        lr = opt.param_groups[0]["lr"]
        if epoch % args.val_every and epoch != args.epochs:  # --val_every: train-only epoch, checkpoint for resume
            print(
                f">>> [{variant}] epoch {epoch} | train loss {np.mean(losses):.4f} | patch MAE {np.mean(maes):.2f} | "
                f"skipped {skipped} | OOM {oom} | peak {peak_gib:.1f} GiB | (no validation) | lr {lr:.2e} | {train_s:.0f}s"
            )
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        epoch,
                        np.mean(losses),
                        np.mean(maes),
                        "",
                        "",
                        "",
                        "",
                        lr,
                        round(train_s),
                        0,
                        oom,
                        round(peak_gib, 2),
                    ]
                )
            torch.save(
                {
                    "epoch": epoch,
                    "variant": variant,
                    "model_state_dict": model.state_dict(),
                    "val": None,
                    "ema_state_dict": None if ema is None else ema.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best": best,
                    "args": vars(args),
                },
                last_path,
            )
            continue
        t1 = time.time()
        precise_bn(eval_model, bn_loader(), device)
        val = evaluate(
            eval_model, val_scenes, val_loader(), device, f"[{variant}] val{'' if ema is None else ' (EMA)'}"
        )
        val_s = time.time() - t1
        print(
            f">>> [{variant}] epoch {epoch} | train loss {np.mean(losses):.4f} | patch MAE {np.mean(maes):.2f} | "
            f"skipped {skipped} | OOM {oom} | peak {peak_gib:.1f} GiB | VAL MAE {val['mae']:.2f} RMSE {val['rmse']:.2f} NAE {val['nae']:.3f} "
            f"bias {val['bias']:+.2f} | lr {lr:.2e} | {train_s:.0f}s + {val_s:.0f}s"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    np.mean(losses),
                    np.mean(maes),
                    val["mae"],
                    val["rmse"],
                    val["nae"],
                    val["bias"],
                    lr,
                    round(train_s),
                    round(val_s),
                    oom,
                    round(peak_gib, 2),
                ]
            )
        if val["mae"] < best["mae"]:
            best = {"mae": val["mae"], "epoch": epoch, "val": val}
            # model_state_dict = the evaluated weights (EMA unless --no_ema); load into AblationModel(variant, args)
            torch.save(
                {
                    "epoch": epoch,
                    "variant": variant,
                    "model_state_dict": eval_model.state_dict(),
                    "val": val,
                    "ema": ema is not None,
                    "args": vars(args),
                },
                os.path.join(out_dir, "best.pth"),
            )
            print(f"    new best val MAE {val['mae']:.2f}")
        torch.save(
            {
                "epoch": epoch,
                "variant": variant,
                "model_state_dict": model.state_dict(),
                "val": val,
                "ema_state_dict": None if ema is None else ema.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best": best,
                "args": vars(args),
            },
            last_path,
        )
        if args.patience and epoch - best["epoch"] >= args.patience:  # checked at validation epochs
            print(f">>> [{variant}] early stop at epoch {epoch} (best {best['mae']:.2f} at epoch {best['epoch']})")
            break

    test = test_final = None
    final_epoch = epoch if args.epochs >= start_epoch else start_epoch - 1
    final_val = val if args.epochs >= start_epoch else None
    if not args.no_test:

        def test_loader():
            return make_loader(test_scenes, args.test_frames_per_scene, args.val_seed + 13, args.num_workers)

        # 1) final weights (eval_model holds them right after training, or restored from last.pth on resume)
        precise_bn(eval_model, bn_loader(), device)
        test_final = evaluate(eval_model, test_scenes, test_loader(), device, f"[{variant}] test (final)")
        print(
            f">>> [{variant}] TEST final (epoch {final_epoch}) MAE {test_final['mae']:.2f} RMSE {test_final['rmse']:.2f} "
            f"NAE {test_final['nae']:.3f} bias {test_final['bias']:+.2f}"
        )
        # 2) best-val weights
        eval_model.load_state_dict(
            torch.load(os.path.join(out_dir, "best.pth"), map_location=device, weights_only=False)["model_state_dict"]
        )
        precise_bn(eval_model, bn_loader(), device)
        test = evaluate(eval_model, test_scenes, test_loader(), device, f"[{variant}] test (best)")
        print(
            f">>> [{variant}] TEST best (epoch {best['epoch']}) MAE {test['mae']:.2f} RMSE {test['rmse']:.2f} "
            f"NAE {test['nae']:.3f} bias {test['bias']:+.2f}"
        )
    result = {
        "variant": variant,
        "fusion_params": n_fusion,
        "head_params": n_head,
        "best_epoch": best["epoch"],
        "final_epoch": final_epoch,
        "ema": ema is not None,
        "val": best["val"],
        "final_val": final_val,
        "test": test,
        "test_final": test_final,
        "args": vars(args),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    del model, opt, ema, eval_model
    torch.cuda.empty_cache()


def summarize(out_root):
    rows = []
    for v in VARIANTS:
        p = os.path.join(out_root, v, "result.json")
        if os.path.isfile(p):
            with open(p) as f:
                rows.append(json.load(f))
    if not rows:
        print(f"[summary] no finished variants in {out_root}")
        return
    nan = float("nan")
    print(f"\n{'=' * 124}\nNECK ABLATION SUMMARY  ({out_root})\n{'=' * 124}")
    print(
        f"{'variant':<16}{'fusion params':>14}{'EMA':>5}{'best ep':>8} | {'val MAE':>8}{'NAE':>7} | "
        f"{'test best: MAE':>15}{'RMSE':>8}{'NAE':>7}{'bias':>8} | {'test final: MAE':>16}{'bias':>8}"
    )
    for r in rows:
        v, t, tf = r["val"], r["test"] or {}, r.get("test_final") or {}
        print(
            f"{r['variant']:<16}{r['fusion_params']:>14,}{('yes' if r.get('ema') else 'no'):>5}{r['best_epoch']:>8} | "
            f"{v['mae']:>8.2f}{v['nae']:>7.3f} | {t.get('mae', nan):>15.2f}{t.get('rmse', nan):>8.2f}"
            f"{t.get('nae', nan):>7.3f}{t.get('bias', nan):>+8.2f} | {tf.get('mae', nan):>16.2f}{tf.get('bias', nan):>+8.2f}"
        )
    scenes = sorted(
        {s for r in rows if r["test"] for s in r["test"]["per_scene_mae"]}, key=lambda s: int(s.split("_")[1])
    )
    if scenes:
        print("\nTest MAE per scene (best-val weights)")
        print(f"{'variant':<12}" + "".join(f"{s.replace('scene_', 's'):>8}" for s in scenes))
        for r in rows:
            if r["test"]:
                print(
                    f"{r['variant']:<12}"
                    + "".join(f"{r['test']['per_scene_mae'].get(s, float('nan')):>8.1f}" for s in scenes)
                )
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description="Neck ablation: identical backbone, head, loss and data; only the fusion varies."
    )
    ap.add_argument("--run_name", required=True, help="folder under runs/neck_ablation/")
    ap.add_argument("--variants", default=",".join(VARIANTS), help=f"comma-separated subset of {VARIANTS}")
    ap.add_argument("--summarize", action="store_true", help="only print the table of finished variants")
    ap.add_argument("--overwrite", action="store_true", help="retrain variants that already have a result.json")
    ap.add_argument(
        "--backbone", default=os.path.join(src_dir, "runs", "train_2d", "new_best", "resnet18_2d_pretrained_ep4.pth")
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--frames_per_scene", type=int, default=10, help="training frames per scene per epoch")
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument(
        "--lr_schedule",
        choices=["cosine", "constant"],
        default="cosine",
        help="cosine = warmup + cosine (ablation default); constant = the baseline's OT fine-tune",
    )
    ap.add_argument(
        "--loss",
        choices=["mse", "ot"],
        default="mse",
        help="mse = GaussianMSE_norm (ablation default); ot = the baseline's OT loss "
        "(losses/ot_loss/genloss_synmvcrowd.py, 'ed' cost), meant with --init_from",
    )
    ap.add_argument(
        "--init_from",
        default=None,
        help="last.pth of a finished variant (relative to src) to fine-tune from, e.g. "
        "runs/neck_ablation/ablation_v2/ca_sa/last.pth; must match the variant and refs",
    )
    ap.add_argument(
        "--freeze_fusion",
        action="store_true",
        help="with --init_from: keep the fusion frozen (eval mode) and fine-tune the decoder only",
    )
    ap.add_argument("--max_grad_norm", type=float, default=5.0, help="0 = no clipping")
    ap.add_argument(
        "--patience", type=int, default=0, help="early stop after N epochs without a new best val MAE (0 = off)"
    )
    ap.add_argument("--val_frames_per_scene", type=int, default=20, help="fixed val frames per scene (0 = all)")
    ap.add_argument("--val_every", type=int, default=1, help="validate every N epochs (and always after the last)")
    ap.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="per-step EMA decay of the evaluated weights (0.999 ~ last 1000 steps ~ 3.5 epochs)",
    )
    ap.add_argument("--no_ema", action="store_true", help="validate / select / test the raw weights instead of the EMA")
    ap.add_argument("--test_frames_per_scene", type=int, default=20, help="0 = all test frames")
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--bev_layers", type=int, default=3)
    ap.add_argument(
        "--refs_file",
        default="master_refs.pt",
        help="reference-point cache per scene; master_refs_k8.pt = 8 anchors from the feet of people 2.5 m "
        "below the average head plane to the heads of people 2.5 m above it (utils/cache_reference.py)",
    )
    ap.add_argument("--z_anchors", type=int, default=4, help="height anchors per pillar; must match --refs_file")
    ap.add_argument(
        "--no_grad_checkpoint",
        action="store_true",
        help="bev3 / mvha: keep activations instead of recomputing them in the backward pass (more VRAM)",
    )
    ap.add_argument(
        "--max_oom_per_epoch",
        type=int,
        default=10,
        help="skip up to N out-of-memory training steps per epoch before aborting (resumable)",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--val_seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_root", default=None, help="output folder (default: runs/neck_ablation/<run_name>)")
    args = ap.parse_args()
    args.out_root = (
        os.path.abspath(args.out_root)
        if args.out_root
        else os.path.join(src_dir, "runs", "neck_ablation", args.run_name)
    )
    os.makedirs(args.out_root, exist_ok=True)
    if args.summarize:
        summarize(args.out_root)
        return

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}; choose from {VARIANTS}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"--- neck ablation '{args.run_name}' on {device}: {variants} ---\n{vars(args)}")

    with open(os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")) as f:
        cams_cfg = json.load(f)
    excluded = load_excluded_scenes()
    data = tuple(
        load_split(s, cams_cfg, excluded, args.refs_file, args.z_anchors) for s in ("train", "validate", "test")
    )
    print(f"scenes: {len(data[0])} train / {len(data[1])} val / {len(data[2])} test")

    for v in variants:
        run_variant(v, args, data, device)
    summarize(args.out_root)


if __name__ == "__main__":
    main()
