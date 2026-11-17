"""Loss functions and the per-patch loss of each curriculum stage."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.gaussian_mse import GaussianMSE_norm, get_map_kernel
from losses.ot_loss.genloss import GeneralizedLoss
from losses.ot_loss.genloss_synmvcrowd import SynMVCrowdGeneralizedLoss, patch_points

from . import config as cfg
from .targets import cell_routing_targets, extract_peaks_via_nms, filter_points_outside_mask, region_points


def binary_focal_loss(pred, target, alpha=0.75, gamma=2.0, eps=1e-6):
    """Binary focal loss (Lin et al., 2017); alpha weights the POSITIVE (dense = 1) class, tuned for the oracle mask's
    ~3-10% coverage, not RetinaNet's 0.25."""
    pred = torch.clamp(pred, min=eps, max=1.0 - eps)
    pt = torch.where(target == 1, pred, 1.0 - pred)
    alpha_t = torch.where(target == 1, alpha, 1.0 - alpha)
    loss = -alpha_t * (1.0 - pt).pow(gamma) * torch.log(pt)
    return loss.mean()


def tversky_loss(pred, target, alpha=0.5, beta=0.5, smooth=1e-6):
    """Tversky loss (Salehi et al., 2017); alpha = beta = 0.5 is Dice. Ratio-based, so invariant to the oracle mask's
    per-patch coverage swings."""
    pred = torch.clamp(pred, min=0.0, max=1.0)
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky_index


def normalized_mse_loss(pred, target, target_mass, min_mass=10.0, eps=1e-6):
    """MSE divided once by the target mass, floored at min_mass so a near-empty patch cannot blow up the gradient."""
    raw_mse = F.mse_loss(pred, target)
    scale = torch.clamp(target_mass, min=min_mass) + eps
    return raw_mse / scale


def penalty_reduced_focal_loss(pred, dots_map, gaussian_map, region, alpha=2.0, beta=4.0, eps=1e-6):
    """CornerNet/CenterNet-style penalty-reduced focal loss for a point heatmap.
    pred: sigmoid heatmap [1,1,H,W]; dots_map: person-pixel count map (>0 at centres); gaussian_map: peak-normalised
    Gaussian-blurred target used to reduce the penalty near centres; region: 1 inside the sparse region, else 0.
    Normalised by the number of positives (min 1)."""
    p = pred.clamp(min=eps, max=1.0 - eps)
    is_pos = (dots_map > 0).float()
    pos = is_pos * region
    neg = (1.0 - is_pos) * region
    y = gaussian_map.clamp(min=0.0, max=0.999)
    pos_loss = -((1.0 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1.0 - y) ** beta) * (p**alpha) * torch.log(1.0 - p) * neg
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)


def relative_cell_loss(pred, target, sizes, eps=1e-3):
    """Mean over cell sizes of MSE(avg-pooled pred, avg-pooled target) / mean(pooled target^2) -- a dimensionless
    (1 - skill)-like loss on the spatial allocation of density at coarser scales."""
    total = pred.new_zeros(())
    for sz in sizes:
        pc = F.avg_pool2d(pred, kernel_size=sz, stride=sz)
        tc = F.avg_pool2d(target, kernel_size=sz, stride=sz)
        total = total + F.mse_loss(pc, tc) / tc.pow(2).mean().clamp(min=eps)
    return total / len(sizes)


def weighted_routing_bce(cell_prob, targets, eps=1e-6):
    """Cost-sensitive BCE of the router's per-cell probability against the performance labels, normalised by the
    total weight (0 if every cell is a tie)."""
    p = cell_prob.clamp(eps, 1.0 - eps)
    y, w = targets["label"], targets["weight"]
    bce = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
    return (w * bce).sum() / w.sum().clamp(min=eps)


def mask_loss(pred_mask, gt_mask):
    """Router mask vs the oracle mask: Tversky (STAGE3_LOSS_TYPE 'dice') or focal."""
    if cfg.STAGE3_LOSS_TYPE == "dice":
        return tversky_loss(pred_mask, gt_mask, alpha=cfg.TVERSKY_ALPHA, beta=cfg.TVERSKY_BETA)
    return binary_focal_loss(pred_mask, gt_mask, alpha=cfg.FOCAL_ALPHA, gamma=cfg.FOCAL_GAMMA)


class RoutingLabelGenerator(nn.Module):
    """Legacy Stage 3 soft labels (STAGE3_PERF_WEIGHT > 0, used by no recipe). Its dense_map / GAUSSIAN_SCALE has
    the wrong units (one person integrates to the kernel sum, not GAUSSIAN_SCALE); fix before using it."""

    def __init__(self, temperature=1.0, grid_size=10):
        super().__init__()
        self.T = temperature
        self.grid_size = grid_size

    @torch.no_grad()
    def forward(self, sparse_points, dense_map, target_patch):
        B, C, H, W = dense_map.shape
        device = dense_map.device

        sparse_map = torch.zeros((1, 1, H, W), device=device)
        if sparse_points.shape[0] > 0:
            xs = torch.clamp((sparse_points[:, 0] * W).long(), 0, W - 1)
            ys = torch.clamp((sparse_points[:, 1] * H).long(), 0, H - 1)
            sparse_map[0, 0, ys, xs] = 1.0

        pool_factor = self.grid_size**2
        sparse_grid = F.avg_pool2d(sparse_map, kernel_size=self.grid_size, stride=self.grid_size) * pool_factor
        dense_grid = (
            F.avg_pool2d(dense_map / cfg.GAUSSIAN_SCALE, kernel_size=self.grid_size, stride=self.grid_size)
            * pool_factor
        )
        gt_grid = F.avg_pool2d(target_patch, kernel_size=self.grid_size, stride=self.grid_size) * pool_factor

        err_sparse = torch.abs(sparse_grid - gt_grid)
        err_dense = torch.abs(dense_grid - gt_grid)

        soft_target_grid = torch.sigmoid((err_sparse - err_dense) / self.T)
        soft_target_map = F.interpolate(soft_target_grid, size=(H, W), mode="nearest")

        return soft_target_map


def stage2_regions(gt_mask):
    """(dense, sparse) supervision regions: the oracle split, or the whole patch for both with STAGE2_UNMASKED."""
    if cfg.STAGE2_UNMASKED:
        return torch.ones_like(gt_mask), torch.ones_like(gt_mask)
    return gt_mask, 1.0 - gt_mask


class StageLosses:
    """Criteria, units and peak thresholds of one stage, and its per-patch loss (training, and Stage 3/4 validation).

    Units: get_map_kernel is peak-normalised, so one person integrates to map_kernel.sum() (~31.4) in map_gt_conv,
    not GAUSSIAN_SCALE. Stage 1 counts in GAUSSIAN_SCALE units (GaussianMSE_norm rescales internally); Stages 2-4
    count in kernel-sum units (KERNEL_MASS_PER_PERSON), and Stage 4's (and --joint's) MSE is switched to them."""

    def __init__(self, stage, device, mass_normalize_dense=False, mass_normalize_sparse=False):
        self.stage, self.device = stage, device
        self.mass_normalize_dense, self.mass_normalize_sparse = mass_normalize_dense, mass_normalize_sparse
        self.mse_criterion = GaussianMSE_norm(scale=cfg.GAUSSIAN_SCALE).to(device).float()
        self.ot_criterion = GeneralizedLoss(
            loss_type="md", reduction="mean", scale=cfg.OT_SCALE, device=device, top_k=cfg.SPARSE_GENLOSS_TOPK
        )
        # --dense_loss ot: the baseline's OT loss on the dense map, in its units (one person = GAUSSIAN_SCALE)
        self.dense_ot_criterion = (
            SynMVCrowdGeneralizedLoss(cost="ed", scale=cfg.GAUSSIAN_SCALE) if cfg.DENSE_LOSS == "ot" else None
        )
        self.dense_to_count_units = cfg.GAUSSIAN_SCALE / float(
            get_map_kernel(sigma=cfg.GAUSSIAN_SIGMA, kernel_size=cfg.GAUSSIAN_KERNEL_SIZE).sum()
        )
        self.bce_criterion = nn.BCELoss().to(device)
        self.map_kernel = get_map_kernel(sigma=cfg.GAUSSIAN_SIGMA, kernel_size=cfg.GAUSSIAN_KERNEL_SIZE).to(device)

        self.kernel_mass = self.map_kernel.sum().item()  # KERNEL_MASS_PER_PERSON
        if stage in (4, "joint"):
            self.mse_criterion.scale = self.kernel_mass
            print(
                f"[UNITS] Stage {stage}: overriding mse_criterion.scale -> {self.kernel_mass:.4f} "
                f"(was GAUSSIAN_SCALE={cfg.GAUSSIAN_SCALE}) to match dense_expert/sparse_expert's "
                f"ACTUAL inherited training convention from Stage 2/3."
            )
        self.count_mae_divisor = self.kernel_mass if stage in (4, "joint") else cfg.GAUSSIAN_SCALE
        print(
            f"[UNITS] map_kernel.sum() = {self.kernel_mass:.4f} (person-count-unit divisor "
            f"for Stage 2/3 mass-based diagnostics; GAUSSIAN_SCALE={cfg.GAUSSIAN_SCALE} used only "
            f"for Stage 1, where GaussianMSE_norm's internal rescale already accounts for it)"
        )
        if stage in (2, "joint"):
            print(
                f"[config] Dense background suppression: "
                f"{'OFF (weight 0.0)' if cfg.BG_SUPPRESSION_WEIGHT <= 0 else f'{cfg.BG_SUPPRESSION_MODE}, weight {cfg.BG_SUPPRESSION_WEIGHT}, ring width {cfg.RING_WIDTH}'}"
            )
        if stage in (2, "joint") and cfg.DENSE_OUT_WEIGHT > 0.0:
            print(
                f"[config] Dense outside-mask supervision toward the true density: weight {cfg.DENSE_OUT_WEIGHT} "
                f"(adds weight * mean((dense_pred - map_gt_conv)^2 over pixels outside gt_mask))"
            )
        if stage in (2, "joint"):
            print(
                f"[config] Count-matching L1 term weight: {cfg.COUNT_LOSS_WEIGHT} "
                f"(dense_count_loss + sparse_count_loss, matching total mass per patch)"
            )

        self.label_generator = RoutingLabelGenerator(temperature=1.0, grid_size=10).to(device)
        single_person_peak = cfg.GAUSSIAN_SCALE / (2 * torch.pi * (cfg.GAUSSIAN_SIGMA**2))
        self.sparse_peak_threshold = (
            single_person_peak * cfg.PEAK_THRESHOLD_FRACTION * cfg.SPARSE_PEAK_THRESHOLD_MULTIPLIER
        )
        self.dense_peak_threshold = single_person_peak * cfg.PEAK_THRESHOLD_FRACTION

    def sparse_peaks(self, sparse_map):
        return extract_peaks_via_nms(
            sparse_map, threshold=self.sparse_peak_threshold, kernel_size=cfg.SPARSE_NMS_KERNEL_SIZE
        )

    def patch_loss(self, model, pred_patch, pred_mask, t, stats):
        """Training loss of one patch (t: its PatchTargets; stats: the epoch's TrainStats, Stage 2 logs into it)."""
        if self.stage == 1:
            return self.stage1(pred_patch, t)
        if self.stage == 2:
            return self.stage2(pred_patch, t, stats)
        if self.stage == 3:
            return self.stage3(model, pred_patch, pred_mask, t)[0]
        if self.stage == "joint":
            return self.joint(pred_patch, pred_mask, t, stats)
        return self.stage4(pred_patch, pred_mask, t)

    def stage1(self, pred_patch, t):
        return self.mse_criterion(pred_patch, t.target_patch, self.map_kernel)[0]

    def stage2(self, pred_patch, t, stats, parts=None):
        """Both experts against their regions; adds the logged components and train count errors to `stats`.
        parts (joint run): receives each expert's share of the returned loss ('dense', 'sparse'), as floats."""
        S = cfg.GAUSSIAN_SCALE
        map_gt_conv, target_dots = t.map_gt_conv, t.target_dots
        dense_pred, sparse_pred = pred_patch[:, 0:1, ...], pred_patch[:, 1:2, ...]
        dense_region, sparse_region = stage2_regions(t.gt_mask)

        dense_target_density = map_gt_conv * dense_region
        if self.mass_normalize_dense:
            dense_mse = normalized_mse_loss(
                dense_pred * dense_region, dense_target_density, dense_target_density.sum() / S
            )
        else:
            dense_mse = F.mse_loss(dense_pred * dense_region, dense_target_density)

        masked_sparse_pred = sparse_pred * sparse_region
        sparse_target_density = map_gt_conv * sparse_region
        if self.mass_normalize_sparse:
            sparse_mse = normalized_mse_loss(masked_sparse_pred, sparse_target_density, sparse_target_density.sum() / S)
        else:
            sparse_mse = F.mse_loss(masked_sparse_pred, sparse_target_density)

        dense_count_loss = torch.abs((dense_pred * dense_region).sum() / S - dense_target_density.sum() / S)
        sparse_count_loss = torch.abs(masked_sparse_pred.sum() / S - sparse_target_density.sum() / S)

        if cfg.SPARSE_LOSS == "focal":
            sparse_term = cfg.SPARSE_FOCAL_WEIGHT * penalty_reduced_focal_loss(
                sparse_pred,
                t.target_patch,
                map_gt_conv,
                sparse_region,
                alpha=cfg.FOCAL_HEATMAP_ALPHA,
                beta=cfg.FOCAL_HEATMAP_BETA,
            )
        elif cfg.SPARSE_LOSS == "genloss":
            sparse_target_dots = filter_points_outside_mask(target_dots, 1.0 - sparse_region)
            sparse_term = cfg.SPARSE_GENLOSS_WEIGHT * self.ot_criterion(
                masked_sparse_pred, sparse_target_dots.to(self.device), t.cam_locs
            )
            if cfg.SPARSE_GENLOSS_BG_WEIGHT > 0.0:
                # the OT term has no per-pixel penalty for mass where nobody is: anchor empty sparse-region pixels
                empty_region = sparse_region * (sparse_target_density < cfg.SPARSE_BG_EMPTY_THR).float()
                if cfg.SPARSE_GENLOSS_BG_MODE == "mass":
                    # empty-ground mass in count units; the pixel-mean MSE (~1e-4 vs OT ~5) never mattered
                    sparse_bg_loss = (masked_sparse_pred * empty_region).sum() / S
                else:
                    sparse_bg_loss = F.mse_loss(masked_sparse_pred * empty_region, torch.zeros_like(masked_sparse_pred))
                sparse_term = sparse_term + cfg.SPARSE_GENLOSS_BG_WEIGHT * sparse_bg_loss
            if cfg.SPARSE_GENLOSS_COUNT_WEIGHT > 0.0:
                # the OT cost is computed on the density renormalised to a PMF, so it cannot see the count
                sparse_term = sparse_term + cfg.SPARSE_GENLOSS_COUNT_WEIGHT * sparse_count_loss
        else:
            sparse_term = sparse_mse
        if self.dense_ot_criterion is not None:
            dense_term = self.dense_ot_criterion(
                (dense_pred * dense_region).float() * self.dense_to_count_units,
                patch_points(t.target_patch[0, 0] * dense_region[0, 0]),
            )
        else:
            dense_term = dense_mse
        patch_loss = dense_term + sparse_term + cfg.COUNT_LOSS_WEIGHT * (dense_count_loss + sparse_count_loss)
        cell_loss = 0.0
        if cfg.SPARSE_CELL_LOSS_WEIGHT > 0.0:
            cell_loss = cfg.SPARSE_CELL_LOSS_WEIGHT * relative_cell_loss(
                masked_sparse_pred, sparse_target_density, cfg.SPARSE_CELL_SIZES
            )
            patch_loss = patch_loss + cell_loss

        if cfg.BG_SUPPRESSION_WEIGHT > 0.0:
            if cfg.BG_SUPPRESSION_MODE == "ring":
                ring_k = 2 * cfg.RING_WIDTH + 1
                dilated_gt_mask = F.max_pool2d(dense_region, kernel_size=ring_k, stride=1, padding=cfg.RING_WIDTH)
                suppression_region = dilated_gt_mask - dense_region
            else:
                suppression_region = 1.0 - dense_region
            dense_bg_loss = F.mse_loss(dense_pred * suppression_region, torch.zeros_like(dense_pred))
            patch_loss = patch_loss + cfg.BG_SUPPRESSION_WEIGHT * dense_bg_loss

        if cfg.DENSE_OUT_WEIGHT > 0.0:
            dense_out_loss = ((dense_pred - map_gt_conv) ** 2 * (1.0 - dense_region)).mean()
            patch_loss = patch_loss + cfg.DENSE_OUT_WEIGHT * dense_out_loss

        if parts is not None:
            parts["sparse"] = float(sparse_term + cfg.COUNT_LOSS_WEIGHT * sparse_count_loss + cell_loss)
            parts["dense"] = patch_loss.item() - parts["sparse"]
        stats.batch_dense_loss += dense_mse
        stats.batch_sparse_loss += sparse_mse
        with torch.no_grad():
            train_dense_count = (dense_pred * dense_region).sum().item() / self.kernel_mass
            train_dense_target_mass = dense_target_density.sum().item() / self.kernel_mass
            stats.train_dense_errors.append(abs(train_dense_count - train_dense_target_mass))
            sparse_target_count = len(region_points(target_dots, sparse_region)) if t.gt_count > 0 else 0
            stats.train_sparse_errors.append(abs(self.sparse_peaks(masked_sparse_pred).shape[0] - sparse_target_count))
        return patch_loss

    def stage3(self, model, pred_patch, pred_mask, t):
        """Router loss on the frozen neck features. Returns (loss, safe_pred_mask, performance targets or None,
        per-cell router probability or None)."""
        safe_pred_mask = torch.clamp(pred_mask, min=1e-6, max=1.0 - 1e-6)
        ct = cell_prob = None
        if cfg.STAGE3_TARGET == "performance":
            # per cell: which frozen expert's count is closer to the GT count (no oracle mask)
            with torch.no_grad():
                ct = cell_routing_targets(
                    torch.relu(model.head.dense_expert(pred_patch)),
                    model.head.sparse_map(pred_patch),
                    t.map_gt_conv,
                    cfg.ROUTE_CELL,
                    self.kernel_mass,
                )
            cell_prob = F.avg_pool2d(safe_pred_mask, cfg.ROUTE_CELL, stride=cfg.ROUTE_CELL)
            patch_loss = weighted_routing_bce(cell_prob, ct)
        else:
            patch_loss = cfg.STAGE3_ORACLE_WEIGHT * mask_loss(safe_pred_mask, t.gt_mask)

        if cfg.STAGE3_PERF_WEIGHT > 0.0:
            with torch.no_grad():
                dense_pred = model.head.dense_expert(pred_patch)
                sparse_points = self.sparse_peaks(model.head.sparse_expert(pred_patch))
                soft_target_map = self.label_generator(sparse_points, dense_pred, t.target_patch)
            patch_loss = patch_loss + cfg.STAGE3_PERF_WEIGHT * self.bce_criterion(safe_pred_mask, soft_target_map)
        return patch_loss, safe_pred_mask, ct, cell_prob

    def stage4(self, pred_patch, pred_mask, t):
        """Fused-map loss: (1 - OT_WEIGHT) * MSE + OT_WEIGHT * OT, plus the optional oracle-mask anchor."""
        if cfg.OT_WEIGHT > 0.0:
            ot_spatial_loss = self.ot_criterion(pred_patch, t.target_dots, t.cam_locs)
        else:
            ot_spatial_loss = torch.zeros((), device=self.device)
        count_loss, _ = self.mse_criterion(pred_patch, t.target_patch, self.map_kernel)
        patch_loss = (cfg.OT_WEIGHT * ot_spatial_loss) + ((1.0 - cfg.OT_WEIGHT) * count_loss)
        if cfg.STAGE4_MASK_ANCHOR_WEIGHT > 0.0:
            anchor_loss = mask_loss(torch.clamp(pred_mask, min=1e-6, max=1.0 - 1e-6), t.gt_mask)
            patch_loss = patch_loss + cfg.STAGE4_MASK_ANCHOR_WEIGHT * anchor_loss
        return patch_loss

    def joint(self, pred_patch, pred_mask, t, stats):
        """--joint: Stage 2's expert losses + w_r x the router's loss to the oracle mask + w_f x Stage 4's fused-map
        loss (pred_patch channels: fused, dense, sparse); the unweighted terms go to stats.joint_terms."""
        parts = {}
        experts = self.stage2(pred_patch[:, 1:3, ...], t, stats, parts)
        router = mask_loss(torch.clamp(pred_mask, min=1e-6, max=1.0 - 1e-6), t.gt_mask)
        fused = self.stage4(pred_patch[:, 0:1, ...], pred_mask, t)
        parts["router"], parts["fused"] = router.item(), fused.item()
        for name, value in parts.items():
            stats.joint_terms[name] += value
        stats.joint_patches += 1
        return experts + cfg.JOINT_ROUTER_WEIGHT * router + cfg.JOINT_FUSED_WEIGHT * fused
