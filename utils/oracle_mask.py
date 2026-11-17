"""
Oracle dense-zone mask: the single definition of which BEV pixels count as "dense" (routed to dense_expert).

Previously copied inline in ~15 places (training stages, evaluation, diagnostics). Every multiscene consumer
should now go through train_proposed_multiscene.oracle_mask_from_conv, which binds this function to that
script's constants, so training and evaluation can never silently disagree on the boundary.

The mask has two independent axes:

1. Density (threshold): pixel is a candidate if the Gaussian-blurred GT density map_gt_conv exceeds
   `threshold`. get_map_kernel(sigma=5) uses 5 as the COVARIANCE (std ~2.24 px = 0.45 m at 5 px/m) and normalises
   the peak to 1, so one isolated person peaks at 1.0 and a uniform crowd of D persons/px^2 gives D * kernel.sum()
   (~31.4). The pipeline's threshold, 2 x GAUSSIAN_SCALE / (2 pi GAUSSIAN_SIGMA^2) = 1.273, is therefore 1.27
   single-person peaks: two people within ~0.85 m already exceed it, and for a uniform crowd it corresponds to
   ~1.0 persons/m^2.
   Alternative (density_window > 1): average the density over a window x window neighbourhood and threshold it in
   persons/m^2 -- "is this a crowd at the dense expert's scale?" rather than "is someone's neighbour close?". The
   pointwise criterion makes ~half of all dense components bare pairs (analyse_oracle_mask_extent.py).
2. Spatial extent (min_extent / min_area, both off by default): a candidate region is only kept if it is large
   enough for the dense expert to operate on.
   - min_extent (morphological opening with a min_extent x min_extent square): keeps exactly the pixels covered
     by some k x k window lying fully inside the region -- "the expert's window fits inside the crowd". Deletes
     thin shapes (queues) regardless of their total size.
   - min_area (connected-component area filter, pixels): keeps components with at least min_area pixels,
     regardless of shape.

The smoothing step (max-pool then avg-pool > 0.5 with smoothing_kernel) is the original cosmetic cleanup;
it only moves boundaries by ~smoothing_kernel // 2 pixels and imposes no minimum size.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _check_odd(name, k):
    if k < 1 or k % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer (symmetric padding), got {k}")


def morphological_opening(mask, k):
    """Binary opening (erosion then dilation) with a k x k square on a [B, 1, H, W] {0, 1} float mask.
    max_pool2d pads with -inf, so the patch border does not erode the mask: a crowd cut by the patch edge is
    treated as continuing past it, not as ending there."""
    _check_odd("min_extent", k)
    if k == 1:
        return mask
    eroded = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=k // 2)
    return F.max_pool2d(eroded, kernel_size=k, stride=1, padding=k // 2)


def remove_small_components(mask, min_area):
    """Drops 8-connected components with fewer than min_area pixels from a [B, 1, H, W] {0, 1} float mask.
    Runs on CPU (cv2); cheap at 200 x 200."""
    import cv2

    if min_area <= 1:
        return mask
    out = mask.clone()
    m_np = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
    for b in range(m_np.shape[0]):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m_np[b, 0], connectivity=8)
        small = np.where(stats[1:, cv2.CC_STAT_AREA] < min_area)[0] + 1
        if len(small):
            drop = torch.from_numpy(np.isin(labels, small)).to(mask.device)
            out[b, 0][drop] = 0.0
    return out


def window_density(map_gt_conv, window, kernel_sum, px_per_m):
    """Mean people density (persons/m^2) over the window x window neighbourhood of every pixel. Box-averaging the
    blurred map and dividing by the kernel's mass equals box-averaging the dot map, up to the kernel's soft edge.
    Only in-map pixels are averaged (count_include_pad=False), so a crowd at the patch/scene border is not diluted
    by pixels that do not exist."""
    _check_odd("density_window", window)
    mean_per_px = (
        F.avg_pool2d(map_gt_conv, kernel_size=window, stride=1, padding=window // 2, count_include_pad=False)
        / kernel_sum
    )
    return mean_per_px * px_per_m**2


def compute_oracle_mask(
    map_gt_conv,
    threshold,
    smoothing_kernel=3,
    min_extent=1,
    min_area=0,
    density_window=1,
    kernel_sum=None,
    px_per_m=None,
    occupied_threshold=None,
):
    """map_gt_conv: [B, 1, H, W] Gaussian-blurred GT density.
    density_window == 1: threshold is in map_gt_conv units (pointwise density, the original criterion).
    density_window > 1:  threshold is in persons/m^2, compared against window_density (needs kernel_sum, px_per_m).
                         This measures crowding at the dense expert's own spatial scale instead of the 0.45 m target
                         kernel's, which only measures how close each person's nearest neighbour is.
    occupied_threshold: if set, additionally require map_gt_conv > occupied_threshold, i.e. drop the empty margin a
                        window criterion adds around a crowd (pixels far from any person).
    Returns a [B, 1, H, W] {0, 1} float mask. With all optional arguments at their defaults this reproduces the
    original inline formula exactly."""
    _check_odd("smoothing_kernel", smoothing_kernel)
    if density_window > 1:
        if kernel_sum is None or px_per_m is None:
            raise ValueError("density_window > 1 needs kernel_sum and px_per_m")
        raw = window_density(map_gt_conv, density_window, kernel_sum, px_per_m) > threshold
    else:
        raw = map_gt_conv > threshold
    if occupied_threshold is not None:
        raw = raw & (map_gt_conv > occupied_threshold)
    raw = raw.float()
    pad = smoothing_kernel // 2
    dilated = F.max_pool2d(raw, kernel_size=smoothing_kernel, stride=1, padding=pad)
    mask = (F.avg_pool2d(dilated, kernel_size=smoothing_kernel, stride=1, padding=pad) > 0.5).float()
    mask = morphological_opening(mask, min_extent)
    return remove_small_components(mask, min_area)
