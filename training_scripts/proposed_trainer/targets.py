"""Ground-truth maps, the oracle dense-zone mask, BEV geometry and peak extraction."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from losses.gaussian_mse import get_map_kernel
from utils.oracle_mask import compute_oracle_mask

from . import config as cfg


def oracle_density_threshold():
    """map_gt_conv value above which a pixel is a dense-zone candidate. NB: the formula assumes a sigma=5-std,
    GAUSSIAN_SCALE-normalised kernel, but get_map_kernel is peak-normalised (one person = 1.0) with covariance 5, so
    the value (1.273 at the defaults) is 1.27 true single-person peaks, not DENSITY_PEAK_MULTIPLIER. Kept as-is so
    the mask matches every existing checkpoint; see utils/oracle_mask.py."""
    return cfg.GAUSSIAN_SCALE / (2 * math.pi * cfg.GAUSSIAN_SIGMA**2) * cfg.DENSITY_PEAK_MULTIPLIER


def oracle_mask_config():
    """Stored with every checkpoint, so evaluation can tell which oracle boundary a model was supervised against."""
    return {
        "density_peak_multiplier": cfg.DENSITY_PEAK_MULTIPLIER,
        "smoothing_kernel": cfg.MASK_SMOOTHING_KERNEL,
        "min_extent": cfg.ORACLE_MIN_EXTENT,
        "min_area": cfg.ORACLE_MIN_AREA,
        "density_window": cfg.ORACLE_DENSITY_WINDOW,
        "min_density_per_m2": cfg.ORACLE_MIN_DENSITY,
        "occupied_only": cfg.ORACLE_OCCUPIED_ONLY,
    }


_MAP_KERNEL_STATS = {}


def _map_kernel_stats():
    """(peak, sum) of the target kernel, computed once."""
    if not _MAP_KERNEL_STATS:
        k = get_map_kernel(sigma=cfg.GAUSSIAN_SIGMA, kernel_size=cfg.GAUSSIAN_KERNEL_SIZE)
        _MAP_KERNEL_STATS.update(peak=float(k.max()), sum=float(k.sum()))
    return _MAP_KERNEL_STATS["peak"], _MAP_KERNEL_STATS["sum"]


def oracle_mask_from_conv(map_gt_conv):
    """THE oracle dense-zone mask for this pipeline. Every multiscene consumer (training stages, evaluation,
    diagnostics) must call this rather than re-deriving the formula. Reads the settings at call time, so CLI
    overrides apply."""
    peak, ksum = _map_kernel_stats()
    window = cfg.ORACLE_DENSITY_WINDOW > 1
    return compute_oracle_mask(
        map_gt_conv,
        cfg.ORACLE_MIN_DENSITY if window else oracle_density_threshold(),
        smoothing_kernel=cfg.MASK_SMOOTHING_KERNEL,
        min_extent=cfg.ORACLE_MIN_EXTENT,
        min_area=cfg.ORACLE_MIN_AREA,
        density_window=cfg.ORACLE_DENSITY_WINDOW,
        kernel_sum=ksum,
        px_per_m=cfg.BEV_PX_PER_M,
        occupied_threshold=cfg.ORACLE_OCCUPIED_PEAKS * peak if cfg.ORACLE_OCCUPIED_ONLY else None,
    )


def extract_camera_locations(camera_paras):
    if camera_paras.dim() == 2:
        camera_paras = camera_paras.unsqueeze(0)
    B, N, _ = camera_paras.shape
    rvecs, tvecs = camera_paras[..., 9:12], camera_paras[..., 12:15]
    theta = torch.norm(rvecs, dim=-1, keepdim=True)
    axes = torch.where(theta > 1e-6, rvecs / theta, torch.zeros_like(rvecs))
    K = torch.zeros(B, N, 3, 3, device=camera_paras.device)
    K[..., 0, 1], K[..., 0, 2] = -axes[..., 2], axes[..., 1]
    K[..., 1, 0], K[..., 1, 2] = axes[..., 2], -axes[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -axes[..., 1], axes[..., 0]
    eye = torch.eye(3, device=camera_paras.device).expand(B, N, 3, 3)
    R = eye + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta).unsqueeze(-1)) * torch.matmul(K, K)
    return -torch.matmul(R.transpose(-2, -1), tvecs.unsqueeze(-1)).squeeze(-1)


def extract_target_coordinates(gt_world_xy, wld_map_paras, hw_offset, bev_shape=cfg.BEV_SHAPE):
    if gt_world_xy.dim() == 3:
        gt_world_xy = gt_world_xy.squeeze(0)
    if wld_map_paras.dim() == 2:
        wld_map_paras = wld_map_paras.squeeze(0)
    if hw_offset.dim() == 2:
        hw_offset = hw_offset.squeeze(0)
    num_people = gt_world_xy.shape[0]
    if num_people == 0:
        return torch.zeros((1, 1, 0, 2))
    r, a, b = wld_map_paras[0].item(), wld_map_paras[1].item(), wld_map_paras[2].item()
    w_min, h_min = wld_map_paras[6].item(), wld_map_paras[7].item()
    h_start, w_start = hw_offset[0].item(), hw_offset[1].item()
    valid_coords = []
    for i in range(num_people):
        col, row = (
            ((gt_world_xy[i, 0].item() - w_min + a) * r) - w_start,
            ((gt_world_xy[i, 1].item() - h_min + b) * r) - h_start,
        )
        if 0 <= row < bev_shape[0] and 0 <= col < bev_shape[1]:
            valid_coords.append([col / bev_shape[1], row / bev_shape[0]])
    if not valid_coords:
        return torch.zeros((1, 1, 0, 2))
    return torch.tensor(valid_coords, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def filter_points_outside_mask(target_dots, gt_mask):
    """Keeps only the GT points (target_dots: [1,1,N,2], normalized x,y) that fall outside gt_mask ([1,1,H,W],
    1 = dense region): restricts the sparse-expert GeneralizedLoss to the sparse region."""
    pts = target_dots[0, 0]
    if pts.shape[0] == 0:
        return target_dots
    h, w = gt_mask.shape[-2], gt_mask.shape[-1]
    cols = (pts[:, 0] * w).long().clamp(0, w - 1)
    rows = (pts[:, 1] * h).long().clamp(0, h - 1)
    keep = gt_mask[0, 0, rows, cols] == 0
    return pts[keep].unsqueeze(0).unsqueeze(0)


def region_points(target_dots, region):
    """GT points (target_dots: [1,1,N,2], normalized x,y) whose pixel lies in region ([1,1,H,W] > 0), as BEV-pixel
    (x, y) coordinates."""
    xy = []
    td = target_dots[0, 0]
    for point_idx in range(td.shape[0]):
        c = min(max(int(td[point_idx, 0].item() * cfg.BEV_SHAPE[1]), 0), cfg.BEV_SHAPE[1] - 1)
        r = min(max(int(td[point_idx, 1].item() * cfg.BEV_SHAPE[0]), 0), cfg.BEV_SHAPE[0] - 1)
        if region[0, 0, r, c] > 0:
            xy.append([td[point_idx, 0].item() * cfg.BEV_SHAPE[1], td[point_idx, 1].item() * cfg.BEV_SHAPE[0]])
    return xy


def project_cameras_to_patch(cam_locations, wld_map_paras, hw_offset, bev_shape=cfg.BEV_SHAPE):
    cam_x, cam_y, cam_z = cam_locations[:, 0], cam_locations[:, 1], cam_locations[:, 2]
    if wld_map_paras.dim() == 2:
        wld_map_paras = wld_map_paras.squeeze(0)
    if hw_offset.dim() == 2:
        hw_offset = hw_offset.squeeze(0)
    r, a, b = wld_map_paras[0].item(), wld_map_paras[1].item(), wld_map_paras[2].item()
    w_min, h_min = wld_map_paras[6].item(), wld_map_paras[7].item()
    h_start, w_start = hw_offset[0].item(), hw_offset[1].item()
    norm_cam_x = (((cam_x - w_min + a) * r) - w_start) / bev_shape[1]
    norm_cam_y = (((cam_y - h_min + b) * r) - h_start) / bev_shape[0]
    return torch.stack([norm_cam_x, norm_cam_y, cam_z], dim=1)


def create_ground_truth_heatmap(gt_world_xy, wld_map_paras, hw_offset, bev_shape=cfg.BEV_SHAPE):
    heatmap = np.zeros(bev_shape, dtype=np.float32)
    if gt_world_xy.dim() == 3:
        gt_world_xy = gt_world_xy.squeeze(0)
    if wld_map_paras.dim() == 2:
        wld_map_paras = wld_map_paras.squeeze(0)
    if hw_offset.dim() == 2:
        hw_offset = hw_offset.squeeze(0)
    if gt_world_xy.shape[0] == 0:
        return torch.zeros((1, 1, bev_shape[0], bev_shape[1]))
    r, a, b = wld_map_paras[0].item(), wld_map_paras[1].item(), wld_map_paras[2].item()
    w_min, h_min = wld_map_paras[6].item(), wld_map_paras[7].item()
    # int(): a float zero offset (full-scene evaluation) would otherwise turn row/col into floats
    h_start, w_start = int(hw_offset[0].item()), int(hw_offset[1].item())
    for i in range(gt_world_xy.shape[0]):
        col = int(round((gt_world_xy[i, 0].item() - w_min + a) * r)) - w_start
        row = int(round((gt_world_xy[i, 1].item() - h_min + b) * r)) - h_start
        if 0 <= row < bev_shape[0] and 0 <= col < bev_shape[1]:
            heatmap[row, col] += 1.0
    return torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)


class PatchTargets:
    """Ground truth of one training / validation patch, each map built on first use and kept."""

    def __init__(self, gt_world_xy, wld_paras, hw_offset, cam_locations, map_kernel, device):
        self._args = (gt_world_xy, wld_paras, hw_offset)
        self._cam_locations, self._kernel, self._device = cam_locations, map_kernel, device
        self._cache = {}

    def _get(self, name, build):
        if name not in self._cache:
            self._cache[name] = build()
        return self._cache[name]

    @property
    def target_dots(self):
        """[1, 1, N, 2] normalised (x, y) of the people in the patch."""
        return self._get(
            "dots", lambda: extract_target_coordinates(*self._args, bev_shape=cfg.BEV_SHAPE).to(self._device)
        )

    @property
    def gt_count(self):
        return self.target_dots.shape[2]

    @property
    def cam_locs(self):
        return self._get(
            "cams",
            lambda: project_cameras_to_patch(
                self._cam_locations, self._args[1], self._args[2], bev_shape=cfg.BEV_SHAPE
            ).to(self._device),
        )

    @property
    def target_patch(self):
        """Person-pixel count map [1, 1, H, W]."""
        return self._get(
            "patch", lambda: create_ground_truth_heatmap(*self._args, bev_shape=cfg.BEV_SHAPE).to(self._device)
        )

    @property
    def map_gt_conv(self):
        """Target density: the count map blurred with the peak-normalised map kernel."""
        return self._get("conv", lambda: F.conv2d(self.target_patch, self._kernel, padding=self._kernel.shape[-1] // 2))

    @property
    def gt_mask(self):
        return self._get("mask", lambda: oracle_mask_from_conv(self.map_gt_conv))


def extract_peaks_via_nms(tensor, threshold=0.01, kernel_size=3):
    """Returns NORMALIZED [0,1] (x, y) peak coordinates -- NOTE the (x, y) convention differs from
    evaluation/metrics/localization.py's (row, col)."""
    pad = kernel_size // 2
    local_max = F.max_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=pad)
    keep_mask = (local_max == tensor) & (tensor > threshold)

    coords = keep_mask.squeeze(0).squeeze(0).nonzero(as_tuple=False)
    if coords.shape[0] == 0:
        return torch.zeros((0, 2), device=tensor.device)

    norm_coords = torch.empty_like(coords, dtype=torch.float32)
    norm_coords[:, 0] = coords[:, 1].float() / tensor.shape[3]
    norm_coords[:, 1] = coords[:, 0].float() / tensor.shape[2]
    return norm_coords


def count_matches_np(pred_xy, gt_xy, radius):
    """One-to-one matches within `radius` px (Hungarian if scipy is available, else greedy)."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return 0
    d = np.linalg.norm(pred_xy[:, None, :] - gt_xy[None, :, :], axis=2)
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(d)
        return int((d[r, c] <= radius).sum())
    except ImportError:
        used_p, used_g, tp = set(), set(), 0
        for flat in np.argsort(d, axis=None):
            i, j = divmod(int(flat), d.shape[1])
            if d[i, j] > radius:
                break
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            tp += 1
        return tp


def cell_routing_targets(dense_map, sparse_map, map_gt_conv, cell, kernel_mass):
    """Per-cell performance-routing targets. All maps [B, 1, H, W] in map_gt_conv units (one person integrates to
    kernel_mass), i.e. the units the fused prediction mask * dense + (1 - mask) * sparse is counted in. Returns
    per-cell [B, 1, H / cell, W / cell] tensors: dense / sparse / gt counts (persons), err_dense, err_sparse,
    label (1 = dense expert closer to the GT count) and weight (= |err_dense - err_sparse|, the people a wrong
    choice costs; ~0 where both are equally good, e.g. empty ground)."""
    H, W = map_gt_conv.shape[-2:]
    if H % cell or W % cell:
        raise ValueError(f"ROUTE_CELL={cell} must divide the patch size {H}x{W}")

    def to_cells(x):
        return F.avg_pool2d(x, cell, stride=cell) * (cell * cell) / kernel_mass

    d, s, g = to_cells(dense_map), to_cells(sparse_map), to_cells(map_gt_conv)
    err_d, err_s = (d - g).abs(), (s - g).abs()
    return {
        "dense": d,
        "sparse": s,
        "gt": g,
        "err_dense": err_d,
        "err_sparse": err_s,
        "label": (err_d < err_s).float(),
        "weight": (err_d - err_s).abs(),
    }
