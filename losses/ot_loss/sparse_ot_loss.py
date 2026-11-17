"""
sparse_ot_loss.py

Isotropic OT loss between a predicted density map and ground-truth points, in [0,1]-normalised BEV coordinates
(written for the first sparse-expert OT tests; the trainers use genloss.py instead). With equal sigma axes and no
distance-dependent tightening, MDCost's per-point Mahalanobis covariance is a constant multiple of the identity, i.e. a
rescaled Euclidean distance that `blur` already covers, so geomloss's built-in p=1 cost is used as is and the camera
locations are not needed.

  - debias=True: the divergence stays >= 0 (no negative mass-flooding runaway).
  - backend="online": KeOps lazy reductions, no O(N^2) cost matrix.
  - p=1: matches blur's epsilon = blur**p scale (p=2 gave a near-zero-temperature Sinkhorn).
  - Source and target mass both sum to 1 (shape only); the source denominator is detached, so the OT gradient is not
    crushed by the total magnitude. A separate L1 term on the raw predicted mass handles the count.

The input must already be non-negative. Prefer an activation without a hard zero (softplus, not relu): ReLU units
that this loss pushes to zero get no gradient back.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# `geomloss` is the vendored package next to this file (losses/ot_loss/geomloss/), not the pip package.
try:
    from .geomloss import SamplesLoss
except ImportError as e:
    raise ImportError(
        "Could not import the local geomloss package -- confirm this file "
        "lives at losses/ot_loss/sparse_ot_loss.py, alongside genloss.py "
        "and the geomloss/ folder, not somewhere else."
    ) from e


def build_bev_coord_grid(bev_shape, device):
    """[0,1]-normalized (x, y) coordinates of every BEV grid cell (same convention as create_grid() in genloss.py)."""
    h, w = bev_shape
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 1) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 1) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # [H*W, 2]


class SparseOTLoss(nn.Module):
    """Sinkhorn divergence (geomloss, Euclidean p=1 cost) on the normalised mass + count_weight * L1 on the count."""

    def __init__(self, blur=0.05, scaling=0.75, reach=None, count_weight=0.01, backend="online"):
        super().__init__()
        self.blur = blur
        self.scaling = scaling
        self.reach = reach
        self.count_weight = count_weight
        self.backend = backend
        self._cached_coords = None
        self._cached_shape = None

        # cost=None: geomloss's built-in p=1 cost ("Norm2(X-Y)") on both backends; `blur` is the only scale knob.
        self.ot_loss_fn = SamplesLoss(
            loss="sinkhorn",
            p=1,
            blur=self.blur,
            scaling=self.scaling,
            debias=True,
            backend=self.backend,
            reach=self.reach,
            cost=None,
        )

    def _get_coords(self, bev_shape, device):
        if self._cached_shape != bev_shape or self._cached_coords is None or self._cached_coords.device != device:
            self._cached_coords = build_bev_coord_grid(bev_shape, device)
            self._cached_shape = bev_shape
        return self._cached_coords

    def forward(self, pred_density_map, gt_points, camera_locations_2d=None):
        """
        Args:
            pred_density_map: [1, 1, H, W], non-negative (see the module docstring).
            gt_points: [M, 2] GT (x, y) in [0,1]-normalized coordinates.
            camera_locations_2d: unused (the cost has no camera geometry); kept for call-site compatibility.

        Returns:
            total_loss, logs: "ot", "count_l1", "pred_count", "gt_count" (floats) and "ot_tensor" / "count_tensor"
            (the weighted terms, for per-term gradient norms).
        """
        device = pred_density_map.device
        _, _, H, W = pred_density_map.shape
        coords = self._get_coords((H, W), device)  # [H*W, 2]

        raw_mass = pred_density_map.reshape(-1)
        pred_count = raw_mass.sum()

        num_gt = gt_points.shape[0]
        if num_gt == 0:
            count_loss = pred_count.abs()
            return self.count_weight * count_loss, {
                "ot": 0.0,
                "count_l1": count_loss.item(),
                "pred_count": pred_count.item(),
                "gt_count": 0,
                "ot_tensor": torch.zeros((), device=device),
                "count_tensor": self.count_weight * count_loss,
                "note": "empty_gt_fallback",
            }

        # Detached denominator: the OT gradient sees the shape, not the total magnitude (see the module docstring).
        weights_raw = raw_mass + 1e-8
        weights = (weights_raw / weights_raw.sum().detach()).unsqueeze(0)  # [1, H*W], sums to ~1
        x = coords.unsqueeze(0)  # [1, H*W, 2]

        target_weights = torch.full((1, num_gt), 1.0 / num_gt, device=device)  # sums to 1
        y = gt_points.unsqueeze(0)  # [1, M, 2]

        # The vendored geomloss takes the camera locations as a 5th positional argument (process_args accepts 2, 5 or 6
        # arguments, not 4); this cost does not use them, and the patched sinkhorn_online ignores the slot.
        ot_term = self.ot_loss_fn(weights, x, target_weights, y, None)

        count_term = F.l1_loss(pred_count, torch.tensor(float(num_gt), device=device))
        total = ot_term + self.count_weight * count_term

        return total, {
            "ot": ot_term.item(),
            "count_l1": count_term.item(),
            "pred_count": pred_count.item(),
            "gt_count": num_gt,
            "ot_tensor": ot_term,
            "count_tensor": self.count_weight * count_term,
        }
