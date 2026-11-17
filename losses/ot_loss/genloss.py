# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from .Cost import EDCost, MDCost
from .geomloss import SamplesLoss

EPS = 1e-20


def create_grid(height, width, stride):
    """Generates normalized grid coordinates."""
    coord_x = (torch.arange(0, width, step=stride) + stride) / width
    coord_y = (torch.arange(0, height, step=stride) + stride) / height
    y, x = torch.meshgrid([coord_y.float(), coord_x.float()], indexing="ij")
    return torch.stack((x, y), dim=2).view(-1, 2)


class GeneralizedLoss(_Loss):
    def __init__(self, loss_type="md", reduction="mean", scale=10, device="cuda:0", top_k=None) -> None:
        super().__init__()
        self.reduction = reduction
        self.tau1 = 1
        self.tau2 = 1
        # Keeps only the top_k predicted cells in the OT solve, so the cost matrix (and MDCost's per-camera tensors)
        # stays bounded when the prediction covers most of the patch. None: no cap.
        self.top_k = top_k

        self.cost = MDCost() if loss_type == "md" else EDCost()

        # Sinkhorn hyperparameters
        self.blur = 0.05  # 0.05 rather than 0.01: stable in FP32, as in geomloss's defaults
        self.scaling = 0.75
        self.reach = 0.5
        self.p = 1

        self.uot = SamplesLoss(
            blur=self.blur,
            scaling=self.scaling,
            debias=False,
            backend="tensorized",
            cost=self.cost,
            reach=self.reach,
            p=self.p,
        )

        self.point_loss = nn.L1Loss(reduction=reduction)
        self.pixel_loss = nn.MSELoss(reduction=reduction)
        self.scale = scale
        self.device = device

    def forward(self, density_batch, points_batch, camera_locations):
        """
        Calculates the Generalized Optimal Transport Loss.
        density_batch: [Batch, 1, H, W]
        points_batch: [Batch, 1, num_points, 2]
        camera_locations: [num_cameras, 3]
        """
        batch_size = density_batch.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            # Non-negative, tiny values dropped, normalised to sum to 1
            density = torch.nn.functional.relu(density_batch[i, 0])
            density = torch.where(density > 1e-4, density, torch.zeros_like(density))
            density_sum = density.sum()
            if density_sum > 1e-8:
                density = density / density_sum

            points = points_batch[i]  # [1, num_points, 2]

            # Empty patch or zero prediction: push the raw output toward 0 instead
            if points.numel() == 0 or density.sum() < 1e-8:
                fallback_loss = torch.abs(density_batch[i, 0]).mean() * 3
                total_loss += fallback_loss
                continue

            target_weights = (self.scale * torch.ones(1, points.shape[1], 1)).to(self.device)
            target_coords = points.to(self.device)

            pred_weights, pred_coords = self._density_to_coords(density)

            emd_loss_raw, dual_f, dual_g = self.uot(
                pred_weights, pred_coords, target_weights, target_coords, camera_locations
            )
            cost_matrix = self.cost(pred_coords, target_coords, camera_locations=camera_locations)

            # Prevent FP32 exponential overflow
            exponent = (dual_f.view(1, -1, 1) + dual_g.view(1, 1, -1) - cost_matrix).detach() / (self.blur**self.p)
            exponent = torch.clamp(exponent, max=40.0)
            transport_plan = torch.exp(exponent) * pred_weights * target_weights.view(1, 1, -1)

            entropy = torch.mean((EPS + transport_plan) * torch.log(EPS + transport_plan))
            emd_loss = torch.mean(emd_loss_raw)

            point_loss = self.point_loss(transport_plan.sum(dim=1).view(1, -1, 1), target_weights)
            pixel_loss = self.pixel_loss(transport_plan.sum(dim=2).detach().view(1, -1, 1), pred_weights)

            mass_plan = torch.sum(transport_plan)
            mass_target = torch.sum(target_weights)
            mass_pred = torch.sum(pred_weights)

            actual_tau1 = self._get_tau1(mass_pred, mass_target, mass_plan)
            actual_tau2 = self._get_tau2(mass_pred, mass_target, mass_plan)

            sample_loss = emd_loss + (actual_tau2 * pixel_loss) + (actual_tau1 * point_loss) + (self.blur * entropy)

            # Per-patch normalization: pred mass is always 1 (density renormalized
            # to a PMF above), while target mass is scale * num_points -- so the
            # unbalanced-OT mass-mismatch penalty in emd_loss grows with crowd
            # size, letting a single crowded patch dominate the batch loss (and
            # spike total_loss by orders of magnitude across epochs). Dividing by
            # num_points puts every patch on a comparable "per-person" scale,
            # matching how the rest of the training loop normalizes count terms.
            sample_loss = sample_loss / target_weights.shape[1]

            if sample_loss.item() > 50.0:
                # Diagnostic only, no effect on the loss value: a handful of patches spike to 100-1000x the
                # normal per-patch scale even after the num_points normalization above (observed reproducibly,
                # e.g. scene_79 at a fixed epoch/seed -- see doc/sparse_ot_diagnose.md #8/#12). Dumping the
                # inputs here lets that be diagnosed from the next run's log instead of re-guessing.
                print(
                    f"[genloss WARNING] sample_loss={sample_loss.item():.2f} num_points={target_weights.shape[1]} "
                    f"pred_active_px={pred_weights.shape[1]} mass_pred={mass_pred.item():.4f} "
                    f"mass_target={mass_target.item():.4f} mass_plan={mass_plan.item():.4f} "
                    f"emd_loss={emd_loss.item():.4f} point_loss={point_loss.item():.4f} "
                    f"pixel_loss={pixel_loss.item():.4f} entropy={entropy.item():.4f} "
                    f"tau1={actual_tau1} tau2={actual_tau2} "
                    f"target_xy_range=({target_coords.min().item():.4f},{target_coords.max().item():.4f})"
                )

            total_loss += sample_loss

        return total_loss / batch_size

    def _get_tau1(self, mass_pred, mass_target, mass_plan):
        if mass_pred < mass_target < mass_plan or mass_plan < mass_target < mass_pred:
            return 0
        return self.tau1

    def _get_tau2(self, mass_pred, mass_target, mass_plan):
        if mass_target < mass_pred < mass_plan or mass_plan < mass_pred < mass_target:
            return 0
        return self.tau2

    def _density_to_coords(self, density):
        coords = torch.nonzero(density).float()
        idx = coords[:, 0].long(), coords[:, 1].long()
        density_values = density[idx]

        if self.top_k is not None and density_values.numel() > self.top_k:
            # Keep only the highest-density cells. Cheap (topk is O(N log K)),
            # and safe: Sinkhorn's entropic blur already smooths over several
            # grid cells, so dropping the lowest-mass tail changes very little
            # while bounding the pred-point count fed to the cost matrix.
            top_values, top_idx = torch.topk(density_values, self.top_k)
            coords = coords[top_idx]
            density_values = top_values

        height, width = density.shape[-2], density.shape[-1]
        coords[:, 0] /= float(height)
        coords[:, 1] /= float(width)

        # (row, col) -> (x, y)
        coords = coords[:, [1, 0]]

        return density_values.view(1, -1, 1), coords.unsqueeze(0)


def test_covariance_matrices():
    import matplotlib.pyplot as plt

    cam_pos = [102, 384 - 391]

    x_lin = np.linspace(0, 400, 1000)
    y_lin = np.linspace(0, 400, 1000)
    x_mesh, y_mesh = np.meshgrid(x_lin, y_lin)

    gt_x = np.linspace(20, 320, 5)
    gt_y = np.linspace(20, 384, 5)
    gt_xx, gt_yy = np.meshgrid(gt_x, gt_y)
    gt_coords = np.column_stack((gt_xx.reshape(-1), gt_yy.reshape(-1)))

    direction_y = gt_coords - cam_pos
    rot_col1 = direction_y.copy()

    abs_dy = np.sqrt(np.sum(direction_y**2, axis=-1))
    sigma1 = np.ones_like(abs_dy) * 0.5
    sigma2 = np.ones_like(abs_dy) * 1.5

    inv_var = np.zeros((abs_dy.shape[0], 2, 2))
    inv_var[:, 0, 0] = 1 / sigma1
    inv_var[:, 1, 1] = 1 / sigma2

    rot_col1 = rot_col1 / np.sqrt(rot_col1[:, 0] ** 2 + rot_col1[:, 1] ** 2)[..., None]

    # Apply 90-degree rotation (x, y) -> (-y, x)
    rot_col2 = rot_col1.copy()
    rot_col2[:, [0, 1]] = rot_col2[:, [1, 0]]
    rot_col2[:, 0] = -1 * rot_col2[:, 0]

    rot_col1 = rot_col1[..., None]
    rot_col2 = rot_col2[..., None]

    rot_mat = np.concatenate((rot_col1, rot_col2), axis=-1)
    rot_mat_t = np.transpose(rot_mat, (0, 2, 1))

    inv_cov = rot_mat @ inv_var @ rot_mat_t

    for i in range(25):
        z_vals = (
            inv_cov[i, 0, 0] * (x_mesh - gt_coords[i, 0]) ** 2
            + inv_cov[i, 1, 1] * (y_mesh - gt_coords[i, 1]) ** 2
            + (inv_cov[i, 1, 0] + inv_cov[i, 0, 1]) * (x_mesh - gt_coords[i, 0]) * (y_mesh - gt_coords[i, 1])
            - 100
        )
        plt.contour(x_mesh, y_mesh, z_vals, levels=[0])

    plt.show()
