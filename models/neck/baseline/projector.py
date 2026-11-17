import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialProjector(nn.Module):
    def __init__(self, bev_shape=(200, 200), original_image_shape=(1080, 1920)):
        """
        Projects 2D image features onto a 3D Bird's Eye View ground plane.
        """
        super().__init__()
        self.default_bev_height, self.default_bev_width = bev_shape
        self.img_height, self.img_width = original_image_shape

    def _get_dynamic_grid(self, H, W, device):
        """Generates the indices grid dynamically based on target H and W."""
        y_linspace = torch.linspace(0, H - 1, H, device=device)
        x_linspace = torch.linspace(0, W - 1, W, device=device)
        y_coords, x_coords = torch.meshgrid(y_linspace, x_linspace, indexing="ij")

        x_flat = x_coords.reshape(-1)
        y_flat = y_coords.reshape(-1)
        ones = torch.ones_like(x_flat)
        return torch.stack([x_flat, y_flat, ones], dim=0)

    def forward(self, features, camera_paras, wld_map_paras, hw_offset, target_shape=None):
        B, N, C, H_feat, W_feat = features.shape

        # 1. Determine spatial bounds dynamically
        H, W = target_shape if target_shape is not None else (self.default_bev_height, self.default_bev_width)

        features_flat = features.view(B * N, C, H_feat, W_feat)
        cam_paras_flat = camera_paras.view(B * N, 15).float()
        # dataset appends d_delta as an 11th entry; only the first 10 are used here
        wld_paras_expanded = wld_map_paras.unsqueeze(1).repeat(1, N, 1).view(B * N, wld_map_paras.shape[-1]).float()

        # 2. Use dynamic grid
        base_grid = self._get_dynamic_grid(H, W, features.device)
        grid = base_grid.unsqueeze(0).repeat(B * N, 1, 1).float()

        h_start = hw_offset[:, 0].unsqueeze(1).repeat(1, N).view(B * N, 1, 1).float()
        w_start = hw_offset[:, 1].unsqueeze(1).repeat(1, N).view(B * N, 1, 1).float()

        r = wld_paras_expanded[:, 0:1].unsqueeze(-1)
        a = wld_paras_expanded[:, 1:2].unsqueeze(-1)
        b = wld_paras_expanded[:, 2:3].unsqueeze(-1)
        d_mean = wld_paras_expanded[:, 5:6].unsqueeze(-1)
        w_min = wld_paras_expanded[:, 6:7].unsqueeze(-1)
        h_min = wld_paras_expanded[:, 7:8].unsqueeze(-1)

        Xw = ((grid[:, 0:1, :] + w_start) * 1.0) / r + w_min - a
        Yw = ((grid[:, 1:2, :] + h_start) * 1.0) / r + h_min - b
        Zw = grid[:, 2:3, :] * d_mean

        world_coords = torch.cat([Xw, Yw, Zw], dim=1)  # [B*N, 3, H*W]

        rvec = cam_paras_flat[:, 9:12]
        theta = torch.norm(rvec, dim=1, keepdim=True)
        rvec_norm = rvec / (theta + 1e-8)
        rx, ry, rz = rvec_norm[:, 0], rvec_norm[:, 1], rvec_norm[:, 2]

        zero = torch.zeros_like(rx)
        row1 = torch.stack([zero, -rz, ry], dim=1)
        row2 = torch.stack([rz, zero, -rx], dim=1)
        row3 = torch.stack([-ry, rx, zero], dim=1)
        skew = torch.stack([row1, row2, row3], dim=1)

        r1 = torch.cos(theta).unsqueeze(2) * torch.eye(3, device=features.device).unsqueeze(0)
        rvec_unsq = rvec_norm.unsqueeze(2)
        r2 = (1.0 - torch.cos(theta).unsqueeze(2)) * torch.bmm(rvec_unsq, rvec_unsq.transpose(1, 2))
        r3 = torch.sin(theta).unsqueeze(2) * skew
        R = r1 + r2 + r3

        tvec = cam_paras_flat[:, 12:15].unsqueeze(2)
        cam_coords = torch.bmm(R, world_coords) + tvec

        xc, yc, zc = cam_coords[:, 0, :], cam_coords[:, 1, :], cam_coords[:, 2, :]

        Xu0 = xc / (zc + 1.0 / 19200.0)
        Yu0 = yc / (zc + 1.0 / 10800.0)

        k1, k2, p1, p2, k3 = cam_paras_flat[:, 4:9].unbind(dim=1)
        k1, k2, p1, p2, k3 = k1.unsqueeze(1), k2.unsqueeze(1), p1.unsqueeze(1), p2.unsqueeze(1), k3.unsqueeze(1)

        r_sq = Xu0**2 + Yu0**2
        beta = 1.0 + k1 * r_sq + k2 * r_sq**2 + k3 * r_sq**3
        Xu1 = Xu0 * beta + 2 * p1 * Xu0 * Yu0 + p2 * (r_sq + 2 * Xu0**2)
        Yu1 = Yu0 * beta + 2 * p2 * Xu0 * Yu0 + p1 * (r_sq + 2 * Yu0**2)

        fx, fy, u, v = cam_paras_flat[:, 0:4].unbind(dim=1)
        fx, fy, u, v = fx.unsqueeze(1), fy.unsqueeze(1), u.unsqueeze(1), v.unsqueeze(1)

        Xu2 = (fx * Xu1 + u) * self.img_width
        Yu2 = (fy * Yu1 + v) * self.img_height

        grid_x = (Xu2 * 2.0) / self.img_width - 1.0
        grid_y = (Yu2 * 2.0) / self.img_height - 1.0
        grid_x = torch.clamp(grid_x, -10.0, 10.0)
        grid_y = torch.clamp(grid_y, -10.0, 10.0)

        # 3. Use dynamic H and W for the sampling grid
        sampling_grid = torch.stack([grid_x, grid_y], dim=-1).view(B * N, H, W, 2)
        sampling_grid = sampling_grid.to(features.dtype)

        projected = F.grid_sample(
            features_flat, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        return projected.view(B, N, C, H, W)
