import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableCrossAttention(nn.Module):
    def __init__(self, embed_dim=256, num_cameras=5, num_points=4):
        """
        num_points (K): How many pixels to sample per camera for a single BEV query.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.num_points = num_points

        # Per query: an (dx, dy) offset and a weight for each of the K points of each camera
        self.offset_predictor = nn.Linear(embed_dim, num_cameras * num_points * 2)
        self.weight_predictor = nn.Linear(embed_dim, num_cameras * num_points)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # camera ablation: the cameras' samples are summed, so trained on 5 and run on N the sum grows by N/5;
        # camera_scale = 5/N restores the trained magnitude (utils/camera_slots.py)
        self.camera_scale = 1.0

    def forward(self, bev_queries, features_2d, reference_points):
        """
        bev_queries: [B, BEV_H * BEV_W, C]
        features_2d: [B, N, C, H_img, W_img]
        reference_points: [N, BEV_H, BEV_W, 2]
        """
        B, N, C, H_img, W_img = features_2d.shape
        _, L, _ = bev_queries.shape  # L = BEV_H * BEV_W

        # Offsets scaled by 0.1 to keep the sampling close to the reference points
        offsets = self.offset_predictor(bev_queries).view(B, L, N, self.num_points, 2) * 0.1

        attention_weights = self.weight_predictor(bev_queries).view(B, L, N, self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1)  # softmax over the K points of each camera

        # reference_points: one projected point per z-anchor (K = num_points) -> [B, L, N, K, 2]
        ref_points = reference_points.reshape(B, N, L, self.num_points, 2).transpose(1, 2)
        sampling_locations = ref_points + offsets  # [B, L, N, K, 2]

        fused_features = torch.zeros_like(bev_queries)

        for cam_idx in range(N):
            cam_features = features_2d[:, cam_idx]  # [B, C, H_img, W_img]
            # grid_sample treats L as height and K as width
            cam_samples = sampling_locations[:, :, cam_idx, :, :]  # [B, L, K, 2]
            sampled = F.grid_sample(
                cam_features,
                cam_samples,
                mode="bilinear",
                padding_mode="zeros",  # samples outside the image contribute 0
                align_corners=False,
            ).permute(0, 2, 3, 1)  # [B, L, K, C]
            cam_weights = attention_weights[:, :, cam_idx, :].unsqueeze(-1)  # [B, L, K, 1]
            fused_features += (sampled * cam_weights).sum(dim=2)  # weighted sum over K, summed over cameras

        if self.camera_scale != 1.0:
            fused_features = fused_features * self.camera_scale
        return self.output_proj(fused_features)
