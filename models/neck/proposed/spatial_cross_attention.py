import torch
import torch.nn as nn
import torch.nn.functional as F
from .deformable_attention import DeformableCrossAttention


class DeformableSelfAttention(nn.Module):
    def __init__(self, embed_dim=256, num_points=4, bev_shape=(200, 200)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_points = num_points
        self.bev_h, self.bev_w = bev_shape

        self.offset_predictor = nn.Linear(embed_dim, num_points * 2)
        self.weight_predictor = nn.Linear(embed_dim, num_points)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Each query's own normalized (x, y) position -- this IS its reference point.
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, self.bev_h), torch.linspace(-1, 1, self.bev_w), indexing="ij")
        self.register_buffer("self_ref_points", torch.stack([xs, ys], dim=-1).view(-1, 2))  # [L, 2]

    def forward(self, bev_queries):
        # bev_queries: [B, L, C]
        B, L, C = bev_queries.shape

        offsets = self.offset_predictor(bev_queries).view(B, L, self.num_points, 2) * 0.1
        weights = F.softmax(self.weight_predictor(bev_queries), dim=-1)  # [B, L, K]

        sampling_locations = self.self_ref_points.view(1, L, 1, 2) + offsets  # [B, L, K, 2]

        # Reshape queries into a spatial map so grid_sample can pull neighbors from it
        query_map = bev_queries.transpose(1, 2).view(B, C, self.bev_h, self.bev_w)

        sampled = F.grid_sample(
            query_map, sampling_locations, mode="bilinear", padding_mode="zeros", align_corners=False
        )  # [B, C, L, K]
        sampled = sampled.permute(0, 2, 3, 1)  # [B, L, K, C]

        fused = (sampled * weights.unsqueeze(-1)).sum(dim=2)  # [B, L, C]
        return self.output_proj(fused) + bev_queries  # residual


class SpatialCrossAttentionNeck(nn.Module):
    def __init__(
        self, in_channels=512, embed_dim=256, out_channels=64, full_bev_shape=(200, 200), num_cameras=5, num_z_anchors=4
    ):
        """
        The modular Neck for Hybrid Multi-View Crowd Counting.
        num_z_anchors: The number of vertical points sampled per BEV pillar (e.g., 4 points from Z=0.0 to Z=1.8m)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.full_h, self.full_w = full_bev_shape

        self.cam_compression = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True)
        )

        # One learnable query per BEV cell
        self.bev_queries = nn.Parameter(torch.randn(self.full_h * self.full_w, embed_dim))

        self.cross_attention = DeformableCrossAttention(
            embed_dim=embed_dim, num_cameras=num_cameras, num_points=num_z_anchors
        )

        self.self_attention = DeformableSelfAttention(embed_dim=embed_dim, num_points=num_z_anchors)

        # Smooths the attention output and reduces it to out_channels
        self.head_processor = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, features_2d, reference_points):
        """
        Args:
            features_2d: [Batch, N, 512, H_img, W_img]
            reference_points: [Batch, N, BEV_H * BEV_W, N_z_anchors, 2]
        """
        B, N, C, H_i, W_i = features_2d.shape

        features_flat = self.cam_compression(features_2d.view(B * N, C, H_i, W_i))
        compressed_features = features_flat.view(B, N, self.embed_dim, H_i, W_i)

        # Self-attention over the BEV queries first, then cross-attention into the cameras (sa_ca)
        queries = self.bev_queries.unsqueeze(0).expand(B, -1, -1)  # [B, BEV_H*BEV_W, embed_dim]
        queries = self.self_attention(queries)
        attn_output = self.cross_attention(
            bev_queries=queries, features_2d=compressed_features, reference_points=reference_points
        )

        bev_spatial = attn_output.transpose(1, 2).view(B, self.embed_dim, self.full_h, self.full_w)

        fused_bev = self.head_processor(bev_spatial)

        return fused_bev  # Output Shape: [B, 64, full_h, full_w]
