import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _pixel_centre_grid(h, w):
    """Normalised (x, y) of every cell centre in grid_sample convention (align_corners=False): [h*w, 2]."""
    ys = (torch.arange(h, dtype=torch.float32) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, dtype=torch.float32) + 0.5) / w * 2 - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).view(-1, 2)


class LearnedBEVPositionalEncoding(nn.Module):
    """BEVFormer's LearnedPositionalEncoding: a row and a column embedding, concatenated -> [h*w, embed_dim]."""

    def __init__(self, embed_dim, bev_shape):
        super().__init__()
        self.h, self.w = bev_shape
        self.row_embed = nn.Embedding(self.h, embed_dim // 2)
        self.col_embed = nn.Embedding(self.w, embed_dim - embed_dim // 2)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self):
        rows = self.row_embed.weight[:, None, :].expand(self.h, self.w, -1)
        cols = self.col_embed.weight[None, :, :].expand(self.h, self.w, -1)
        return torch.cat([rows, cols], dim=-1).reshape(self.h * self.w, -1)


class MultiHeadDeformableSelfAttention(nn.Module):
    """Deformable self-attention among BEV queries (BEVFormer's TSA without the previous-frame BEV).

    Each query samples num_points locations per head around its own cell of the CURRENT query map, so from the
    second encoder layer on it mixes neighbours that already carry image evidence. Offsets are predicted in BEV
    pixels and initialised as a ring of directions per head (Deformable DETR init)."""

    offset_units = "px"

    def __init__(self, embed_dim=256, num_heads=8, num_points=4, bev_shape=(200, 200), dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim, self.num_heads, self.num_points = embed_dim, num_heads, num_points
        self.bev_h, self.bev_w = bev_shape
        self.offset_predictor = nn.Linear(embed_dim, num_heads * num_points * 2)
        self.weight_predictor = nn.Linear(embed_dim, num_heads * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("ref_points", _pixel_centre_grid(self.bev_h, self.bev_w), persistent=False)
        self.register_buffer("px_to_norm", torch.tensor([2.0 / self.bev_w, 2.0 / self.bev_h]), persistent=False)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.offset_predictor.weight)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 2).repeat(1, self.num_points, 1)
        grid = grid * torch.arange(1, self.num_points + 1, dtype=torch.float32).view(1, -1, 1)
        with torch.no_grad():
            self.offset_predictor.bias.copy_(grid.view(-1))
        nn.init.zeros_(self.weight_predictor.weight)
        nn.init.zeros_(self.weight_predictor.bias)
        for lin in (self.value_proj, self.output_proj):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, query, query_pos):
        # query, query_pos: [B, L, C]
        B, L, C = query.shape
        H, P, d = self.num_heads, self.num_points, C // self.num_heads
        q = query + query_pos

        value = self.value_proj(query).view(B, self.bev_h, self.bev_w, H, d)
        value = value.permute(0, 3, 4, 1, 2).reshape(B * H, d, self.bev_h, self.bev_w)

        offsets = self.offset_predictor(q).view(B, L, H, P, 2) * self.px_to_norm
        loc = self.ref_points.view(1, L, 1, 1, 2) + offsets  # [B, L, H, P, 2]
        loc = loc.permute(0, 2, 1, 3, 4).reshape(B * H, L, P, 2)
        weights = self.weight_predictor(q).view(B, L, H, P).softmax(-1)
        weights = weights.permute(0, 2, 1, 3).reshape(B * H, L, P)

        sampled = F.grid_sample(
            value, loc.to(value.dtype), mode="bilinear", padding_mode="zeros", align_corners=False
        )  # [B*H, d, L, P]
        out = torch.einsum("bdlp,blp->bdl", sampled, weights.to(sampled.dtype))
        out = out.reshape(B, C, L).transpose(1, 2)
        return self.dropout(self.output_proj(out))


class MultiHeadSpatialCrossAttention(nn.Module):
    """BEVFormer's spatial cross-attention: every BEV query samples points_per_anchor points around each of its
    num_z_anchors projected pillar points in every camera, per head. A camera contributes to a query only if at least
    one of the query's pillar points projects inside that image, and the sum is divided by the number of such cameras
    (BEVFormer's hit-count normalisation). Offsets are predicted in feature-map pixels and start at zero, i.e. at the
    geometric reference points."""

    offset_units = "px"

    def __init__(self, embed_dim=256, num_heads=8, num_cameras=5, num_z_anchors=4, points_per_anchor=1, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim, self.num_heads, self.num_cameras = embed_dim, num_heads, num_cameras
        self.num_z_anchors, self.points_per_anchor = num_z_anchors, points_per_anchor
        self.num_points = num_z_anchors * points_per_anchor
        self.offset_predictor = nn.Linear(embed_dim, num_cameras * num_heads * self.num_points * 2)
        self.weight_predictor = nn.Linear(embed_dim, num_cameras * num_heads * self.num_points)
        self.value_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for lin in (self.offset_predictor, self.weight_predictor):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.xavier_uniform_(self.value_proj.weight.view(self.embed_dim, self.embed_dim))
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, query, query_pos, features_2d, reference_points):
        """
        query, query_pos: [B, L, C]
        features_2d:      [B, N, C, H_img, W_img] (already compressed to embed_dim)
        reference_points: [B, N, L, Z, 2] normalised image coordinates of the pillar points
        """
        B, L, C = query.shape
        _, N, _, H_img, W_img = features_2d.shape
        H, P, d, Z = self.num_heads, self.num_points, C // self.num_heads, self.num_z_anchors
        q = query + query_pos

        value = self.value_proj(features_2d.flatten(0, 1)).view(B, N, H, d, H_img, W_img)

        px_to_norm = query.new_tensor([2.0 / W_img, 2.0 / H_img])
        offsets = self.offset_predictor(q).view(B, L, N, H, P, 2) * px_to_norm
        weights = self.weight_predictor(q).view(B, L, N, H, P).softmax(-1)

        refs = reference_points.unsqueeze(-2).expand(B, N, L, Z, self.points_per_anchor, 2).reshape(B, N, L, P, 2)
        inside = ((refs[..., 0].abs() <= 1) & (refs[..., 1].abs() <= 1)).any(-1)  # [B, N, L] camera sees the cell
        hits = inside.sum(1).clamp(min=1).to(query.dtype)  # [B, L]

        fused = query.new_zeros(B, C, L)
        for n in range(N):
            loc = refs[:, n].unsqueeze(2) + offsets[:, :, n]  # [B, L, H, P, 2]
            loc = loc.permute(0, 2, 1, 3, 4).reshape(B * H, L, P, 2)
            v = value[:, n].reshape(B * H, d, H_img, W_img)
            sampled = F.grid_sample(
                v, loc.to(v.dtype), mode="bilinear", padding_mode="zeros", align_corners=False
            )  # [B*H, d, L, P]
            w = weights[:, :, n].permute(0, 2, 1, 3).reshape(B * H, L, P).to(sampled.dtype)
            out = torch.einsum("bdlp,blp->bdl", sampled, w).reshape(B, C, L)
            fused = fused + out * inside[:, n].unsqueeze(1).to(out.dtype)
        fused = (fused / hits.unsqueeze(1)).transpose(1, 2)
        return self.dropout(self.output_proj(fused))


class BEVEncoderLayer(nn.Module):
    """self_attn -> norm -> cross_attn -> norm -> ffn -> norm (BEVFormer's post-norm operation order)."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_cameras,
        num_z_anchors,
        points_per_anchor,
        sa_points,
        ffn_dim,
        bev_shape,
        dropout,
    ):
        super().__init__()
        self.self_attention = MultiHeadDeformableSelfAttention(embed_dim, num_heads, sa_points, bev_shape, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attention = MultiHeadSpatialCrossAttention(
            embed_dim, num_heads, num_cameras, num_z_anchors, points_per_anchor, dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, query, query_pos, features_2d, reference_points):
        query = self.norm1(query + self.self_attention(query, query_pos))
        query = self.norm2(query + self.cross_attention(query, query_pos, features_2d, reference_points))
        return self.norm3(query + self.ffn(query))


class BEVFormerEncoderNeck(nn.Module):
    """Drop-in replacement for SpatialCrossAttentionNeck with a stack of BEVFormer encoder layers.

    Same inputs and output as the legacy neck. Unlike the legacy neck, from the second layer on the self-attention
    and the cross-attention's offsets/weights are computed from queries that already contain image features."""

    def __init__(
        self,
        in_channels=512,
        embed_dim=256,
        out_channels=64,
        full_bev_shape=(200, 200),
        num_cameras=5,
        num_z_anchors=4,
        num_layers=3,
        num_heads=8,
        points_per_anchor=1,
        sa_points=4,
        ffn_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.full_h, self.full_w = full_bev_shape
        self.num_z_anchors = num_z_anchors
        self.grad_checkpoint = False  # set by the training script; only used while training

        self.cam_compression = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True)
        )
        self.bev_queries = nn.Parameter(torch.randn(self.full_h * self.full_w, embed_dim))
        self.bev_pos = LearnedBEVPositionalEncoding(embed_dim, full_bev_shape)
        self.layers = nn.ModuleList(
            [
                BEVEncoderLayer(
                    embed_dim,
                    num_heads,
                    num_cameras,
                    num_z_anchors,
                    points_per_anchor,
                    sa_points,
                    ffn_dim,
                    full_bev_shape,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.head_processor = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self._desc = (
            f"bevformer: {num_layers} x [deformable SA ({num_heads} heads x {sa_points} pts) -> LN -> "
            f"SCA ({num_heads} heads x {num_z_anchors}x{points_per_anchor} pts/cam, hit-normalised) -> LN -> "
            f"FFN({ffn_dim}) -> LN], dropout {dropout}"
        )

    def describe(self):
        return self._desc

    def forward(self, features_2d, reference_points):
        """
        Args:
            features_2d: [Batch, N, 512, H_img, W_img]
            reference_points: [Batch, N, BEV_H, BEV_W, N_z_anchors, 2]
        """
        B, N, C, H_i, W_i = features_2d.shape
        feats = self.cam_compression(features_2d.reshape(B * N, C, H_i, W_i)).view(B, N, self.embed_dim, H_i, W_i)
        refs = reference_points.reshape(B, N, self.full_h * self.full_w, self.num_z_anchors, 2)

        query = self.bev_queries.unsqueeze(0).expand(B, -1, -1)
        pos = self.bev_pos().unsqueeze(0).expand(B, -1, -1).to(query.dtype)
        for layer in self.layers:
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                query = checkpoint(layer, query, pos, feats, refs, use_reentrant=False)
            else:
                query = layer(query, pos, feats, refs)

        bev = query.transpose(1, 2).reshape(B, self.embed_dim, self.full_h, self.full_w)
        return self.head_processor(bev)  # [B, out_channels, full_h, full_w]
