import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MultiViewHeightAttention(nn.Module):
    """Key-based fusion over cameras x height anchors for every BEV cell.

    Every BEV cell has num_cameras * num_z_anchors candidate samples ("tokens"): the camera feature at the cell's
    pillar point at each anchor height (the cached reference points, no learned offsets). One softmax over all valid
    tokens of the cell decides where the cell's feature comes from; per head the logit of token (camera n, anchor z) is

        q . k_nz / sqrt(d)                 content: k_nz is the SAMPLED image feature, so the weights depend on what
                                           each camera actually shows at that 3D point
      + MLP(geometry, consistency)_nz      x, y in the image, pixels per metre along the pillar (distance), the
                                           cosine similarity of k_nz to the other cameras' keys at the same 3D point
                                           (multi-view photo-consistency: high where a head really is, low for a
                                           camera whose view of the point is blocked), number of cameras that see it
      + W(q)_z + b_z                       query-dependent and learned height prior

    Unlike the deformable cross-attentions (weights = softmax(Linear(query)), the same for every image unless the
    query changes, see doc/neck_ablation.md section 7), these weights cannot be a fixed blend: two of the three logit
    terms are functions of the sampled features. The query is built from the tokens themselves (masked max and mean
    over the cell's tokens), optionally plus a BEV feature map passed in as `base` (e.g. the homography BEV).

    forward() returns the fused features [B, L, embed_dim] and keeps, if keep_aux is set, the head-averaged weight
    marginals in self.aux: 'height' [B, L, Z] (sums to 1 where a camera sees the cell, else 0) and 'camera' [B, L, N].
    """

    def __init__(
        self,
        embed_dim=256,
        key_dim=64,
        value_dim=128,
        num_heads=8,
        num_cameras=5,
        num_z_anchors=8,
        base_dim=None,
        geom_hidden=32,
        grad_checkpoint=True,
        zero_init_output=False,
    ):
        super().__init__()
        assert key_dim % num_heads == 0 and value_dim % num_heads == 0
        self.embed_dim, self.key_dim, self.value_dim = embed_dim, key_dim, value_dim
        self.num_heads, self.num_cameras, self.num_z_anchors = num_heads, num_cameras, num_z_anchors
        self.grad_checkpoint = grad_checkpoint
        self.keep_aux = False
        self.aux = None

        self.key_map = nn.Conv2d(embed_dim, key_dim, kernel_size=1)  # applied before sampling (commutes with it)
        self.value_map = nn.Conv2d(embed_dim, value_dim, kernel_size=1)
        self.query_proj = nn.Linear(2 * key_dim, key_dim)
        self.base_proj = nn.Linear(base_dim, key_dim) if base_dim else None
        self.query_height = nn.Linear(key_dim, num_heads * num_z_anchors)
        self.height_bias = nn.Parameter(torch.zeros(num_heads, num_z_anchors))
        self.geom_mlp = nn.Sequential(
            nn.Linear(6, geom_hidden), nn.ReLU(inplace=True), nn.Linear(geom_hidden, num_heads)
        )
        self.output_proj = nn.Linear(value_dim, embed_dim)
        self._reset_parameters(zero_init_output)

    def _reset_parameters(self, zero_init_output):
        # the two bias terms start neutral, so at step 0 the weights come from content alone
        for lin in (self.query_height, self.geom_mlp[-1]):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        if zero_init_output:  # used as a residual on another BEV: start as exactly that BEV
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _pillar_scale(refs, h_img, w_img):
        """log(1 + feature pixels between neighbouring anchors) per token; 0 where a neighbour is invalid.
        refs: [B, N, L, Z, 2] normalised coords, invalid points are the -10 sentinel."""
        px = refs * refs.new_tensor([w_img / 2.0, h_img / 2.0])
        valid = (refs.abs() <= 1).all(-1)
        step = (px[..., 1:, :] - px[..., :-1, :]).norm(dim=-1)  # [B, N, L, Z-1]
        ok = valid[..., 1:] & valid[..., :-1]
        step = torch.where(ok, step, torch.zeros_like(step))
        # each anchor takes the step to its upper neighbour, the top anchor the step to its lower one
        scale = torch.cat([step, step[..., -1:]], dim=-1)
        return torch.log1p(scale)

    def _attend(self, kmap, vmap, refs, base):
        B, N, Dk, h_img, w_img = kmap.shape
        Dv, H, Z = self.value_dim, self.num_heads, self.num_z_anchors
        L = refs.shape[2]
        dh, dvh = Dk // H, Dv // H
        valid = (refs.abs() <= 1).all(-1)  # [B, N, L, Z]
        validf = valid.to(kmap.dtype)
        grid = refs.flatten(0, 1).to(kmap.dtype)  # [B*N, L, Z, 2]

        # sampled keys and values; grid_sample runs in fp32 under autocast, cast back to keep the token tensors small
        k = F.grid_sample(kmap.flatten(0, 1), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        k = k.to(kmap.dtype).view(B, N, Dk, L, Z)
        v = F.grid_sample(vmap.flatten(0, 1), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        v = v.to(vmap.dtype).view(B, N, H, dvh, L, Z)

        # multi-view consistency: cosine of each camera's key with the mean of the OTHER cameras' keys at the same point
        kn = F.normalize(k.float(), dim=2).to(k.dtype) * validf.unsqueeze(2)
        ksum = kn.sum(1, keepdim=True)  # [B, 1, Dk, L, Z]
        n_valid = validf.sum(1, keepdim=True)  # [B, 1, L, Z]
        others = (n_valid - validf).clamp(min=1).unsqueeze(2)
        sim = (kn * (ksum - kn) / others).sum(2)  # [B, N, L, Z]
        has_other = (n_valid - validf > 0).to(kmap.dtype)
        geom = torch.stack(
            [
                refs[..., 0].clamp(-1, 1),
                refs[..., 1].clamp(-1, 1),
                self._pillar_scale(refs, h_img, w_img).to(kmap.dtype),
                sim,
                has_other,
                n_valid.expand_as(sim) / N,
            ],
            dim=-1,
        ).to(kmap.dtype)  # [B, N, L, Z, 6]
        geom_logit = self.geom_mlp(geom)  # [B, N, L, Z, H]

        # query from the cell's own tokens (+ an optional BEV map)
        cnt = validf.sum((1, 3)).clamp(min=1)  # [B, L]
        kmean = (k * validf.unsqueeze(2)).sum((1, 4)) / cnt.unsqueeze(1)  # [B, Dk, L]
        kmax = k.masked_fill(~valid.unsqueeze(2), float("-inf")).amax(dim=(1, 4))
        kmax = torch.where(torch.isfinite(kmax), kmax, torch.zeros_like(kmax))
        q = self.query_proj(torch.cat([kmax, kmean], 1).transpose(1, 2))  # [B, L, Dk]
        if self.base_proj is not None:
            q = q + self.base_proj(base)
        height_logit = self.query_height(q).view(B, L, H, Z) + self.height_bias  # [B, L, H, Z]

        content = torch.einsum("blhd,bnhdlz->bnlzh", q.view(B, L, H, dh), k.view(B, N, H, dh, L, Z)) / math.sqrt(dh)
        logits = content + geom_logit + height_logit.unsqueeze(1)  # [B, N, L, Z, H]
        logits = logits.float().masked_fill(~valid.unsqueeze(-1), -1e4)
        w = logits.permute(0, 2, 4, 1, 3).reshape(B, L, H, N * Z).softmax(-1)
        any_valid = valid.any(-1).any(1)  # [B, L]
        w = (w * any_valid.view(B, L, 1, 1)).view(B, L, H, N, Z)  # cells no camera sees -> 0

        out = torch.einsum("blhnz,bnhdlz->blhd", w.to(v.dtype), v).reshape(B, L, Dv)
        out = self.output_proj(out)
        return out, w.mean(2).sum(2), w.mean(2).sum(3)  # [B,L,C], [B,L,Z], [B,L,N]

    def forward(self, features_2d, reference_points, base=None):
        """
        features_2d:      [B, N, embed_dim, H_img, W_img] (already compressed)
        reference_points: [B, N, L, Z, 2] normalised image coordinates of the pillar points (-10 = invalid)
        base:             optional [B, L, base_dim] BEV features added to the query
        """
        B, N, C, h_img, w_img = features_2d.shape
        flat = features_2d.flatten(0, 1)
        kmap = self.key_map(flat).view(B, N, self.key_dim, h_img, w_img)
        vmap = self.value_map(flat).view(B, N, self.value_dim, h_img, w_img)
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            out, height, camera = checkpoint(self._attend, kmap, vmap, reference_points, base, use_reentrant=False)
        else:
            out, height, camera = self._attend(kmap, vmap, reference_points, base)
        self.aux = {"height": height, "camera": camera} if self.keep_aux else None
        return out
