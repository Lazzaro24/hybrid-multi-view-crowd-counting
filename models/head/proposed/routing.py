import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNet(nn.Module):
    """Congestion Gating Network -- replaces the original 2-layer
    (Conv3x3 -> BN -> ReLU -> Conv1x1 -> Sigmoid) design with a deeper,
    dual-branch structure, ~220k parameters vs. the original ~37k.

    Motivation: overfit_gating_net_masked.py (with the downstream-mass-error
    selection metric, not raw IoU) plateaued around ~13-17% relative mass
    error regardless of loss function (Focal vs. Dice/Tversky) or Neck
    quality (Stage-1-only vs. jointly-trained-with-experts) -- the SAME
    persistent failure survived both fixes: large, over-broad, MERGED
    predicted-dense regions where gt_mask shows scattered, distinct blobs
    (FP:FN ratio ~6:1, predicted coverage ~2.5x true coverage, consistent
    across every checkpoint tested). That pattern -- not "no signal" but
    "can't hold a sharp boundary AND enough context to know when to stop"
    -- is what a 2-layer network with a small, undilated receptive field
    would produce: it has to compromise between local sharpness and
    knowing whether a nearby empty patch is truly between two clusters,
    since one shallow path can't do both. Ruling out the loss function and
    Neck features as the (sole) cause pointed at this network's own
    capacity/receptive-field as the next thing to test.

    Design: a FINE branch (plain 3x3s, no dilation) preserves local
    boundaries; a CONTEXT branch (dilated 3x3s, same spatial resolution)
    gives the network a much larger receptive field without shrinking
    resolution, so it can check "is this actually near a real cluster" before
    committing to dense; a FUSE stage combines both views into one per-pixel
    logit. Mirrors dense_expert's own use of dilation for a related tension,
    but as two parallel paths (not one dilated stack) so fine boundary
    information is never diluted away by the wider context path.

    NOTE: this is a capacity-focused hypothesis test, not a proven fix --
    compare against the original gating_net's ~13-17% mass error on the SAME
    checkpoint/metric (overfit_gating_net_masked.py's downstream mass error)
    before assuming this resolves the over-prediction pattern. Existing
    checkpoints will report gating_net as missing/unexpected keys when
    loaded (strict=False, already used everywhere this model is loaded) --
    expected, since the parameter shapes changed; gating_net simply starts
    fresh.

    CONTRAST branch, added after multi-scene diagnostics (see
    doc/router_diagnosis.md) found the fine+context design above still fails
    specifically on busy scenes: the router over-covers with HIGH confidence
    (not merely uncertain -- mask sharpening at inference could not recover
    the gap), and the false-positive rate scales with overall scene density
    rather than with cluster sharpness. Recomputed the context branch's real
    receptive field (it operates on fine_feat, not the raw input, so its RF
    is ADDITIVE on top of fine's own 5px -> ~19px total, not the "sees the
    whole scene" mechanism originally assumed) -- narrower than expected,
    but still wide enough that on busy scenes (real clusters ~1-3m apart)
    it can contain multiple SEPARATE clusters at once, so a genuine gap
    pixel between two real clusters still looks "surrounded by cluster
    evidence" to it. That is a cluster-density-confusion problem, not a
    whole-scene-context problem, so simply narrowing the context branch was
    rejected as the fix (risks reintroducing the original shallow-network's
    merged-region failure this dual-branch design was built to solve).

    The contrast branch instead gives the network an explicit band-pass
    signal -- fine_feat minus a large local average of itself (CONTRAST_
    BLUR_KERNEL, default 25px, matching dense_expert's own receptive field
    as an already-established "local zone" scale in this project) -- so it
    can directly see "how much sharper is THIS pixel than its neighbourhood
    average" instead of inferring peakiness indirectly from raw activation
    level, which conflates a real sharp peak with merely being embedded in
    a generally busy area. This is a hypothesis, not a proven fix -- compare
    diagnose_routing.py's Dice/IoU/mask mean-in-out and the per-scene
    density correlation (doc/router_diagnosis.md #1) against this branch's
    predecessor before trusting an improvement.
    """

    CONTRAST_BLUR_KERNEL = 25

    def __init__(self, in_channels=64, hidden_channels=64, use_contrast=True):
        super().__init__()
        # use_contrast=False gives the fine + context router (the tversky_alpha=0.7 reference, doc/router_diagnosis.md);
        # the contrast branch was a negative result there.
        self.use_contrast = use_contrast
        self.fine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        # no learnable parameters -- a fixed local-average filter used to build the band-pass
        # contrast signal below. count_include_pad=False so patch-edge pixels average only real
        # neighbours, not the zero padding, keeping the "local mean" honest near the boundary.
        self.local_avg = nn.AvgPool2d(
            kernel_size=self.CONTRAST_BLUR_KERNEL,
            stride=1,
            padding=self.CONTRAST_BLUR_KERNEL // 2,
            count_include_pad=False,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * (3 if use_contrast else 2), hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fine_feat = self.fine(x)
        context_feat = self.context(fine_feat)
        # contrast_feat: how much sharper fine_feat is HERE than its own local neighbourhood
        # average -- a direct peakiness/sharpness signal, distinct from context_feat's raw (smoothed)
        # activation level, which conflates a real sharp cluster with merely sitting in a busy area.
        feats = [fine_feat, context_feat]
        if self.use_contrast:
            feats.append(fine_feat - self.local_avg(fine_feat))
        fused = torch.cat(feats, dim=1)
        return self.sigmoid(self.fuse(fused))


class HybridRoutingHead(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=64, router_contrast=True, dense_arch="legacy"):
        """
        Hybrid Routing Head with specialized Experts.
        dense_arch='wide512': the dense expert is the SynMVCrowd baseline decoder's layer stack (3x3 convs
        in -> 512, 512 -> 512 + ReLU, 512 -> 512 + ReLU, 512 -> 1 without bias), to test whether the 64-channel
        legacy expert limits the count (doc/head_ablation.md §8).
        """
        super().__init__()

        # 1. Congestion Gating Network -- see GatingNet's docstring above for
        # why this was changed from the original 2-layer design.
        self.gating_net = GatingNet(in_channels, hidden_channels, use_contrast=router_contrast)

        # 2. Expert A: Dense Regression (The "Blob" Expert)
        self.dense_expert = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=5, padding=8, dilation=4),
        )
        if dense_arch == "wide512":
            self.dense_expert = nn.Sequential(
                nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),  # baseline: feat_before_merge, no activation
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 1, kernel_size=3, padding=1, bias=False),
            )
        elif dense_arch != "legacy":
            raise ValueError(f"unknown dense_arch {dense_arch!r}")

        # 3. Expert B: Sparse Detection (The "Point" Expert)
        self.sparse_expert = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

        # How the sparse expert's output enters the fusion (doc/head_ablation.md §6b): None = relu(sparse);
        # {'mode': 'peaks', 'kernel': k, 'threshold': t} = one map kernel (one person) per NMS peak, i.e. the sparse
        # expert read as a detector. Not part of the state_dict; checkpoints store it as 'sparse_render'.
        self.sparse_render = None
        self.render_kernel = None

    def set_sparse_render(self, config, map_kernel=None):
        if config and config.get("mode") == "peaks" and map_kernel is None:
            raise ValueError("sparse_render 'peaks' needs the map kernel")
        self.sparse_render = dict(config) if config and config.get("mode") == "peaks" else None
        self.render_kernel = map_kernel if self.sparse_render else None

    def sparse_map(self, x):
        """The sparse expert's contribution to the fused map (relu output, or rendered NMS peaks)."""
        s = torch.nn.functional.relu(self.sparse_expert(x))
        if self.sparse_render is None:
            return s
        s = s.float()
        k = self.sparse_render["kernel"]
        peaks = ((F.max_pool2d(s, k, stride=1, padding=k // 2) == s) & (s > self.sparse_render["threshold"])).float()
        kernel = self.render_kernel.to(device=s.device, dtype=s.dtype)
        return F.conv2d(peaks, kernel, padding=kernel.shape[-1] // 2)

    def forward(self, x):
        """
        Args:
            x: Fused BEV Tensor from the Neck [Batch, 64, 200, 200]
        Returns:
            final_map: The routed density prediction
            mask: The congestion map (needed for Auxiliary Loss)
        """
        mask = self.gating_net(x)
        dense_out = self.dense_expert(x)
        sparse_out = self.sparse_map(x)

        dense_out = torch.nn.functional.relu(dense_out)

        final_map = (mask * dense_out) + ((1.0 - mask) * sparse_out)

        return final_map, mask
