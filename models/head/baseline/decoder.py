import torch
import torch.nn as nn


class BaselineDecoder(nn.Module):
    def __init__(self, num_cam=5, out_channel=512):
        super().__init__()
        self.num_cam = num_cam

        # SynMVCrowd's `feat_before_merge`: one 3x3 conv (no activation) on the fused features
        self.feat_before_merge = nn.Conv2d(out_channel, 512, kernel_size=3, padding=1)

        self.map_classifier = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, bev_features):
        """
        Args:
            bev_features: [Batch, Cameras, Channels, BEV_H, BEV_W]
        """
        # 1. Multi-View Fusion via Max-Pooling across the camera dimension (dim=1)
        # Shape: [B, N, C, H, W] -> [B, C, H, W]
        fused_features, _ = torch.max(bev_features, dim=1)
        fused_features = self.feat_before_merge(fused_features)

        # 2. Pass through the baseline's map classifier
        map_result = self.map_classifier(fused_features)

        # 3. Absolute value in train AND eval, as in the original (its eval-time clamp is a no-op after abs)
        return torch.abs(map_result)
