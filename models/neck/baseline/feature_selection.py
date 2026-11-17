import torch.nn as nn


class ViewFeatureSelector(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=64):
        """
        SynMVCrowd's spatial feature selection (`depth_classifier` in mvdetr.py).
        Predicts one score per image pixel and softmaxes it ACROSS the camera dimension,
        so every image-space feature is weighted by how much its view should contribute.
        Applied before projection (the original uses depth_scales=1, whose softmax(dim=0)
        runs over the views).
        """
        super().__init__()
        # camera ablation: the softmax weights average 1/N, so a model trained on 5 views sees its features
        # scaled by 5/N with N views; view_scale = N/5 restores the trained mean weight (utils/camera_slots.py)
        self.view_scale = 1.0
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
        )

    def forward(self, features):
        """
        Args:
            features: [Batch, Cameras, Channels, H_feat, W_feat]
        """
        B, N, C, H, W = features.shape
        scores = self.classifier(features.reshape(B * N, C, H, W)).view(B, N, 1, H, W)
        weights = scores.softmax(dim=1)
        if self.view_scale != 1.0:
            weights = weights * self.view_scale
        return features * weights
