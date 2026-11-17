import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import multivariate_normal


def get_map_kernel(sigma=5, kernel_size=20):
    """
    Replicates the exact Gaussian kernel generation from the baseline's frameDataset.py.
    This creates the blurring kernel applied to the target heatmap inside the loss function.
    """
    x, y = np.meshgrid(np.arange(-kernel_size, kernel_size + 1), np.arange(-kernel_size, kernel_size + 1))
    pos = np.stack([x, y], axis=2)
    map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * sigma)
    map_kernel = map_kernel / map_kernel.max()

    # Conv2d weight layout [out, in, H, W]
    k_size = map_kernel.shape[0]
    kernel_tensor = torch.zeros([1, 1, k_size, k_size], requires_grad=False)
    kernel_tensor[0, 0] = torch.from_numpy(map_kernel)

    return kernel_tensor


class GaussianMSE_norm(nn.Module):
    def __init__(self, scale=100):
        super().__init__()
        self.scale = scale

    def _target_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.inference_mode():
            target = F.conv2d(
                target, kernel.to(dtype=target.dtype, device=target.device), padding=int((kernel.shape[-1] - 1) / 2)
            )
        return target

    def forward(self, x, target, kernel):
        """
        Args:
            x (Tensor): The network's prediction [B, 1, H, W]
            target (Tensor): The ground truth heatmap [B, 1, H, W]
            kernel (Tensor): The Gaussian blur kernel [1, 1, K, K]
        """
        # Blur the target, then rescale it so it sums to count * scale
        before_sum = target.sum()
        target = self._target_transform(x, target, kernel)
        after_sum = target.sum()
        target = target * before_sum * self.scale / (after_sum + 1e-6)
        loss = F.mse_loss(x, target)

        return loss, target
