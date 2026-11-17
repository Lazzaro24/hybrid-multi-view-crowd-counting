"""
Faithful port of the SynMVCrowd baseline's OT loss (SynMVBase/.../loss/ot_loss/genloss.py,
GeneralizedLoss), used for the "Baseline (OT)" variant.

Kept separate from genloss.py on purpose: that file was adapted for the proposed model's sparse
expert (density renormalised to a PMF, ReLU + 1e-4 cut-off, blur 0.05, top-k, per-point loss
normalisation) and optimises a different objective. Here, as in the original:
- predicted mass A = the raw density value of every nonzero cell (the baseline outputs abs(),
  so that is effectively every cell of the 200x200 patch)
- target mass B = `scale` (100) per GT point, so a converged map sums to count * 100 --
  the same scale as the MSE variant, counted with pred.sum() / 100
- unbalanced Sinkhorn (geomloss SamplesLoss, tensorized): blur 1e-4, scaling 0.75, reach 0.5, p 1
- cost 'ed': exp(euclidean distance) - 1 on coordinates normalised by the patch size
- loss = emd + tau2 * pixel_loss + tau1 * point_loss + blur * entropy, with the original's
  mass-ordering gates on tau1 / tau2
Only change: tensors go to the density's device instead of a hard-coded 'cuda:0'.
"""

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from .Cost import EDCost, MDCost
from .geomloss import SamplesLoss

EPS = 1e-20


class SynMVCrowdGeneralizedLoss(_Loss):
    def __init__(self, cost="ed", reduction="mean", scale=100) -> None:
        super().__init__()
        self.tau1 = 1
        self.tau2 = 1
        self.cost = MDCost() if cost == "md" else EDCost()
        self.blur = 0.0001
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

    def forward(self, dens, dots, cam_loc=None):
        """
        dens: [1, 1, H, W] predicted density of one patch
        dots: [1, 1, N, 2] GT points as (x, y), normalised by the patch size (index / H)
        cam_loc: camera locations, only read by the 'md' cost
        """
        den = dens[0, 0]
        if dots.shape[2] < 1:
            # original trainer: a patch without GT points is trained towards zero density
            return torch.abs(den).mean()
        if den.sum() < 1e-8:
            # original genloss: all-zero prediction (point + pixel + emd fallback terms)
            return 3 * torch.abs(den).mean()

        device = den.device
        B = self.scale * torch.ones(1, dots.shape[2], 1, device=device)
        B_coord = dots[0].to(device)
        A, A_coord = self.den2coord(den)

        oploss, F, G = self.uot(A, A_coord, B, B_coord, cam_loc)
        C = self.cost(A_coord, B_coord, cam_loc)
        PI = torch.exp((F.view(1, -1, 1) + G.view(1, 1, -1) - C).detach() / (self.blur**self.p)) * A * B.view(1, 1, -1)

        entropy = torch.mean((EPS + PI) * torch.log(EPS + PI))
        emd_loss = torch.mean(oploss)
        point_loss = self.point_loss(PI.sum(dim=1).view(1, -1, 1), B)
        pixel_loss = self.pixel_loss(PI.sum(dim=2).detach().view(1, -1, 1), A)

        mp, mb, ma = torch.sum(PI), torch.sum(B), torch.sum(A)
        tau1 = 0 if ma < mb < mp or mp < mb < ma else self.tau1
        tau2 = 0 if mb < ma < mp or mp < ma < mb else self.tau2
        return emd_loss + tau2 * pixel_loss + tau1 * point_loss + self.blur * entropy

    @staticmethod
    def den2coord(den):
        coord = torch.nonzero(den)
        values = den[coord[:, 0], coord[:, 1]]
        coord = coord.float() / float(den.shape[-2])  # original divides both axes by H
        coord = coord[:, [1, 0]]  # (row, col) -> (x, y)
        return values.reshape(1, -1, 1), coord.unsqueeze(0)


def patch_points(point_map):
    """GT points of one [H, W] patch in the loss's format, exactly as the original trainer builds them."""
    points = torch.nonzero(point_map).float() / float(point_map.shape[-2])
    return points[:, [1, 0]].unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2] as (x, y)
