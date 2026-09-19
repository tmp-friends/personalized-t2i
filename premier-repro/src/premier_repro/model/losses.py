"""Flow-matching loss, timestep sampling and the dispersion loss (Premier Sec. 4.2)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .loading import flux_shift, shift_sigmas


def sample_timesteps(batch: int, image_seq_len: int, device, scheme: str = "logit_normal",
                     logit_mean: float = 0.0, logit_std: float = 1.0, shift: float | str = "dynamic") -> torch.Tensor:
    """t ∈ (0,1); z_t = (1-t) z_0 + t z_1.  logit-normal density + FLUX resolution-dependent shift."""
    if scheme == "logit_normal":
        u = torch.sigmoid(torch.randn(batch, device=device) * logit_std + logit_mean)
    elif scheme == "uniform":
        u = torch.rand(batch, device=device)
    else:
        raise ValueError(scheme)
    s = flux_shift(image_seq_len) if shift == "dynamic" else float(shift)
    return shift_sigmas(u, s).clamp(1e-4, 1 - 1e-4)


def flow_matching_loss(model_pred: torch.Tensor, noise: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    target = noise - latents
    return F.mse_loss(model_pred.float(), target.float())


def dispersion_loss(delta: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """InfoNCE-style dispersion loss (Wang & He 2025 "Diffuse and Disperse"; Premier Eq. 7).

        L_disp = log  mean_{u≠u'} exp( -D(Δ_u, Δ_u') / τ ),   D = ||Δ_u - Δ_u'||² / dim

    delta: [U, ...] modulation directions of U *different* users for the same (empty) prompt.
    """
    u = delta.shape[0]
    if u < 2:
        return delta.new_zeros(())
    z = delta.reshape(u, -1).float()
    d = torch.pdist(z).pow(2) / z.shape[1]          # [U(U-1)/2]
    return torch.logsumexp(-d / tau, dim=0) - math.log(d.numel())
