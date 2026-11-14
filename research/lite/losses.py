"""Composite objective (projection-head spec section 4):
    L = l_inv * L_inv + l_var * L_var + l_cov * L_cov + l_pres * L_pres
Each term can be switched off by setting its weight to 0. All terms are logged separately."""
from __future__ import annotations

import torch


def invariance(z: torch.Tensor, zt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """||z - z~||^2 per pair, averaged over pairs. reduction='mean' also averages over the d dims (default):
    the spec's sum over 128 dims makes this term ~128x the variance term at the given weights, the same
    imbalance that stopped M2/M3 from learning. 'sum' reproduces the spec formula exactly."""
    sq = (z - zt).pow(2)
    return (sq.mean(1) if reduction == "mean" else sq.sum(1)).mean()


def variance(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """(1/d) sum_j max(0, gamma - sigma_j)^2. Needs >= 2 samples; returns 0 (not NaN) for a batch of 1."""
    if z.shape[0] < 2:
        return z.new_zeros(())
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.relu(gamma - std).pow(2).mean()


def covariance(z: torch.Tensor) -> torch.Tensor:
    """(1/d) sum_{i != j} C_ij^2 of the centered batch covariance. Returns 0 for a batch of 1."""
    n, d = z.shape
    if n < 2:
        return z.new_zeros(())
    zc = z - z.mean(0)
    c = (zc.T @ zc) / (n - 1)
    off = c - torch.diag(torch.diag(c))
    return off.pow(2).sum() / d


def preservation(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE: only reliable target entries count. Returns 0 if nothing is valid."""
    m = mask.float()
    denom = m.sum()
    if denom == 0:
        return pred.new_zeros(())
    return ((pred - target).pow(2) * m).sum() / denom


def composite(z, zt, w: dict, acoustic_head=None, target=None, mask=None, gamma=1.0, inv_reduction="mean"):
    """Returns (total, parts dict of floats). Variance/covariance on each view separately, then averaged.
    Preservation uses the CLEAN-reference targets for both the clean and the augmented view."""
    parts = {"inv": invariance(z, zt, inv_reduction),
             "var": 0.5 * (variance(z, gamma) + variance(zt, gamma)),
             "cov": 0.5 * (covariance(z) + covariance(zt))}
    if w.get("pres", 0) > 0 and acoustic_head is not None:
        parts["pres"] = 0.5 * (preservation(acoustic_head(z), target, mask)
                               + preservation(acoustic_head(zt), target, mask))
    total = sum(w.get(k, 0) * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
