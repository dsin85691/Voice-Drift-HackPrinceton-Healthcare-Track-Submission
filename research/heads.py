"""Projection head + heads shared by training (research/) and serving (server/pipeline/acoustic.py)."""
from __future__ import annotations

import torch
from torch import nn


class Projector(nn.Module):
    """P: 768 -> 256 -> 128 (GELU, dropout 0.2, LayerNorm). Includes the input standardizer."""

    def __init__(self, d_in: int = 768, d_hid: int = 256, d_out: int = 128, p: float = 0.2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d_in))
        self.register_buffer("std", torch.ones(d_in))
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(d_hid, d_out), nn.LayerNorm(d_out))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net((z - self.mean) / self.std)


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grl(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GRL.apply(x, lam)


class Heads(nn.Module):
    """f_d: Linear 128->1 ; f_e: Linear 128->64->n_env behind a GRL."""

    def __init__(self, d: int = 128, n_env: int = 6):
        super().__init__()
        self.f_d = nn.Linear(d, 1)
        self.f_e = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, n_env))

    def disease(self, h):
        return self.f_d(h).squeeze(-1)

    def env(self, h, lam: float):
        return self.f_e(grl(h, lam))
