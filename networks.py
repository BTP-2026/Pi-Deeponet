"""
networks.py
===========
Shared neural network architectures for DeepONet-based PDE solvers.
"""

import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple feedforward network."""

    def __init__(self, in_dim, out_dim, hidden=(256, 256), activation=nn.Tanh):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FourierFeatureEncoder(nn.Module):
    """
    Encodes 2-D coordinates (x, y) into Fourier features to overcome spectral
    bias and help the trunk capture high-frequency spatial content.

    Output dim = 2 + 4 * n_freqs  (raw coords + sin/cos for each axis)
    For n_freqs=8 → output dim = 34.
    """

    def __init__(self, n_freqs: int = 8):
        super().__init__()
        # Fixed frequency bands: 1, 2, ..., n_freqs  (in units of 2π)
        freqs = torch.arange(1, n_freqs + 1, dtype=torch.float32) * (2.0 * math.pi)
        self.register_buffer("freqs", freqs)          # (n_freqs,)
        self.out_dim = 2 + 4 * n_freqs

    def forward(self, xy):
        """xy : (n_pts, 2) → features : (n_pts, 2 + 4*n_freqs)"""
        x = xy[:, 0:1]   # (n_pts, 1)
        y = xy[:, 1:2]
        fx = x * self.freqs          # (n_pts, n_freqs)
        fy = y * self.freqs
        return torch.cat([xy, torch.sin(fx), torch.cos(fx),
                              torch.sin(fy), torch.cos(fy)], dim=-1)


class DeepONet2D(nn.Module):
    """
    DeepONet for 2-D operator learning.

    branch_in_dim = variable (depends on problem inputs)
    trunk input   = (x, y)  (2-D coordinates)
    p             = latent dim (inner-product size)

    Parameters
    ----------
    use_fourier : bool
        If True, encode trunk (x,y) inputs with Fourier features before the
        trunk MLP.  This helps the trunk resolve high-frequency spatial
        content (e.g. sharp Gaussian peaks) without requiring many more epochs.
    n_fourier : int
        Number of frequency bands used in the Fourier encoder (default 8).
        Trunk input dim becomes 2 + 4*n_fourier.
    """

    def __init__(self, branch_in_dim, p=128,
                 branch_hidden=(512, 512), trunk_hidden=(256, 256),
                 activation=nn.Tanh,
                 use_fourier: bool = False, n_fourier: int = 8):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier_enc = FourierFeatureEncoder(n_freqs=n_fourier)
            trunk_in_dim = self.fourier_enc.out_dim
        else:
            self.fourier_enc = None
            trunk_in_dim = 2

        self.branch = MLP(branch_in_dim, p, hidden=branch_hidden, activation=activation)
        self.trunk  = MLP(trunk_in_dim, p, hidden=trunk_hidden, activation=activation)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input, xy_grid):
        """
        Parameters
        ----------
        branch_input : (B, branch_in_dim)
        xy_grid      : (n_pts, 2)

        Returns
        -------
        u_pred : (B, n_pts)
        """
        b = self.branch(branch_input)                                  # (B, p)
        t_in = self.fourier_enc(xy_grid) if self.use_fourier else xy_grid
        t = self.trunk(t_in)                                           # (n_pts, p)
        out = torch.sum(b.unsqueeze(1) * t.unsqueeze(0), dim=-1)      # (B, n_pts)
        return out + self.bias
