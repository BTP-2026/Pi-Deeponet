"""
plot_pinto_style.py
===================
Generates a Pinto-style 3×5 comparison figure:
  Row 1: COMSOL reference
  Row 2: DeepONet + v0*(1-x) prediction  (with MSE in title)
  Row 3: Absolute error                   (with max error in title)

5 columns: v0 = 0, 5, 10, 15, 20
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from deeponet_2d_dirichlet_linear import load_dirichlet_data, DirichletLinearTrainer
from networks import DeepONet2D

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE    = "data_v0.txt"
FORCING_FILE = "Surface_Solution.txt"
CKPT         = "./output_dirichlet_linear/model_best.pth"
OUT_FILE     = "./output_dirichlet_linear/pinto_style_result.png"
N            = 31
F_SCALE      = 100.0
V0_PLOT      = [0.0, 5.0, 10.0, 15.0, 20.0]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
x, y, v0_values, u_grids, f_comsol, u0_grid = load_dirichlet_data(
    DATA_FILE, FORCING_FILE, N=N)
f_raw = f_comsol / F_SCALE

# ── Build & load model ────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DeepONet2D(
    branch_in_dim=N * N,
    p=512,
    branch_hidden=(256, 256),
    trunk_hidden=(512, 512, 512),
    activation=nn.SiLU,
    use_fourier=True,
    n_fourier=8,
)
model.load_state_dict(torch.load(CKPT, map_location=device))
print(f"Loaded checkpoint: {CKPT}")

# Dummy optimizer (not used for inference)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = DirichletLinearTrainer(
    model=model, x=x, y=y,
    f_raw=f_raw, u0_true=u0_grid,
    optimizer=optimizer,
    device=device, f_scale=F_SCALE,
)

# ── Collect panels ────────────────────────────────────────────────────────────
X, Y = np.meshgrid(x, y)
cols = []
for v0_target in V0_PLOT:
    idx = int(np.argmin(np.abs(v0_values - v0_target)))
    v0  = v0_values[idx]

    u_ref  = u_grids[idx]
    u_pred = trainer.predict(v0)
    err    = np.abs(u_ref - u_pred)
    mse    = float(np.mean((u_ref - u_pred) ** 2))
    maxerr = float(err.max())

    cols.append(dict(v0=v0, u_ref=u_ref, u_pred=u_pred, err=err,
                     mse=mse, maxerr=maxerr))

# ── Plot ──────────────────────────────────────────────────────────────────────
n_cols = len(cols)
fig, axes = plt.subplots(3, n_cols, figsize=(4.2 * n_cols, 12))

for j, c in enumerate(cols):
    v0     = c["v0"]
    u_ref  = c["u_ref"]
    u_pred = c["u_pred"]
    err    = c["err"]
    mse    = c["mse"]
    maxerr = c["maxerr"]

    vmin = u_ref.min()
    vmax = u_ref.max()

    # ── Row 0: COMSOL reference ──────────────────────────────────────────────
    ax = axes[0, j]
    cf = ax.contourf(X, Y, u_ref, levels=64, cmap="jet", vmin=vmin, vmax=vmax)
    ax.set_title(f"COMSOL (v0={v0:.0f})", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 1: DeepONet prediction ───────────────────────────────────────────
    ax = axes[1, j]
    cf = ax.contourf(X, Y, u_pred, levels=64, cmap="jet", vmin=vmin, vmax=vmax)
    ax.set_title(f"PINTO (MSE={mse:.2e})", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 2: Absolute error ────────────────────────────────────────────────
    ax = axes[2, j]
    cf = ax.contourf(X, Y, err, levels=64, cmap="jet")
    ax.set_title(f"Error (max={maxerr:.2e})", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("DeepONet (linear decomposition) vs COMSOL  —  2D Poisson, variable Dirichlet BC",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved -> {OUT_FILE}")
