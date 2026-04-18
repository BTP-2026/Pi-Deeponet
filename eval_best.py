"""
eval_best.py
============
Load the best hyperparameter-search checkpoint and generate 3-panel
comparison plots for v0 = 0, 5, 10, 15, 20.
"""

import os
import numpy as np
import torch
import torch.nn as nn

from deeponet_2d_neumann import load_neumann_data, NeumannPdeTrainer, plot_three_panel, plot_error_summary
from networks import DeepONet2D

# ---- Config from best Phase-2 result (config_01) ----
GRID_SIZE    = 31
F_SCALE      = 100.0
DATA_FILE    = "data_v0.txt"
FORCING_FILE = "Surface_Solution.txt"
CKPT         = "hparam_results/phase2/config_01/model_best.pth"
OUT_DIR      = "hparam_results/phase2/config_01/plots"

CFG = {
    "lr": 1e-3,
    "bc_d_weight": 5.0,
    "bc_n_weight": 1.0,
    "p_dim": 512,
    "branch_h": [256, 256],
    "trunk_h": [512, 512, 512],
    "n_fourier": 8,
    "batch_size": None,
}

os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
N = GRID_SIZE

print("Loading data ...")
x, y, v0_values, u_grids, f_comsol = load_neumann_data(DATA_FILE, FORCING_FILE, N=N)
f_raw = f_comsol / F_SCALE

# Build model
branch_in_dim = N * N + 1
model = DeepONet2D(
    branch_in_dim=branch_in_dim,
    p=CFG["p_dim"],
    branch_hidden=tuple(CFG["branch_h"]),
    trunk_hidden=tuple(CFG["trunk_h"]),
    activation=nn.SiLU,
    use_fourier=True,
    n_fourier=CFG["n_fourier"],
).to(device)

print(f"Loading checkpoint: {CKPT}")
model.load_state_dict(torch.load(CKPT, map_location=device))

# Build trainer (needed for predict / u_scales)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
trainer = NeumannPdeTrainer(
    model=model, x=x, y=y,
    v0_values=v0_values, u_grids=u_grids, f_raw=f_raw,
    optimizer=optimizer, scheduler=None,
    device=device, f_scale=F_SCALE,
    bc_d_weight=CFG["bc_d_weight"],
    bc_n_weight=CFG["bc_n_weight"],
    batch_size=CFG["batch_size"],
)

# Evaluate all v0
print("\nEvaluating all v0 values ...")
rel_errors = np.zeros(len(v0_values))
for i, v0 in enumerate(v0_values):
    u_pred = trainer.predict(f_raw, v0)
    u_ref  = u_grids[i]
    rel = np.sqrt(np.sum((u_pred - u_ref) ** 2) / (np.sum(u_ref ** 2) + 1e-12))
    rel_errors[i] = rel

print(f"\n{'v0':>6}  {'rel-L2 (%)':>12}")
print("-" * 22)
for v0, err in zip(v0_values, rel_errors):
    print(f"{v0:6.1f}  {err*100:12.4f}")
print(f"\nMean rel-L2: {rel_errors.mean()*100:.4f}%")
print(f"Max  rel-L2: {rel_errors.max()*100:.4f}%")

# Error bar chart
plot_error_summary(v0_values, rel_errors, save_dir=OUT_DIR, fname="error_vs_v0.png")

# 3-panel plots for selected v0 values
for v0_plot in [0.0, 5.0, 10.0, 15.0, 20.0]:
    idx = int(np.argmin(np.abs(v0_values - v0_plot)))
    v0_actual = v0_values[idx]
    u_pred = trainer.predict(f_raw, v0_actual)
    u_ref  = u_grids[idx]
    plot_three_panel(
        x, y, u_ref, u_pred,
        save_dir=OUT_DIR,
        fname=f"comparison_v0_{v0_actual:.1f}.png",
        ref_label=f"COMSOL  v0={v0_actual:.1f}",
    )
    print(f"  v0={v0_actual:.1f}  rel-L2={rel_errors[idx]*100:.4f}%")

print(f"\nAll plots saved to {os.path.abspath(OUT_DIR)}/")
