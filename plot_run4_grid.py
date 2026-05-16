"""
plot_run4_grid.py
=================
Generates a 3-row × 10-column comparison grid for run4.
  Rows   : COMSOL reference | PI-DeepONet prediction | Absolute error
  Columns: v0 = 0, 2, 4, 6, 8, 10, 12, 14, 16, 18  (step 2, 10 values)
"""

import re
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from networks import DeepONet2D


N_V0_FREQS   = 4
P_DIM        = 256
BRANCH_H     = (64, 64)
TRUNK_H      = (256, 256, 256)
N_FOURIER    = 16
OUTPUT_SCALE = 20.0
CHECKPOINT   = "output_v0_run4/model_best.pth"
LOG_DIR      = "output_v0_run4"

V0_COLS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]   # 10 columns


def encode_v0(v0_array, n_freqs=4):
    v0  = np.asarray(v0_array, dtype=np.float32).reshape(-1, 1)
    ks  = np.arange(1, n_freqs + 1, dtype=np.float32)
    arg = np.pi * ks[None, :] * v0 / 20.0
    return np.concatenate([v0, np.sin(arg), np.cos(arg)], axis=1)


def load_data():
    surf = np.loadtxt("Surface_Solution.txt", comments='%')
    xy   = surf[:, :2].astype(np.float32)

    v0_values = []
    with open("data_v0.txt") as fh:
        for line in fh:
            if not line.startswith('%'):
                break
            v0_values.extend(float(m) for m in re.findall(r'v0=([0-9.]+)', line))
    v0_values = np.array(v0_values, dtype=np.float32)

    raw      = np.loadtxt("data_v0.txt", comments='%')
    u_comsol = raw[:, 2:].astype(np.float32)
    return xy, v0_values, u_comsol


def predict_selected(model, xy, v0_targets, v0_values, device):
    model.eval()
    xy_t    = torch.tensor(xy, device=device)
    results = {}
    with torch.no_grad():
        for v0t in v0_targets:
            idx = int(np.argmin(np.abs(v0_values - v0t)))
            v0  = v0_values[idx]
            enc = encode_v0([v0], n_freqs=N_V0_FREQS)
            b   = torch.tensor(enc, dtype=torch.float32, device=device)
            u   = OUTPUT_SCALE * model(b, xy_t)
            results[v0t] = (idx, u.squeeze(0).cpu().numpy())
    return results


def main():
    xy, v0_values, u_comsol = load_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = DeepONet2D(
        branch_in_dim=1 + 2 * N_V0_FREQS,
        p=P_DIM,
        branch_hidden=BRANCH_H,
        trunk_hidden=TRUNK_H,
        activation=nn.SiLU,
        use_fourier=True,
        n_fourier=N_FOURIER,
    ).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    print(f"Loaded {CHECKPOINT}")

    preds = predict_selected(model, xy, V0_COLS, v0_values, device)
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1])

    n_cols = len(V0_COLS)   # 10
    n_rows = 3              # COMSOL | Predicted | Error

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols, 2.8 * n_rows),
        gridspec_kw=dict(hspace=0.35, wspace=0.15,
                         left=0.04, right=0.97,
                         top=0.93, bottom=0.04)
    )

    row_labels = ["COMSOL", "PI-DeepONet", "|Error|"]
    cmaps      = ["jet", "jet", "hot"]

    for col, v0t in enumerate(V0_COLS):
        idx, u_pred = preds[v0t]
        ref  = u_comsol[:, idx]
        err  = np.abs(ref - u_pred)
        rel  = np.sqrt(np.sum((u_pred - ref)**2) / (np.sum(ref**2) + 1e-12))
        vmin, vmax = ref.min(), ref.max()

        data_rows  = [ref, u_pred, err]
        clim_rows  = [(vmin, vmax), (vmin, vmax), (0, err.max())]

        for row in range(3):
            ax = axes[row, col]
            c  = ax.tricontourf(triang, data_rows[row], levels=48,
                                cmap=cmaps[row],
                                vmin=clim_rows[row][0],
                                vmax=clim_rows[row][1])
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])

            if row == 0:
                ax.set_title(f"v0={v0t}", fontsize=9, fontweight='bold')
            if row == 2:
                ax.set_xlabel(f"rel-L2={rel*100:.2f}%", fontsize=7.5)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8)

    fig.suptitle(
        "Run4 — PI-DeepONet 3×10 Comparison  (v0 = 0, 2, 4, …, 18)",
        fontsize=12, fontweight='bold'
    )

    out = f"{LOG_DIR}/grid_3x10.png"
    fig.savefig(out, dpi=160, bbox_inches='tight')
    print(f"Saved -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
