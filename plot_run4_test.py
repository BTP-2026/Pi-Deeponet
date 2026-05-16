"""
plot_run4_test.py
=================
Generates two figures for run4 test v0 in [0, 10]:
  1. Error bar chart (v0=0..10, train=blue, test=red)
  2. N×3 comparison grid (COMSOL | Predicted | |Error|) for test-only v0 values
"""

import re
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.tri as mtri

from networks import DeepONet2D


N_V0_FREQS   = 4
P_DIM        = 256
BRANCH_H     = (64, 64)
TRUNK_H      = (256, 256, 256)
N_FOURIER    = 16
OUTPUT_SCALE = 20.0
CHECKPOINT   = "output_v0_run4/model_best.pth"
TRAIN_STEP   = 2.0
LOG_DIR      = "output_v0_run4"


def encode_v0(v0_array, n_freqs=4):
    v0  = np.asarray(v0_array, dtype=np.float32).reshape(-1, 1)
    ks  = np.arange(1, n_freqs + 1, dtype=np.float32)
    arg = np.pi * ks[None, :] * v0 / 20.0
    return np.concatenate([v0, np.sin(arg), np.cos(arg)], axis=1)


def load_data(forcing_file="Surface_Solution.txt", data_file="data_v0.txt"):
    surf   = np.loadtxt(forcing_file, comments='%')
    xy     = surf[:, :2].astype(np.float32)

    v0_values = []
    with open(data_file) as fh:
        for line in fh:
            if not line.startswith('%'):
                break
            v0_values.extend(float(m) for m in re.findall(r'v0=([0-9.]+)', line))
    v0_values = np.array(v0_values, dtype=np.float32)

    raw      = np.loadtxt(data_file, comments='%')
    u_comsol = raw[:, 2:].astype(np.float32)
    return xy, v0_values, u_comsol


def predict_all(model, xy, v0_values, device):
    model.eval()
    xy_t    = torch.tensor(xy, device=device)
    encoded = encode_v0(v0_values, n_freqs=N_V0_FREQS)
    preds   = []
    with torch.no_grad():
        for enc in encoded:
            b = torch.tensor(enc[None, :], dtype=torch.float32, device=device)
            u = OUTPUT_SCALE * model(b, xy_t)
            preds.append(u.squeeze(0).cpu().numpy())
    return np.stack(preds, axis=1)   # (n_pts, n_v0)


def rel_l2(pred, ref):
    return np.sqrt(np.sum((pred - ref)**2) / (np.sum(ref**2) + 1e-12))


def main():
    xy, v0_values, u_comsol = load_data()

    # Train/test split
    tol        = TRAIN_STEP * 0.01
    train_mask = np.array([abs(v % TRAIN_STEP) < tol or
                            abs(v % TRAIN_STEP - TRAIN_STEP) < tol
                            for v in v0_values])

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

    u_pred = predict_all(model, xy, v0_values, device)

    rel_errors = np.array([rel_l2(u_pred[:, i], u_comsol[:, i])
                           for i in range(len(v0_values))])

    # ── restrict to v0 in [0, 10] ────────────────────────────────────────────
    mask_10     = v0_values <= 10.0 + 1e-6
    v0_sub      = v0_values[mask_10]
    err_sub     = rel_errors[mask_10]
    train_sub   = train_mask[mask_10]
    idx_sub     = np.where(mask_10)[0]

    # ── Figure 1: error bar chart ─────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(12, 4))
    colors = ['#1f77b4' if t else '#d62728' for t in train_sub]
    bars   = ax.bar(v0_sub, err_sub * 100, color=colors, width=0.3, alpha=0.85)

    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#1f77b4', label='Train'),
                       Patch(color='#d62728', label='Test')], fontsize=11)
    ax.set_xlabel("v0 (Dirichlet BC at x=0)", fontsize=12)
    ax.set_ylabel("Rel-L2 error (%)", fontsize=12)

    test_mean = err_sub[~train_sub].mean() * 100
    train_mean = err_sub[train_sub].mean() * 100
    ax.set_title(
        f"Run4 rel-L2 error — v0 ∈ [0, 10]  |  "
        f"Train mean={train_mean:.2f}%  Test mean={test_mean:.2f}%",
        fontsize=12
    )
    ax.set_xticks(v0_sub)
    ax.tick_params(axis='x', labelrotation=45)
    ax.grid(axis='y', alpha=0.4)
    fig1.tight_layout()
    out1 = f"{LOG_DIR}/error_v0_0_10.png"
    fig1.savefig(out1, dpi=150)
    print(f"Saved -> {out1}")
    plt.close(fig1)

    # ── Figure 2: comparison grid for test v0 in [0, 10] ─────────────────────
    test_idx_in_sub = np.where(~train_sub)[0]        # indices within sub-array
    test_global     = idx_sub[test_idx_in_sub]        # global indices
    test_v0_vals    = v0_values[test_global]

    triang  = mtri.Triangulation(xy[:, 0], xy[:, 1])
    n_rows  = len(test_v0_vals)

    fig2 = plt.figure(figsize=(15, 4.0 * n_rows))
    gs   = gridspec.GridSpec(n_rows, 3, hspace=0.45, wspace=0.35,
                              left=0.06, right=0.97, top=0.97, bottom=0.03)

    for row, (gi, v0) in enumerate(zip(test_global, test_v0_vals)):
        ref  = u_comsol[:, gi]
        prd  = u_pred[:, gi]
        err  = np.abs(ref - prd)
        rel  = rel_errors[gi]
        vmin, vmax = ref.min(), ref.max()

        ax0 = fig2.add_subplot(gs[row, 0])
        c0  = ax0.tricontourf(triang, ref, levels=64, cmap='jet', vmin=vmin, vmax=vmax)
        ax0.set_title(f"COMSOL  (v0={v0:.1f})", fontsize=9)
        ax0.set_aspect('equal'); ax0.set_xlabel('x'); ax0.set_ylabel('y')
        fig2.colorbar(c0, ax=ax0, fraction=0.046, pad=0.04)

        ax1 = fig2.add_subplot(gs[row, 1])
        c1  = ax1.tricontourf(triang, prd, levels=64, cmap='jet', vmin=vmin, vmax=vmax)
        ax1.set_title(f"PI-DeepONet  rel-L2={rel*100:.3f}%", fontsize=9)
        ax1.set_aspect('equal'); ax1.set_xlabel('x'); ax1.set_ylabel('y')
        fig2.colorbar(c1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig2.add_subplot(gs[row, 2])
        c2  = ax2.tricontourf(triang, err, levels=64, cmap='hot')
        ax2.set_title(f"|Error|  max={err.max():.4f}", fontsize=9)
        ax2.set_aspect('equal'); ax2.set_xlabel('x'); ax2.set_ylabel('y')
        fig2.colorbar(c2, ax=ax2, fraction=0.046, pad=0.04)

    out2 = f"{LOG_DIR}/comparison_test_v0_0_10.png"
    fig2.savefig(out2, dpi=130, bbox_inches='tight')
    print(f"Saved -> {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
