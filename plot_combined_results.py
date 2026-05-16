"""
plot_combined_results.py
========================
Generates a single combined figure:
  - Top: N×3 grid of (COMSOL | Predicted | |Error|) for selected v0 values
  - Bottom: rel-L2 error bar chart across all v0 values
"""

import os
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.tri as mtri

from networks import DeepONet2D


# ── data loading (same as training script) ──────────────────────────────────

def load_data(forcing_file, data_file):
    surf = np.loadtxt(forcing_file, comments='%')
    xy     = surf[:, :2].astype(np.float32)
    f_vals = surf[:, 2].astype(np.float32)

    v0_values = []
    with open(data_file) as fh:
        for line in fh:
            if not line.startswith('%'):
                break
            v0_values.extend(float(m) for m in re.findall(r'v0=([0-9.]+)', line))
    v0_values = np.array(v0_values)

    raw      = np.loadtxt(data_file, comments='%')
    u_comsol = raw[:, 2:].astype(np.float32)
    return xy, f_vals, v0_values, u_comsol


def predict_all(model, xy, v0_values, output_scale, device):
    model.eval()
    xy_t = torch.tensor(xy, dtype=torch.float32, device=device)
    preds = []
    with torch.no_grad():
        for v0 in v0_values:
            b = torch.tensor([[v0]], dtype=torch.float32, device=device)
            u = output_scale * model(b, xy_t)
            preds.append(u.squeeze(0).cpu().numpy())
    return np.stack(preds, axis=1)   # (4300, 41)


# ── plotting ─────────────────────────────────────────────────────────────────

def make_combined_figure(xy, u_comsol, u_pred, v0_values,
                         v0_plot_list, save_path):
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1])
    n_rows = len(v0_plot_list)

    rel_errors = np.array([
        np.sqrt(np.sum((u_pred[:, i] - u_comsol[:, i])**2) /
                (np.sum(u_comsol[:, i]**2) + 1e-12))
        for i in range(len(v0_values))
    ])

    fig = plt.figure(figsize=(16, 4.2 * n_rows + 4.5))
    gs  = gridspec.GridSpec(
        n_rows + 1, 3,
        height_ratios=[1.0] * n_rows + [1.1],
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97, top=0.96, bottom=0.04
    )

    for row, v0_target in enumerate(v0_plot_list):
        idx = int(np.argmin(np.abs(v0_values - v0_target)))
        v0  = v0_values[idx]
        ref = u_comsol[:, idx]
        prd = u_pred[:, idx]
        err = np.abs(ref - prd)
        rel = rel_errors[idx]
        vmin, vmax = ref.min(), ref.max()

        ax0 = fig.add_subplot(gs[row, 0])
        c0  = ax0.tricontourf(triang, ref, levels=64, cmap='jet', vmin=vmin, vmax=vmax)
        ax0.set_title(f'COMSOL  (v0={v0:.1f})', fontsize=10)
        ax0.set_aspect('equal'); ax0.set_xlabel('x'); ax0.set_ylabel('y')
        fig.colorbar(c0, ax=ax0, fraction=0.046, pad=0.04)

        ax1 = fig.add_subplot(gs[row, 1])
        c1  = ax1.tricontourf(triang, prd, levels=64, cmap='jet', vmin=vmin, vmax=vmax)
        ax1.set_title(f'PI-DeepONet  rel-L2={rel*100:.3f}%', fontsize=10)
        ax1.set_aspect('equal'); ax1.set_xlabel('x'); ax1.set_ylabel('y')
        fig.colorbar(c1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[row, 2])
        c2  = ax2.tricontourf(triang, err, levels=64, cmap='hot')
        ax2.set_title(f'|Error|  max={err.max():.4f}', fontsize=10)
        ax2.set_aspect('equal'); ax2.set_xlabel('x'); ax2.set_ylabel('y')
        fig.colorbar(c2, ax=ax2, fraction=0.046, pad=0.04)

    # Bottom row: error bar chart across all v0
    ax_bar = fig.add_subplot(gs[n_rows, :])
    colors = ['#d62728' if e > 0.01 else '#1f77b4' for e in rel_errors]
    ax_bar.bar(v0_values, rel_errors * 100, color=colors, alpha=0.85, width=0.4)
    ax_bar.axhline(0.5, color='red',    linestyle='--', lw=1.2, label='0.5% threshold')
    ax_bar.axhline(0.25, color='green', linestyle='--', lw=1.2, label='0.25% threshold')
    ax_bar.set_xlabel('v0 (Dirichlet BC at x=0)', fontsize=11)
    ax_bar.set_ylabel('Rel-L2 error (%)', fontsize=11)
    ax_bar.set_title(
        f'PI-DeepONet rel-L2 vs COMSOL  |  Mean={rel_errors.mean()*100:.3f}%  '
        f'Max={rel_errors.max()*100:.3f}%', fontsize=11)
    ax_bar.set_xticks(v0_values[::2])
    ax_bar.legend(fontsize=10); ax_bar.grid(axis='y', alpha=0.4)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved -> {save_path}')
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--forcing_file',  default='Surface_Solution.txt')
    parser.add_argument('--data_file',     default='data_v0.txt')
    parser.add_argument('--checkpoint',    default='output_v0_run2/model_best.pth')
    parser.add_argument('--log_dir',       default='output_v0_run2')
    parser.add_argument('--p_dim',         type=int,   default=128)
    parser.add_argument('--branch_h',      type=int,   nargs='+', default=[64, 64])
    parser.add_argument('--trunk_h',       type=int,   nargs='+', default=[128, 128])
    parser.add_argument('--n_fourier',     type=int,   default=8)
    parser.add_argument('--output_scale',  type=float, default=20.0)
    parser.add_argument('--v0_plot',       type=float, nargs='+',
                        default=[0.0, 5.0, 10.0, 15.0, 20.0],
                        help='v0 values to show comparison panels for')
    parser.add_argument('--out_file',      default='combined_results.png')
    args = parser.parse_args()

    print('Loading data ...')
    xy, f_vals, v0_values, u_comsol = load_data(args.forcing_file, args.data_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = DeepONet2D(
        branch_in_dim=1,
        p=args.p_dim,
        branch_hidden=tuple(args.branch_h),
        trunk_hidden=tuple(args.trunk_h),
        activation=nn.SiLU,
        use_fourier=True,
        n_fourier=args.n_fourier,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    print(f'Loaded checkpoint: {args.checkpoint}')

    print('Running predictions ...')
    u_pred = predict_all(model, xy, v0_values, args.output_scale, device)

    save_path = os.path.join(args.log_dir, args.out_file)
    print('Generating combined figure ...')
    make_combined_figure(xy, u_comsol, u_pred, v0_values, args.v0_plot, save_path)


if __name__ == '__main__':
    main()
