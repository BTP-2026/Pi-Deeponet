"""
eval_utils.py
=============
Shared evaluation and plotting utilities for 2D PDE problems.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def compute_errors(u_pred, u_true):
    """Per-sample MSE and relative L2 error."""
    axes = tuple(range(1, u_true.ndim))
    mse = np.mean((u_pred - u_true) ** 2, axis=axes)
    rel_l2 = np.sqrt(np.sum((u_pred - u_true) ** 2, axis=axes) /
                     (np.sum(u_true ** 2, axis=axes) + 1e-12))
    return mse, rel_l2


def plot_comparison(x, y, u_true, u_pred, sample_idx=0,
                    title_prefix="", save_dir=None):
    """Side-by-side true vs predicted with error field."""
    X, Y = np.meshgrid(x, y)
    ut = u_true[sample_idx] if u_true.ndim == 3 else u_true
    up = u_pred[sample_idx] if u_pred.ndim == 3 else u_pred
    err = np.abs(ut - up)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    c0 = axes[0].contourf(X, Y, ut, levels=50, cmap="viridis")
    axes[0].set_title(f"{title_prefix}True u(x,y)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_aspect("equal"); fig.colorbar(c0, ax=axes[0])

    c1 = axes[1].contourf(X, Y, up, levels=50, cmap="viridis")
    axes[1].set_title(f"{title_prefix}Predicted u(x,y)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    axes[1].set_aspect("equal"); fig.colorbar(c1, ax=axes[1])

    c2 = axes[2].contourf(X, Y, err, levels=50, cmap="hot")
    axes[2].set_title(f"{title_prefix}|Error|  (max={err.max():.4f})")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    axes[2].set_aspect("equal"); fig.colorbar(c2, ax=axes[2])

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f"comparison_sample{sample_idx}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.show()


def plot_cross_sections(x, y, u_true, u_pred, sample_idx=0,
                        title_prefix="", save_dir=None):
    """Cross-sections at y ~ 0.5 and x ~ 0.5."""
    mid_y = len(y) // 2
    mid_x = len(x) // 2
    ut = u_true[sample_idx] if u_true.ndim == 3 else u_true
    up = u_pred[sample_idx] if u_pred.ndim == 3 else u_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(x, ut[mid_y, :], "b-", lw=2, label="True")
    ax1.plot(x, up[mid_y, :], "r--", lw=2, label="Predicted")
    ax1.set_title(f"{title_prefix}Cross-section at y={y[mid_y]:.2f}")
    ax1.set_xlabel("x"); ax1.set_ylabel("u"); ax1.legend(); ax1.grid(True)

    ax2.plot(y, ut[:, mid_x], "b-", lw=2, label="True")
    ax2.plot(y, up[:, mid_x], "r--", lw=2, label="Predicted")
    ax2.set_title(f"{title_prefix}Cross-section at x={x[mid_x]:.2f}")
    ax2.set_xlabel("y"); ax2.set_ylabel("u"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f"cross_section_sample{sample_idx}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.show()


def plot_training_history(history, save_dir=None):
    """Log-scale training curves."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in history:
        ax.plot(np.log10(np.array(history[key]) + 1e-15), label=key)
    ax.set_xlabel("Epoch"); ax.set_ylabel("log10(loss)")
    ax.set_title("Training History"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "training_history.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.show()
