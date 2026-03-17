"""
deeponet_2d_poisson.py
======================
2D Poisson equation solver using Physics-Informed DeepONet (PyTorch).

PDE:  -∇²u = 100·f   on  [0,1]×[0,1]
      u = g            on  ∂Ω  (Dirichlet BCs on all 4 sides)

Architecture
------------
  Branch: MLP([f_flat, bc_left, bc_right, bc_bottom, bc_top]) → ℝ^p
  Trunk:  MLP(x, y)                                            → ℝ^p
  Output: u(x,y) = Σ_k branch_k · trunk_k + bias

Training losses
---------------
  L = L_data  +  bc_weight · L_bc  +  res_weight · L_residual
  where L_residual enforces -∇²u_pred = 100·f via autograd.

Usage
-----
  python deeponet_2d_poisson.py                         # defaults
  python deeponet_2d_poisson.py --n_samples 500 --epochs 500 --grid_size 31
  python deeponet_2d_poisson.py --force_type gaussian   # varying source
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3d projection)

# ---------------------------------------------------------------------------
# Import the FD solver & helpers from temp1.py (same directory)
# ---------------------------------------------------------------------------
_cwd = os.getcwd()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temp1 import (
    solve_poisson_2d,
    get_boundary_indices,
    generate_random_bc,
    plot_3d_solution,
)
os.chdir(_cwd)  # restore in case temp1 changed it


# ============================================================================
#  Dataset generation
# ============================================================================

def gaussian_forcing_2d(X, Y, mu_x, mu_y, sigma, amplitude=1.0):
    """Isotropic 2-D Gaussian bump."""
    return amplitude * np.exp(-((X - mu_x) ** 2 + (Y - mu_y) ** 2) / (2.0 * sigma ** 2))


def generate_dataset(n_samples, N=31, bc_type="four_sides",
                     force_type="constant", bc_low=1.0, bc_high=10.0,
                     seed=42):
    """
    Generate 2-D Poisson dataset using the FD solver from temp1.py.

    Returns
    -------
    x, y : 1-D arrays (N,)
    f_all : (n_samples, N, N)
    u_all : (n_samples, N, N)
    bc_all : dict  {'left': (n_samples, N), 'right': ..., ...}
    """
    np.random.seed(seed)
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    f_all = np.zeros((n_samples, N, N), dtype=np.float64)
    u_all = np.zeros_like(f_all)
    bc_left = np.zeros((n_samples, N), dtype=np.float64)
    bc_right = np.zeros_like(bc_left)
    bc_bottom = np.zeros((n_samples, N), dtype=np.float64)
    bc_top = np.zeros_like(bc_bottom)

    print(f"Generating {n_samples} samples on {N}×{N} grid  "
          f"(bc={bc_type}, force={force_type}) ...")

    for i in range(n_samples):
        # --- source term ---
        if force_type == "constant":
            f = np.ones((N, N))
        elif force_type == "gaussian":
            mu_x = np.random.uniform(0.2, 0.8)
            mu_y = np.random.uniform(0.2, 0.8)
            sigma = np.random.uniform(0.05, 0.20)
            amp = np.random.uniform(0.5, 2.0)
            f = gaussian_forcing_2d(X, Y, mu_x, mu_y, sigma, amp)
        else:
            raise ValueError(f"Unknown force_type: {force_type}")

        # --- boundary conditions ---
        bc = generate_random_bc(N, N, bc_type=bc_type, low=bc_low, high=bc_high)

        # --- solve ---
        u = solve_poisson_2d(x, y, f, bc)

        f_all[i] = f
        u_all[i] = u
        bc_left[i] = bc["left"]
        bc_right[i] = bc["right"]
        bc_bottom[i] = bc["bottom"]
        bc_top[i] = bc["top"]

        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  {i + 1}/{n_samples}")

    bc_all = {
        "left": bc_left,
        "right": bc_right,
        "bottom": bc_bottom,
        "top": bc_top,
    }
    print("Dataset generation complete.")
    return x, y, f_all, u_all, bc_all


# ============================================================================
#  Neural network building blocks
# ============================================================================

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


class DeepONet2D(nn.Module):
    """
    DeepONet for 2-D operator learning.

    branch_in_dim = N*N + 4*N   (flattened source + 4 BC arrays)
    trunk input   = (x, y)       (2-D coordinates)
    p             = latent dim   (inner-product size)
    """

    def __init__(self, branch_in_dim, p=128,
                 branch_hidden=(512, 512), trunk_hidden=(256, 256)):
        super().__init__()
        self.branch = MLP(branch_in_dim, p, hidden=branch_hidden, activation=nn.Tanh)
        self.trunk  = MLP(2, p, hidden=trunk_hidden, activation=nn.Tanh)
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
        t = self.trunk(xy_grid)                                        # (n_pts, p)
        out = torch.sum(b.unsqueeze(1) * t.unsqueeze(0), dim=-1)      # (B, n_pts)
        return out + self.bias


# ============================================================================
#  Physics-Informed Trainer
# ============================================================================

class PdeTrainer2D:
    """
    Physics-informed training loop for 2-D Poisson DeepONet.

    Loss = L_data  +  bc_weight · L_bc  +  res_weight · L_residual
    """

    def __init__(self, model, x, y, f_all, u_all, bc_all,
                 optimizer, scheduler=None,
                 loss_fn=nn.MSELoss(), batches=5, device=None,
                 f_scale=100.0, res_weight=1.0, bc_weight=10.0,
                 n_res_points=200):
        """
        Parameters
        ----------
        model      : DeepONet2D
        x, y       : 1-D grid arrays
        f_all      : (N_train, Ny, Nx)   source fields
        u_all      : (N_train, Ny, Nx)   solutions
        bc_all     : dict  {'left': (N_train, Ny), ...}
        f_scale    : multiplier in PDE  (-∇²u = f_scale * f)
        n_res_points : number of interior points subsampled for residual
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.batches = batches
        self.f_scale = float(f_scale)
        self.res_weight = float(res_weight)
        self.bc_weight = float(bc_weight)
        self.n_res_points = n_res_points

        self.Nx = len(x)
        self.Ny = len(y)
        self.N = f_all.shape[0]

        # ----- mesh bookkeeping -----
        X, Y = np.meshgrid(x, y)                                          # (Ny, Nx)
        self.xy_flat = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)  # (Nx*Ny, 2)
        self.xy_grid = torch.tensor(self.xy_flat, dtype=torch.float32,
                                    device=self.device)

        _, all_boundary = get_boundary_indices(self.Nx, self.Ny)
        self.all_boundary_idx = all_boundary
        self.interior_idx = np.setdiff1d(np.arange(self.Nx * self.Ny), all_boundary)

        # ----- store training data (float32) -----
        self.u_flat = u_all.reshape(self.N, -1).astype(np.float32)        # (N, Nx*Ny)
        f_flat_all  = f_all.reshape(self.N, -1).astype(np.float32)        # (N, Nx*Ny)
        self.f_interior = f_flat_all[:, self.interior_idx]                 # (N, n_int)

        # boundary targets from the true solution (already has correct BCs)
        self.u_boundary = self.u_flat[:, self.all_boundary_idx]            # (N, n_bnd)

        # ----- build branch inputs -----
        self.branch_inputs = self._build_branch_inputs(f_flat_all, bc_all)

    # ------------------------------------------------------------------
    def _build_branch_inputs(self, f_flat_all, bc_all):
        """[f_flat, bc_left, bc_right, bc_bottom, bc_top] per sample."""
        parts = [
            f_flat_all,
            bc_all["left"].astype(np.float32),
            bc_all["right"].astype(np.float32),
            bc_all["bottom"].astype(np.float32),
            bc_all["top"].astype(np.float32),
        ]
        return np.concatenate(parts, axis=1)                               # (N, branch_dim)

    # ------------------------------------------------------------------
    def _get_epoch_splits(self):
        idx = np.arange(self.N)
        np.random.shuffle(idx)
        return np.array_split(idx, self.batches)

    # ------------------------------------------------------------------
    def _compute_laplacian_for_sample(self, branch_single, xy_pts):
        """
        Compute  ∇²u = ∂²u/∂x² + ∂²u/∂y²  for **one** sample at given (x,y) pts.

        Uses PyTorch autograd (create_graph=True for back-prop through the residual).
        """
        b_in = branch_single.unsqueeze(0)                    # (1, branch_dim)
        xy = xy_pts.clone().detach().requires_grad_(True)     # (n, 2)

        u = self.model(b_in, xy).squeeze(0)                   # (n,)

        # first derivatives
        grads = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]   # (n, 2)
        du_dx, du_dy = grads[:, 0], grads[:, 1]

        # second derivatives
        d2u_dx2 = torch.autograd.grad(du_dx.sum(), xy, create_graph=True)[0][:, 0]
        d2u_dy2 = torch.autograd.grad(du_dy.sum(), xy, create_graph=True)[0][:, 1]

        return d2u_dx2 + d2u_dy2                              # (n,)

    # ------------------------------------------------------------------
    def train_step(self, indices):
        """One optimiser step on a mini-batch of samples."""
        self.model.train()
        idx = np.asarray(indices, dtype=int)
        B = len(idx)

        branch_b = torch.tensor(self.branch_inputs[idx],
                                dtype=torch.float32, device=self.device)
        u_b      = torch.tensor(self.u_flat[idx],
                                dtype=torch.float32, device=self.device)
        u_bc_b   = torch.tensor(self.u_boundary[idx],
                                dtype=torch.float32, device=self.device)
        f_int_b  = torch.tensor(self.f_interior[idx],
                                dtype=torch.float32, device=self.device)

        # --- forward at ALL grid points (data + boundary loss) ---
        u_pred = self.model(branch_b, self.xy_grid.detach())              # (B, n_pts)
        data_loss = self.loss_fn(u_pred, u_b)

        u_pred_bc = u_pred[:, self.all_boundary_idx]
        bc_loss   = self.loss_fn(u_pred_bc, u_bc_b)

        # --- residual at SUBSAMPLED interior points ---
        n_res = min(self.n_res_points, len(self.interior_idx))
        res_sel = np.random.choice(len(self.interior_idx), n_res, replace=False)
        grid_sel = self.interior_idx[res_sel]
        xy_res   = torch.tensor(self.xy_flat[grid_sel],
                                dtype=torch.float32, device=self.device)
        f_at_res = f_int_b[:, res_sel]                                    # (B, n_res)

        res_losses = []
        for i in range(B):
            lap = self._compute_laplacian_for_sample(branch_b[i], xy_res)
            pde_res = -lap - self.f_scale * f_at_res[i]                   # should ≈ 0
            res_losses.append(torch.mean(pde_res ** 2))
        residual_loss = torch.stack(res_losses).mean()

        # --- total loss ---
        loss = data_loss + self.bc_weight * bc_loss + self.res_weight * residual_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "data_loss": data_loss.item(),
            "bc_loss": bc_loss.item(),
            "residual_loss": residual_loss.item(),
        }

    # ------------------------------------------------------------------
    def run(self, epochs=300, verbose_freq=10,
            log_dir="./output_2d_deeponet", save_every=0):
        """Full training loop. Returns history dict."""
        os.makedirs(log_dir, exist_ok=True)
        keys = ["loss", "data_loss", "bc_loss", "residual_loss"]
        history = {k: [] for k in keys}
        start = time.time()

        for ep in range(1, epochs + 1):
            splits = self._get_epoch_splits()
            ep_stats = {k: 0.0 for k in keys}
            steps = 0

            for inds in splits:
                if len(inds) == 0:
                    continue
                stats = self.train_step(inds)
                for k in keys:
                    ep_stats[k] += stats[k]
                steps += 1

            for k in keys:
                ep_stats[k] /= max(1, steps)
                history[k].append(ep_stats[k])

            if self.scheduler is not None:
                self.scheduler.step()

            if ep % verbose_freq == 0 or ep == 1:
                lr_now = self.optimizer.param_groups[0]["lr"]
                elapsed = (time.time() - start) / 60.0
                print(f"Epoch {ep:4d}/{epochs} │ loss={ep_stats['loss']:.4e}  "
                      f"data={ep_stats['data_loss']:.4e}  bc={ep_stats['bc_loss']:.4e}  "
                      f"res={ep_stats['residual_loss']:.4e}  "
                      f"lr={lr_now:.2e}  time={elapsed:.1f}min")

            if save_every and ep % save_every == 0:
                torch.save(self.model.state_dict(),
                           os.path.join(log_dir, f"model_ep{ep}.pth"))

        # save final model + history
        torch.save(self.model.state_dict(),
                   os.path.join(log_dir, "model_final.pth"))
        try:
            import pandas as pd
            pd.DataFrame(history).to_csv(
                os.path.join(log_dir, "history.csv"), index=False)
        except ImportError:
            np.savez(os.path.join(log_dir, "history.npz"), **history)

        return history

    # ------------------------------------------------------------------
    def predict(self, f_array, bc_dict, batch_size=8):
        """
        Predict solutions for new (f, BC) pairs.

        Parameters
        ----------
        f_array : (M, Ny, Nx)
        bc_dict : dict  {'left': (M,Ny), 'right': ..., 'bottom': (M,Nx), 'top': ...}

        Returns
        -------
        u_pred : (M, Ny, Nx)
        """
        self.model.eval()
        M = f_array.shape[0]
        f_flat = f_array.reshape(M, -1).astype(np.float32)
        branch_in = np.concatenate([
            f_flat,
            bc_dict["left"].astype(np.float32),
            bc_dict["right"].astype(np.float32),
            bc_dict["bottom"].astype(np.float32),
            bc_dict["top"].astype(np.float32),
        ], axis=1)

        out_parts = []
        for i in range(0, M, batch_size):
            batch = torch.tensor(branch_in[i:i + batch_size],
                                 dtype=torch.float32, device=self.device)
            with torch.no_grad():
                pred = self.model(batch, self.xy_grid.detach())
            out_parts.append(pred.cpu().numpy())

        return np.vstack(out_parts).reshape(M, self.Ny, self.Nx)


# ============================================================================
#  Evaluation helpers
# ============================================================================

def compute_errors(u_pred, u_true):
    """Per-sample MSE and relative L2 error."""
    axes = tuple(range(1, u_true.ndim))  # sum over spatial dims
    mse = np.mean((u_pred - u_true) ** 2, axis=axes)
    rel_l2 = np.sqrt(np.sum((u_pred - u_true) ** 2, axis=axes) /
                     (np.sum(u_true ** 2, axis=axes) + 1e-12))
    return mse, rel_l2


def plot_comparison(x, y, u_true, u_pred, sample_idx=0,
                    title_prefix="", save_dir=None):
    """Side-by-side true vs predicted with error field."""
    X, Y = np.meshgrid(x, y)
    ut = u_true[sample_idx]
    up = u_pred[sample_idx]
    err = np.abs(ut - up)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # true
    c0 = axes[0].contourf(X, Y, ut, levels=50, cmap="viridis")
    axes[0].set_title(f"{title_prefix}True u(x,y)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_aspect("equal"); fig.colorbar(c0, ax=axes[0])

    # predicted
    c1 = axes[1].contourf(X, Y, up, levels=50, cmap="viridis")
    axes[1].set_title(f"{title_prefix}Predicted u(x,y)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    axes[1].set_aspect("equal"); fig.colorbar(c1, ax=axes[1])

    # error
    c2 = axes[2].contourf(X, Y, err, levels=50, cmap="hot")
    axes[2].set_title(f"{title_prefix}|Error|  (max={err.max():.4f})")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    axes[2].set_aspect("equal"); fig.colorbar(c2, ax=axes[2])

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f"comparison_sample{sample_idx}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


def plot_cross_sections(x, y, u_true, u_pred, sample_idx=0,
                        title_prefix="", save_dir=None):
    """Cross-sections at y ≈ 0.5 and x ≈ 0.5."""
    mid_y = len(y) // 2
    mid_x = len(x) // 2
    ut = u_true[sample_idx]
    up = u_pred[sample_idx]

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
        print(f"  Saved → {path}")
    plt.show()


def plot_training_history(history, save_dir=None):
    """Log-scale training curves."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in history:
        ax.plot(np.log10(np.array(history[key]) + 1e-15), label=key)
    ax.set_xlabel("Epoch"); ax.set_ylabel("log₁₀(loss)")
    ax.set_title("Training History"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "training_history.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="2-D Poisson solver via Physics-Informed DeepONet")

    # data
    parser.add_argument("--n_samples",   type=int,   default=200)
    parser.add_argument("--grid_size",   type=int,   default=31,
                        help="N for the N×N grid")
    parser.add_argument("--bc_type",     type=str,   default="four_sides",
                        choices=["constant", "two_sides", "four_sides", "varying"])
    parser.add_argument("--force_type",  type=str,   default="constant",
                        choices=["constant", "gaussian"])
    parser.add_argument("--bc_low",      type=float, default=1.0)
    parser.add_argument("--bc_high",     type=float, default=10.0)

    # model
    parser.add_argument("--p_dim",       type=int,   default=128,
                        help="Latent dimension (inner-product size)")
    parser.add_argument("--branch_h",    type=int,   nargs="+", default=[512, 512])
    parser.add_argument("--trunk_h",     type=int,   nargs="+", default=[256, 256])

    # training
    parser.add_argument("--epochs",      type=int,   default=300)
    parser.add_argument("--batch_size",  type=int,   default=10,
                        help="Number of mini-batches per epoch")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--lr_min",      type=float, default=1e-6)
    parser.add_argument("--res_weight",  type=float, default=1.0)
    parser.add_argument("--bc_weight",   type=float, default=10.0)
    parser.add_argument("--n_res_pts",   type=int,   default=200,
                        help="Interior points subsampled for residual")
    parser.add_argument("--f_scale",     type=float, default=100.0,
                        help="Source scaling factor in PDE (must match solver)")

    # IO
    parser.add_argument("--log_dir",     type=str,   default="./output_2d_deeponet")
    parser.add_argument("--save_every",  type=int,   default=100)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--n_plot",      type=int,   default=3,
                        help="Number of test samples to plot")

    args = parser.parse_args()

    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    N = args.grid_size

    # ------------------------------------------------------------------
    # 1.  Generate dataset
    # ------------------------------------------------------------------
    x, y, f_all, u_all, bc_all = generate_dataset(
        n_samples=args.n_samples, N=N,
        bc_type=args.bc_type, force_type=args.force_type,
        bc_low=args.bc_low, bc_high=args.bc_high,
        seed=args.seed,
    )

    # train / test split (80 / 20)
    n_total  = args.n_samples
    n_train  = int(0.8 * n_total)
    perm     = np.random.permutation(n_total)
    tr, te   = perm[:n_train], perm[n_train:]

    f_train, u_train = f_all[tr], u_all[tr]
    f_test,  u_test  = f_all[te], u_all[te]
    bc_train = {s: bc_all[s][tr] for s in bc_all}
    bc_test  = {s: bc_all[s][te] for s in bc_all}

    print(f"\nSplit: train={n_train}, test={n_total - n_train}")
    print(f"Solution range: [{u_all.min():.3f}, {u_all.max():.3f}]")

    # ------------------------------------------------------------------
    # 2.  Build model
    # ------------------------------------------------------------------
    branch_in_dim = N * N + 4 * N
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Branch input dim: {branch_in_dim}")

    model = DeepONet2D(
        branch_in_dim=branch_in_dim,
        p=args.p_dim,
        branch_hidden=tuple(args.branch_h),
        trunk_hidden=tuple(args.trunk_h),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min)

    # ------------------------------------------------------------------
    # 3.  Train
    # ------------------------------------------------------------------
    trainer = PdeTrainer2D(
        model=model, x=x, y=y,
        f_all=f_train, u_all=u_train, bc_all=bc_train,
        optimizer=optimizer, scheduler=scheduler,
        loss_fn=nn.MSELoss(), batches=args.batch_size,
        device=device, f_scale=args.f_scale,
        res_weight=args.res_weight, bc_weight=args.bc_weight,
        n_res_points=args.n_res_pts,
    )

    print(f"\n{'='*60}")
    print(f" Training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    history = trainer.run(
        epochs=args.epochs, verbose_freq=10,
        log_dir=args.log_dir, save_every=args.save_every,
    )

    # ------------------------------------------------------------------
    # 4.  Evaluate on test set
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" Evaluation on {len(te)} test samples")
    print(f"{'='*60}\n")

    preds_train = trainer.predict(f_train, bc_train, batch_size=8)
    preds_test  = trainer.predict(f_test,  bc_test,  batch_size=8)

    mse_train, rel_train = compute_errors(preds_train, u_train)
    mse_test,  rel_test  = compute_errors(preds_test,  u_test)

    print(f"Train  MSE   mean={mse_train.mean():.6e}  max={mse_train.max():.6e}")
    print(f"Train  relL2 mean={rel_train.mean():.6e}  max={rel_train.max():.6e}")
    print(f"Test   MSE   mean={mse_test.mean():.6e}   max={mse_test.max():.6e}")
    print(f"Test   relL2 mean={rel_test.mean():.6e}   max={rel_test.max():.6e}")

    # ------------------------------------------------------------------
    # 5.  Plots
    # ------------------------------------------------------------------
    plot_training_history(history, save_dir=args.log_dir)

    n_show = min(args.n_plot, len(te))
    for j in range(n_show):
        prefix = f"Test #{j}  MSE={mse_test[j]:.2e}  |  "
        plot_comparison(x, y, u_test, preds_test,
                        sample_idx=j, title_prefix=prefix,
                        save_dir=args.log_dir)
        plot_cross_sections(x, y, u_test, preds_test,
                            sample_idx=j, title_prefix=prefix,
                            save_dir=args.log_dir)

    print(f"\nAll outputs saved to  {os.path.abspath(args.log_dir)}/")
    print("Done.")


if __name__ == "__main__":
    main()
