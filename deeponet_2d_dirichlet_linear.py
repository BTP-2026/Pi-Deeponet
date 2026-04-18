"""
deeponet_2d_dirichlet_linear.py
================================
2D Poisson equation solver using Physics-Informed DeepONet (PyTorch).

Key insight (verified to ~10^-8):
    u(x, y, v0) = u_0(x, y)  +  v0 * (1 - x)

where u_0 is the v0=0 solution and v0*(1-x) is the EXACT harmonic function
satisfying the parametric Dirichlet BC at x=0 (u=v0), zero Dirichlet at x=1,
and zero Neumann at y=0, y=1.

Strategy
--------
Stage 1: Train a single NN for u_0(x,y) using the v0=0 COMSOL column only.
Stage 2: At inference, add v0*(1-x) analytically — zero extra error.

PDE:  -nabla^2 u = 100 * f(x,y)   on [0,1]^2
Actual BCs in data:
  Dirichlet  u = v0   at x = 0
  Dirichlet  u = 0    at x = 1
  Neumann    du/dy=0  at y = 0, 1

For v0 = 0:  u=0 on BOTH x-boundaries (same as Gaussian problem).

Usage
-----
  python deeponet_2d_dirichlet_linear.py
  python deeponet_2d_dirichlet_linear.py --epochs 40000 --grid_size 31
"""

import os
import re
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from networks import DeepONet2D
from solver import get_boundary_indices


# ============================================================================
#  Data loading
# ============================================================================

def load_dirichlet_data(data_file, forcing_file, N=31):
    """
    Load COMSOL multi-v0 data and the shared forcing field.

    Returns
    -------
    x, y       : 1-D arrays (N,)
    v0_values  : (n_v0,) Dirichlet BC values at x=0
    u_grids    : (n_v0, N, N) interpolated COMSOL solutions
    f_grid     : (N, N) raw forcing  (full COMSOL RHS, peak ~1000)
    u0_grid    : (N, N) solution for v0=0
    """
    # --- Parse header to extract v0 values ---
    v0_values = []
    with open(data_file, 'r') as fh:
        for line in fh:
            if not line.startswith('%'):
                break
            matches = re.findall(r'v0=([0-9.]+)', line)
            v0_values.extend(float(m) for m in matches)
    v0_values = np.array(v0_values)
    n_v0 = len(v0_values)
    print(f"Found {n_v0} v0 values: {v0_values[0]:.1f} to {v0_values[-1]:.1f}")

    # --- Load scattered data ---
    raw = np.loadtxt(data_file, comments='%')   # (n_pts, 2 + n_v0)
    xy_pts = raw[:, :2]
    u_pts  = raw[:, 2:]

    # --- Regular grid ---
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    # --- Interpolate each v0 solution ---
    u_grids = np.zeros((n_v0, N, N), dtype=np.float32)
    for i in range(n_v0):
        grid = griddata(xy_pts, u_pts[:, i], (X, Y), method='cubic')
        mask = np.isnan(grid)
        if mask.any():
            grid[mask] = griddata(xy_pts, u_pts[:, i],
                                  (X[mask], Y[mask]), method='nearest')
        u_grids[i] = grid.astype(np.float32)

    # --- Load forcing field ---
    f_data = np.loadtxt(forcing_file, comments='%')   # X Y F
    f_grid = griddata(f_data[:, :2], f_data[:, 2], (X, Y), method='cubic')
    fmask = np.isnan(f_grid)
    if fmask.any():
        f_grid[fmask] = griddata(f_data[:, :2], f_data[:, 2],
                                 (X[fmask], Y[fmask]), method='nearest')

    u0_grid = u_grids[0]   # v0=0 solution
    print(f"Grid {N}x{N}  |  f range [{f_grid.min():.2f}, {f_grid.max():.2f}]")
    print(f"u0 range [{u0_grid.min():.4f}, {u0_grid.max():.4f}]")

    return x, y, v0_values, u_grids, f_grid.astype(np.float32), u0_grid


def sanity_check_linearity(x, v0_values, u_grids, tol=1e-1):
    """
    Verify u(x,y,v0) ≈ u_0(x,y) + v0*(1-x) for all v0.
    Prints max residual for each v0.
    """
    X = np.tile(x, (len(x), 1))   # (Ny, Nx)
    u0 = u_grids[0]
    print("\nLinearity sanity check:  max|u(v0) - u0 - v0*(1-x)|")
    print(f"{'v0':>6}  {'max_residual':>14}  {'rel_residual':>14}")
    print("-" * 40)
    ok = True
    for i, v0 in enumerate(v0_values):
        if v0 == 0.0:
            continue
        u_reconstructed = u0 + v0 * (1.0 - X)
        residual = np.abs(u_grids[i] - u_reconstructed)
        max_res = residual.max()
        rel_res = max_res / (np.abs(u_grids[i]).max() + 1e-8)
        flag = "  OK" if max_res < tol else "  WARN"
        print(f"{v0:6.1f}  {max_res:14.6e}  {rel_res:14.6e}{flag}")
        if max_res >= tol:
            ok = False
    if ok:
        print("=> Linearity verified for all v0.\n")
    else:
        print("=> Some v0 exceeded tolerance — check data.\n")


# ============================================================================
#  Physics-Informed Trainer for u_0(x,y)
# ============================================================================

class DirichletLinearTrainer:
    """
    Trains a single-sample DeepONet to approximate u_0(x,y) — the v0=0 solution.

    Loss = data_loss
         + bc_d_weight * dirichlet_loss   (u=0 on x=0 and x=1)
         + bc_n_weight * neumann_loss     (du/dy=0 on y=0 and y=1)
    """

    def __init__(self, model, x, y, f_raw, u0_true, *,
                 optimizer, scheduler=None,
                 device=None, f_scale=100.0,
                 bc_d_weight=10.0, bc_n_weight=5.0,
                 grad_clip=1.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.f_scale = float(f_scale)
        self.bc_d_weight = float(bc_d_weight)
        self.bc_n_weight = float(bc_n_weight)
        self.grad_clip   = float(grad_clip)

        self.Nx, self.Ny = len(x), len(y)
        N2 = self.Nx * self.Ny

        # Mesh flat arrays
        X, Y = np.meshgrid(x, y)
        self.X_grid = X.astype(np.float32)   # (Ny, Nx) — for v0*(1-x) at inference
        self.xy_flat_np = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
        self.xy_grid = torch.tensor(self.xy_flat_np, dtype=torch.float32,
                                    device=self.device)

        # Branch input: flattened forcing only (N*N dims, no v0)
        branch_np = f_raw.ravel().astype(np.float32)
        self.branch_input = torch.tensor(branch_np, dtype=torch.float32,
                                         device=self.device).unsqueeze(0)  # (1, N*N)

        # Target normalisation by max|u_0|
        self.u_scale = float(np.abs(u0_true).max()) + 1e-8
        u_norm = (u0_true / self.u_scale).ravel().astype(np.float32)
        self.u_target = torch.tensor(u_norm, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)  # (1, N*N)

        # Boundary indices
        bc_idx, _ = get_boundary_indices(self.Nx, self.Ny)
        self.dirichlet_idx = np.union1d(bc_idx['left'], bc_idx['right'])
        neumann_all = np.union1d(bc_idx['bottom'], bc_idx['top'])
        self.neumann_idx = np.setdiff1d(neumann_all, self.dirichlet_idx)

        # Dirichlet target: u_0 = 0 on x=0 and x=1
        self.u_bc_d = torch.zeros(1, len(self.dirichlet_idx),
                                  dtype=torch.float32, device=self.device)

        # Neumann boundary coordinates (static)
        self.neumann_coords_np = self.xy_flat_np[self.neumann_idx]

    # ------------------------------------------------------------------
    def train_step(self):
        self.model.train()
        loss_fn = nn.MSELoss()

        # Pass 1: all grid points — data + Dirichlet
        u_pred_all = self.model(self.branch_input,
                                self.xy_grid.detach())       # (1, N*N)
        data_loss      = loss_fn(u_pred_all, self.u_target)
        dirichlet_loss = loss_fn(u_pred_all[:, self.dirichlet_idx], self.u_bc_d)

        # Pass 2: Neumann boundary with autograd
        xy_n = torch.tensor(self.neumann_coords_np, dtype=torch.float32,
                            device=self.device).requires_grad_(True)
        u_n  = self.model(self.branch_input, xy_n).squeeze(0)
        grad_n   = torch.autograd.grad(u_n.sum(), xy_n, create_graph=True)[0]
        du_dy_n  = grad_n[:, 1]
        neumann_loss = torch.mean(du_dy_n ** 2)

        loss = (data_loss
                + self.bc_d_weight * dirichlet_loss
                + self.bc_n_weight * neumann_loss)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "loss":           loss.item(),
            "data_loss":      data_loss.item(),
            "dirichlet_loss": dirichlet_loss.item(),
            "neumann_loss":   neumann_loss.item(),
        }

    # ------------------------------------------------------------------
    def run(self, epochs=40000, verbose_freq=500,
            log_dir="./output_dirichlet_linear", save_every=0):
        os.makedirs(log_dir, exist_ok=True)
        keys = ["loss", "data_loss", "dirichlet_loss", "neumann_loss"]
        history = {k: [] for k in keys}
        start = time.time()
        best_data_loss = float("inf")

        for ep in range(1, epochs + 1):
            stats = self.train_step()
            for k in keys:
                history[k].append(stats[k])

            if self.scheduler is not None:
                self.scheduler.step()

            if stats["data_loss"] < best_data_loss:
                best_data_loss = stats["data_loss"]
                torch.save(self.model.state_dict(),
                           os.path.join(log_dir, "model_best.pth"))

            if ep % verbose_freq == 0 or ep == 1:
                lr_now = self.optimizer.param_groups[0]["lr"]
                elapsed = (time.time() - start) / 60.0
                print(f"Epoch {ep:5d}/{epochs} | "
                      f"loss={stats['loss']:.4e}  "
                      f"data={stats['data_loss']:.4e}  "
                      f"dir={stats['dirichlet_loss']:.4e}  "
                      f"neu={stats['neumann_loss']:.4e}  "
                      f"lr={lr_now:.2e}  time={elapsed:.1f}min",
                      flush=True)

            if save_every and ep % save_every == 0:
                torch.save(self.model.state_dict(),
                           os.path.join(log_dir, f"model_ep{ep}.pth"))

        print(f"\nBest data_loss: {best_data_loss:.4e}  (saved to model_best.pth)")
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
    def predict_u0(self):
        """Return u_0(x,y) prediction on the full grid — shape (Ny, Nx)."""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.branch_input, self.xy_grid.detach())
        return pred.squeeze(0).cpu().numpy().reshape(self.Ny, self.Nx) * self.u_scale

    def predict(self, v0):
        """
        Full prediction: u(x, y, v0) = u_0(x, y) + v0 * (1 - x).
        Returns (Ny, Nx) array in physical units.
        """
        u0 = self.predict_u0()
        return u0 + float(v0) * (1.0 - self.X_grid)


# ============================================================================
#  Plotting helpers
# ============================================================================

def plot_three_panel(x, y, u_ref, u_pred, title_suffix="",
                     save_dir=None, fname="comparison.png"):
    X, Y = np.meshgrid(x, y)
    err  = np.abs(u_ref - u_pred)
    mse  = float(np.mean((u_ref - u_pred) ** 2))
    rel  = float(np.sqrt(np.sum((u_ref - u_pred) ** 2) /
                         (np.sum(u_ref ** 2) + 1e-12)))
    vmin, vmax = u_ref.min(), u_ref.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    c0 = axes[0].contourf(X, Y, u_ref, levels=64, cmap="jet", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"COMSOL reference  {title_suffix}")
    axes[0].set_aspect("equal"); plt.colorbar(c0, ax=axes[0])
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    c1 = axes[1].contourf(X, Y, u_pred, levels=64, cmap="jet", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"DeepONet+linear  (rel-L2={rel:.4e})")
    axes[1].set_aspect("equal"); plt.colorbar(c1, ax=axes[1])
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

    c2 = axes[2].contourf(X, Y, err, levels=64, cmap="hot")
    axes[2].set_title(f"Absolute error  (max={err.max():.4e})")
    axes[2].set_aspect("equal"); plt.colorbar(c2, ax=axes[2])
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.close(fig)


def plot_error_bars(v0_values, rel_errors, abs_errors_max,
                    save_dir=None, fname="error_vs_v0.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))

    ax1.bar(v0_values, rel_errors * 100, color="steelblue", alpha=0.8)
    ax1.axhline(1.0, color="red", linestyle="--", linewidth=1, label="1% threshold")
    ax1.set_xlabel("v0 (Dirichlet BC at x=0)")
    ax1.set_ylabel("Rel-L2 error (%)")
    ax1.set_title("Rel-L2 error vs COMSOL per v0")
    ax1.legend(); ax1.grid(axis='y', alpha=0.4)

    ax2.bar(v0_values, abs_errors_max, color="darkorange", alpha=0.8)
    ax2.axhline(0.01, color="red", linestyle="--", linewidth=1, label="Target: 0.01")
    ax2.set_xlabel("v0 (Dirichlet BC at x=0)")
    ax2.set_ylabel("Max absolute error")
    ax2.set_title("Max absolute error vs COMSOL per v0")
    ax2.legend(); ax2.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.close(fig)


def plot_training_history(history, save_dir=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, vals in history.items():
        ax.semilogy(vals, label=key)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training History"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "training_history.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.close(fig)


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",    type=str,   default="data_v0.txt")
    parser.add_argument("--forcing_file", type=str,   default="Surface_Solution.txt")
    parser.add_argument("--grid_size",    type=int,   default=31)
    parser.add_argument("--p_dim",        type=int,   default=512)
    parser.add_argument("--branch_h",     type=int,   nargs="+", default=[256, 256])
    parser.add_argument("--trunk_h",      type=int,   nargs="+", default=[512, 512, 512])
    parser.add_argument("--n_fourier",    type=int,   default=8)
    parser.add_argument("--epochs",       type=int,   default=40000)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--bc_d_weight",  type=float, default=10.0)
    parser.add_argument("--bc_n_weight",  type=float, default=5.0)
    parser.add_argument("--grad_clip",    type=float, default=1.0,
                        help="Max gradient norm (0 = disabled)")
    parser.add_argument("--f_scale",      type=float, default=100.0)
    parser.add_argument("--log_dir",      type=str,   default="./output_dirichlet_linear")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--skip_linearity_check", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    N = args.grid_size
    os.makedirs(args.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1.  Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print(" Loading data")
    print("=" * 60)
    x, y, v0_values, u_grids, f_comsol, u0_grid = load_dirichlet_data(
        args.data_file, args.forcing_file, N=N)
    f_raw = f_comsol / args.f_scale   # raw forcing, peak ~10

    # ------------------------------------------------------------------
    # 2.  Sanity-check linearity
    # ------------------------------------------------------------------
    if not args.skip_linearity_check:
        sanity_check_linearity(x, v0_values, u_grids, tol=0.1)

    # ------------------------------------------------------------------
    # 3.  Build model
    # ------------------------------------------------------------------
    branch_in_dim = N * N   # forcing only, no v0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}  |  branch_in_dim: {branch_in_dim}")

    model = DeepONet2D(
        branch_in_dim=branch_in_dim,
        p=args.p_dim,
        branch_hidden=tuple(args.branch_h),
        trunk_hidden=tuple(args.trunk_h),
        activation=nn.SiLU,
        use_fourier=True,
        n_fourier=args.n_fourier,
    )
    trunk_in = 2 + 4 * args.n_fourier
    print(f"Fourier encoding: n_fourier={args.n_fourier}  trunk_in_dim={trunk_in}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ------------------------------------------------------------------
    # 4.  Train on u_0 only
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f" Training for u_0  ({args.epochs} epochs)")
    print(f"{'=' * 60}\n")

    trainer = DirichletLinearTrainer(
        model=model, x=x, y=y,
        f_raw=f_raw, u0_true=u0_grid,
        optimizer=optimizer, scheduler=scheduler,
        device=device, f_scale=args.f_scale,
        bc_d_weight=args.bc_d_weight,
        bc_n_weight=args.bc_n_weight,
        grad_clip=args.grad_clip,
    )

    history = trainer.run(
        epochs=args.epochs, verbose_freq=500,
        log_dir=args.log_dir, save_every=0,
    )

    # ------------------------------------------------------------------
    # 5.  Evaluate across all v0
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f" Evaluation: loading best checkpoint")
    print(f"{'=' * 60}\n")

    best_ckpt = os.path.join(args.log_dir, "model_best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print("Loaded best checkpoint.")

    rel_errors     = np.zeros(len(v0_values))
    abs_errors_max = np.zeros(len(v0_values))
    abs_errors_mean = np.zeros(len(v0_values))

    X_grid = np.tile(x, (N, 1))   # (Ny, Nx)

    print(f"\n{'v0':>6}  {'rel-L2 (%)':>12}  {'max|err|':>12}  {'mean|err|':>12}")
    print("-" * 50)
    for i, v0 in enumerate(v0_values):
        u_pred = trainer.predict(v0)
        u_ref  = u_grids[i]
        err    = np.abs(u_pred - u_ref)
        rel    = float(np.sqrt(np.sum((u_pred - u_ref) ** 2) /
                               (np.sum(u_ref ** 2) + 1e-12)))
        rel_errors[i]      = rel
        abs_errors_max[i]  = err.max()
        abs_errors_mean[i] = err.mean()
        flag = " <== PASS" if err.max() < 0.01 else ""
        print(f"{v0:6.1f}  {rel*100:12.4f}  {err.max():12.6f}  {err.mean():12.6f}{flag}")

    print(f"\nSummary:")
    print(f"  Mean rel-L2  : {rel_errors.mean()*100:.4f}%")
    print(f"  Max  rel-L2  : {rel_errors.max()*100:.4f}%")
    print(f"  Mean max|err|: {abs_errors_max.mean():.6f}")
    print(f"  Overall max  : {abs_errors_max.max():.6f}  "
          f"({'PASS' if abs_errors_max.max() < 0.01 else 'FAIL'} target 0.01)")

    # ------------------------------------------------------------------
    # 6.  Plots
    # ------------------------------------------------------------------
    plot_training_history(history, save_dir=args.log_dir)
    plot_error_bars(v0_values, rel_errors, abs_errors_max, save_dir=args.log_dir)

    for v0_plot in [0.0, 10.0, 20.0]:
        idx = int(np.argmin(np.abs(v0_values - v0_plot)))
        v0_actual = v0_values[idx]
        u_pred = trainer.predict(v0_actual)
        u_ref  = u_grids[idx]
        plot_three_panel(
            x, y, u_ref, u_pred,
            title_suffix=f"v0={v0_actual:.1f}",
            save_dir=args.log_dir,
            fname=f"comparison_v0_{v0_actual:.1f}.png",
        )

    print(f"\nAll outputs saved to  {os.path.abspath(args.log_dir)}/")
    print("Done.")


if __name__ == "__main__":
    main()
