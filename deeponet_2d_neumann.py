"""
deeponet_2d_neumann.py
======================
2D Poisson equation solver using Physics-Informed DeepONet (PyTorch)
with a fixed tail-end Gaussian forcing and variable Neumann boundary conditions.

PDE:  -nabla^2 u = 100 * f(x,y)   on [0,1] x [0,1]
      f(x,y) = (rho0/100) * exp(-((x - mu_x)^2 + (y - mu_y)^2) / (2 * sigma^2))

Boundary conditions
-------------------
  Dirichlet:  u = 0       on x = 0  and  x = 1
  Neumann:    du/dy = v0  on y = 0  and  y = 1    (v0 varies per sample)

Data source
-----------
  data_v0.txt   — COMSOL solutions for v0 = 0, 0.5, 1, ..., 20 (41 samples)
                  Columns: X  Y  u@v0=0  u@v0=0.5  ...  u@v0=20
  Surface_Solution.txt  — COMSOL forcing function (shared across all samples)

Usage
-----
  python deeponet_2d_neumann.py
  python deeponet_2d_neumann.py --grid_size 31 --epochs 10000
"""

import os
import re
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from networks import DeepONet2D
from eval_utils import plot_training_history
from solver import get_boundary_indices


# ============================================================================
#  Data loading
# ============================================================================

def load_neumann_data(data_file, forcing_file, N=31):
    """
    Load COMSOL multi-v0 data and the shared forcing field.

    Parameters
    ----------
    data_file    : path to data_v0.txt (columns: X Y u@v0=0 u@v0=0.5 ...)
    forcing_file : path to Surface_Solution.txt (X Y F)
    N            : grid size for interpolation

    Returns
    -------
    x, y         : 1-D arrays (N,)
    v0_values    : (n_v0,) array of Neumann BC values
    u_grids      : (n_v0, N, N) interpolated COMSOL solutions
    f_grid       : (N, N) forcing field (raw, i.e. full COMSOL RHS)
    """
    # --- Parse header to extract v0 values ---
    v0_values = []
    with open(data_file, 'r') as fh:
        for line in fh:
            if not line.startswith('%'):
                break
            # Look for lines like: % u (V) @ v0=0.5
            matches = re.findall(r'v0=([0-9.]+)', line)
            v0_values.extend(float(m) for m in matches)
    v0_values = np.array(v0_values)
    n_v0 = len(v0_values)
    print(f"Found {n_v0} v0 values: {v0_values[0]:.1f} to {v0_values[-1]:.1f}")

    # --- Load scattered data ---
    raw = np.loadtxt(data_file, comments='%')  # (n_pts, 2 + n_v0)
    xy_pts = raw[:, :2]      # (n_pts, 2)
    u_pts  = raw[:, 2:]      # (n_pts, n_v0)

    # --- Regular grid ---
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    # --- Interpolate each v0 solution onto the grid ---
    u_grids = np.zeros((n_v0, N, N), dtype=np.float32)
    for i in range(n_v0):
        grid = griddata(xy_pts, u_pts[:, i], (X, Y), method='cubic')
        # Fill NaN edges with nearest-neighbour
        mask = np.isnan(grid)
        if mask.any():
            fill = griddata(xy_pts, u_pts[:, i], (X[mask], Y[mask]),
                            method='nearest')
            grid[mask] = fill
        u_grids[i] = grid.astype(np.float32)

    # --- Load forcing field ---
    f_data = np.loadtxt(forcing_file, comments='%')  # (n_pts, 3): X Y F
    f_grid = griddata(f_data[:, :2], f_data[:, 2], (X, Y), method='cubic')
    fmask = np.isnan(f_grid)
    if fmask.any():
        fill = griddata(f_data[:, :2], f_data[:, 2],
                        (X[fmask], Y[fmask]), method='nearest')
        f_grid[fmask] = fill

    print(f"Grid {N}x{N}  |  f range [{f_grid.min():.4f}, {f_grid.max():.4f}]")
    print(f"u range across all v0: [{u_grids.min():.4f}, {u_grids.max():.4f}]")

    return x, y, v0_values, u_grids, f_grid.astype(np.float32)


# ============================================================================
#  Physics-Informed Trainer (variable Neumann BCs)
# ============================================================================

class NeumannPdeTrainer:
    """
    Physics-informed training loop for the Gaussian-forcing Poisson problem
    with variable Neumann boundary conditions du/dy = v0.

    Loss = L_data
         + bc_d_weight * L_dirichlet
         + bc_n_weight * L_neumann
         + res_weight  * L_residual   (disabled by default)

    Branch input = [f_flat, v0]  (dim = N*N + 1)
    """

    def __init__(self, model, x, y, v0_values, u_grids, f_raw, *,
                 optimizer, scheduler=None,
                 device=None, f_scale=100.0,
                 res_weight=0.0, bc_d_weight=10.0, bc_n_weight=5.0,
                 batch_size=None):
        """
        Parameters
        ----------
        model      : DeepONet2D
        x, y       : 1-D grid arrays (N,)
        v0_values  : (n_v0,) Neumann BC values
        u_grids    : (n_v0, N, N) COMSOL reference solutions
        f_raw      : (N, N) raw source (solver convention, f_scale*f_raw = actual RHS)
        f_scale    : PDE multiplier (-nabla^2 u = f_scale * f_raw)
        batch_size : samples per mini-batch (None = full batch)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.f_scale = float(f_scale)
        self.res_weight = float(res_weight)
        self.bc_d_weight = float(bc_d_weight)
        self.bc_n_weight = float(bc_n_weight)

        self.Nx, self.Ny = len(x), len(y)
        self.n_v0 = len(v0_values)
        self.v0_values = v0_values
        self.batch_size = batch_size or self.n_v0

        N2 = self.Nx * self.Ny

        # Mesh
        X, Y = np.meshgrid(x, y)
        self.xy_flat_np = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
        self.xy_grid = torch.tensor(self.xy_flat_np, dtype=torch.float32,
                                    device=self.device)

        # --- Branch inputs: [f_flat, v0] for each sample ---
        f_flat = f_raw.ravel().astype(np.float32)  # (N*N,)
        branch_np = np.zeros((self.n_v0, N2 + 1), dtype=np.float32)
        for i, v0 in enumerate(v0_values):
            branch_np[i, :N2] = f_flat
            branch_np[i, N2]  = float(v0)
        self.branch_inputs = torch.tensor(branch_np, dtype=torch.float32,
                                          device=self.device)  # (n_v0, N*N+1)

        # --- Per-sample normalised targets ---
        # Normalise each u to [0,1] range using per-sample max|u|
        self.u_scales = np.array([float(np.abs(u_grids[i]).max()) + 1e-8
                                   for i in range(self.n_v0)])
        u_norm_np = np.zeros((self.n_v0, N2), dtype=np.float32)
        for i in range(self.n_v0):
            u_norm_np[i] = (u_grids[i] / self.u_scales[i]).ravel()
        self.u_targets = torch.tensor(u_norm_np, dtype=torch.float32,
                                      device=self.device)  # (n_v0, N*N)

        # v0 values as tensor for Neumann loss computation
        self.v0_tensor = torch.tensor(v0_values, dtype=torch.float32,
                                      device=self.device)  # (n_v0,)

        # --- Boundary bookkeeping ---
        bc_idx, _ = get_boundary_indices(self.Nx, self.Ny)

        self.dirichlet_idx = np.union1d(bc_idx['left'], bc_idx['right'])
        neumann_all = np.union1d(bc_idx['bottom'], bc_idx['top'])
        self.neumann_idx = np.setdiff1d(neumann_all, self.dirichlet_idx)

        # Dirichlet target: u = 0 on left/right walls
        # shape (1, n_dirichlet) — broadcast over batch
        self.u_bc_d = torch.zeros(1, len(self.dirichlet_idx),
                                  dtype=torch.float32, device=self.device)

        # Neumann boundary coords (static, no grad needed here)
        self.neumann_coords_np = self.xy_flat_np[self.neumann_idx]  # (n_neu, 2)

    # ------------------------------------------------------------------
    def _sample_batch_idx(self):
        """Return random batch indices into the n_v0 dimension."""
        if self.batch_size >= self.n_v0:
            return np.arange(self.n_v0)
        return np.random.choice(self.n_v0, self.batch_size, replace=False)

    # ------------------------------------------------------------------
    def train_step(self):
        """One optimiser step over a mini-batch of v0 samples."""
        self.model.train()
        loss_fn = nn.MSELoss()

        bidx = self._sample_batch_idx()
        B = len(bidx)

        branch_b  = self.branch_inputs[bidx]    # (B, N*N+1)
        target_b  = self.u_targets[bidx]         # (B, N*N)
        v0_b      = self.v0_tensor[bidx]         # (B,)
        uscale_b  = torch.tensor(self.u_scales[bidx], dtype=torch.float32,
                                 device=self.device)  # (B,)

        # --- Pass 1: all grid points (detached) — data + Dirichlet BC ---
        u_pred_all = self.model(branch_b, self.xy_grid.detach())  # (B, N*N)

        data_loss = loss_fn(u_pred_all, target_b)

        # Dirichlet: u = 0 at x=0, x=1 (in normalised space, u_scale cancels)
        dirichlet_loss = loss_fn(u_pred_all[:, self.dirichlet_idx],
                                 self.u_bc_d.expand(B, -1))

        # --- Pass 2: Neumann boundary points (per-sample grad) ---
        # Enforce du_norm/dy = v0 / u_scale at y=0, y=1 for each sample.
        neumann_loss = self._neumann_loss_per_batch(branch_b, v0_b, uscale_b)

        # --- Residual loss (disabled by default) ---
        if self.res_weight > 0:
            residual_loss = self._residual_loss(branch_b)
        else:
            residual_loss = torch.tensor(0.0, device=self.device)

        # --- Total loss ---
        loss = (data_loss
                + self.bc_d_weight * dirichlet_loss
                + self.bc_n_weight * neumann_loss
                + self.res_weight  * residual_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "data_loss": data_loss.item(),
            "dirichlet_loss": dirichlet_loss.item(),
            "neumann_loss": neumann_loss.item(),
            "residual_loss": residual_loss.item(),
        }

    def _neumann_loss_per_batch(self, branch_b, v0_b, uscale_b):
        """
        Compute Neumann BC loss: mean over samples and boundary points of
        (du_norm/dy - v0/u_scale)^2.

        We need per-sample gradients, so we iterate over the batch.
        For small batch sizes (<=41) this is acceptable.
        """
        B = branch_b.shape[0]
        n_neu = self.neumann_idx.shape[0]
        total = torch.tensor(0.0, device=self.device)

        for i in range(B):
            b_i = branch_b[i:i+1]   # (1, N*N+1)
            xy_i = torch.tensor(self.neumann_coords_np,
                                dtype=torch.float32,
                                device=self.device).requires_grad_(True)
            u_i = self.model(b_i, xy_i).squeeze(0)  # (n_neu,)
            grad_i = torch.autograd.grad(u_i.sum(), xy_i,
                                         create_graph=True)[0]   # (n_neu, 2)
            du_dy_i = grad_i[:, 1]   # (n_neu,)
            target_i = v0_b[i] / uscale_b[i]
            total = total + torch.mean((du_dy_i - target_i) ** 2)

        return total / B

    def _residual_loss(self, branch_b):
        """PDE residual loss (optional, disabled by default)."""
        # Not needed since COMSOL targets already satisfy the PDE
        return torch.tensor(0.0, device=self.device)

    # ------------------------------------------------------------------
    def run(self, epochs=5000, verbose_freq=50,
            log_dir="./output_2d_neumann", save_every=0):
        """Full training loop. Returns history dict."""
        os.makedirs(log_dir, exist_ok=True)
        keys = ["loss", "data_loss", "dirichlet_loss",
                "neumann_loss", "residual_loss"]
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
                      f"lr={lr_now:.2e}  time={elapsed:.1f}min")

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
    def predict(self, f_raw, v0):
        """
        Predict solution for a given forcing field and Neumann BC value.

        Parameters
        ----------
        f_raw : (N, N) forcing field (same scale as training)
        v0    : scalar Neumann BC value

        Returns
        -------
        u_pred : (N, N) predicted solution (physical units)
        """
        self.model.eval()
        f_flat = f_raw.ravel().astype(np.float32)
        branch = np.concatenate([f_flat, [float(v0)]]).astype(np.float32)
        b_in = torch.tensor(branch, dtype=torch.float32,
                            device=self.device).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(b_in, self.xy_grid.detach())

        pred_np = pred.squeeze(0).cpu().numpy().reshape(self.Ny, self.Nx)

        # Find the u_scale for this v0 (closest training sample)
        idx = int(np.argmin(np.abs(self.v0_values - v0)))
        u_scale = self.u_scales[idx]

        return pred_np * u_scale


# ============================================================================
#  Plotting
# ============================================================================

def plot_three_panel(x, y, u_ref, u_pred, save_dir=None,
                     fname="comparison_3panel.png",
                     ref_label="COMSOL (interpolated)"):
    """3-panel: Reference | Prediction | Absolute error."""
    import matplotlib
    matplotlib.use("Agg")

    X, Y = np.meshgrid(x, y)
    err = np.abs(u_ref - u_pred)
    mse = np.mean((u_ref - u_pred) ** 2)
    rel = np.sqrt(np.sum((u_ref - u_pred) ** 2) /
                  (np.sum(u_ref ** 2) + 1e-12))

    vmin, vmax = u_ref.min(), u_ref.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    c0 = axes[0].contourf(X, Y, u_ref, levels=64, cmap="jet",
                          vmin=vmin, vmax=vmax)
    axes[0].set_title(ref_label, fontsize=12)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    plt.colorbar(c0, ax=axes[0])

    c1 = axes[1].contourf(X, Y, u_pred, levels=64, cmap="jet",
                          vmin=vmin, vmax=vmax)
    axes[1].set_title(f"DeepONet  (MSE={mse:.3e}, rel-L2={rel:.3e})", fontsize=12)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    plt.colorbar(c1, ax=axes[1])

    c2 = axes[2].contourf(X, Y, err, levels=64, cmap="hot")
    axes[2].set_title(f"Absolute Error  (max={err.max():.3e})", fontsize=12)
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    plt.colorbar(c2, ax=axes[2])

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.close(fig)


def plot_error_summary(v0_values, rel_errors, save_dir=None,
                       fname="error_vs_v0.png"):
    """Bar chart of rel-L2 error for each v0."""
    import matplotlib
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(v0_values, rel_errors * 100, color="steelblue", alpha=0.8)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="1% threshold")
    ax.set_xlabel("v0 (Neumann BC value)")
    ax.set_ylabel("Rel-L2 error (%)")
    ax.set_title("DeepONet rel-L2 error vs COMSOL for each v0")
    ax.legend()
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.close(fig)


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="2-D Poisson (Gaussian forcing, variable Neumann BC) via "
                    "Physics-Informed DeepONet")

    # data
    parser.add_argument("--data_file",    type=str, default="data_v0.txt")
    parser.add_argument("--forcing_file", type=str, default="Surface_Solution.txt")
    parser.add_argument("--grid_size",    type=int, default=31)

    # model
    parser.add_argument("--p_dim",       type=int, default=512)
    parser.add_argument("--branch_h",    type=int, nargs="+", default=[256, 256])
    parser.add_argument("--trunk_h",     type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--use_fourier", action="store_true", default=True)
    parser.add_argument("--no_fourier",  dest="use_fourier", action="store_false")
    parser.add_argument("--n_fourier",   type=int, default=8)

    # training
    parser.add_argument("--epochs",      type=int,   default=10000)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--res_weight",  type=float, default=0.0)
    parser.add_argument("--bc_d_weight", type=float, default=10.0)
    parser.add_argument("--bc_n_weight", type=float, default=5.0)
    parser.add_argument("--batch_size",  type=int,   default=None,
                        help="Samples per mini-batch (default: full batch of 41)")
    parser.add_argument("--f_scale",     type=float, default=100.0)

    # IO
    parser.add_argument("--log_dir",    type=str, default="./output_2d_neumann")
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--seed",       type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    N = args.grid_size

    # ------------------------------------------------------------------
    # 1.  Load data
    # ------------------------------------------------------------------
    print("Loading data ...")
    x, y, v0_values, u_grids, f_comsol = load_neumann_data(
        args.data_file, args.forcing_file, N=N)

    f_raw = f_comsol / args.f_scale   # raw forcing (peak ~ 10)

    # ------------------------------------------------------------------
    # 2.  Build model
    # ------------------------------------------------------------------
    branch_in_dim = N * N + 1   # flattened forcing + v0 scalar
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}  |  Branch input dim: {branch_in_dim}")

    model = DeepONet2D(
        branch_in_dim=branch_in_dim,
        p=args.p_dim,
        branch_hidden=tuple(args.branch_h),
        trunk_hidden=tuple(args.trunk_h),
        activation=nn.SiLU,
        use_fourier=args.use_fourier,
        n_fourier=args.n_fourier,
    )
    if args.use_fourier:
        print(f"Fourier encoding: {args.n_fourier} bands → trunk dim = "
              f"{2 + 4*args.n_fourier}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ------------------------------------------------------------------
    # 3.  Train
    # ------------------------------------------------------------------
    trainer = NeumannPdeTrainer(
        model=model, x=x, y=y,
        v0_values=v0_values, u_grids=u_grids,
        f_raw=f_raw,
        optimizer=optimizer, scheduler=scheduler,
        device=device, f_scale=args.f_scale,
        res_weight=args.res_weight,
        bc_d_weight=args.bc_d_weight,
        bc_n_weight=args.bc_n_weight,
        batch_size=args.batch_size,
    )

    print(f"\n{'='*60}")
    print(f" Training  |  {len(v0_values)} samples  |  {args.epochs} epochs")
    print(f"{'='*60}\n")

    history = trainer.run(
        epochs=args.epochs, verbose_freq=50,
        log_dir=args.log_dir, save_every=args.save_every,
    )

    # ------------------------------------------------------------------
    # 4.  Evaluate
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" Evaluation")
    print(f"{'='*60}\n")

    # Load best checkpoint
    best_ckpt = os.path.join(args.log_dir, "model_best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print("Loaded best checkpoint for evaluation.")

    rel_errors = np.zeros(len(v0_values))
    for i, v0 in enumerate(v0_values):
        u_pred = trainer.predict(f_raw, v0)
        u_ref  = u_grids[i]
        rel = np.sqrt(np.sum((u_pred - u_ref) ** 2) /
                      (np.sum(u_ref ** 2) + 1e-12))
        rel_errors[i] = rel

    print("\nRel-L2 error vs COMSOL per v0:")
    print(f"{'v0':>6}  {'rel-L2 (%)':>12}")
    print("-" * 22)
    for v0, err in zip(v0_values, rel_errors):
        print(f"{v0:6.1f}  {err*100:12.4f}")
    print(f"\nMean rel-L2: {rel_errors.mean()*100:.4f}%  "
          f"Max: {rel_errors.max()*100:.4f}%")

    # ------------------------------------------------------------------
    # 5.  Plots
    # ------------------------------------------------------------------
    plot_training_history(history, save_dir=args.log_dir)
    plot_error_summary(v0_values, rel_errors, save_dir=args.log_dir)

    # 3-panel plots for selected v0 values
    for v0_plot in [0.0, 5.0, 10.0, 15.0, 20.0]:
        idx = int(np.argmin(np.abs(v0_values - v0_plot)))
        v0_actual = v0_values[idx]
        u_pred = trainer.predict(f_raw, v0_actual)
        u_ref  = u_grids[idx]
        plot_three_panel(x, y, u_ref, u_pred,
                         save_dir=args.log_dir,
                         fname=f"comparison_v0_{v0_actual:.1f}.png",
                         ref_label=f"COMSOL  v0={v0_actual:.1f}")

    print(f"\nAll outputs saved to  {os.path.abspath(args.log_dir)}/")
    print("Done.")


if __name__ == "__main__":
    main()
