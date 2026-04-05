"""
deeponet_2d_gaussian.py
=======================
2D Poisson equation solver using Physics-Informed DeepONet (PyTorch)
with a fixed tail-end Gaussian forcing and mixed boundary conditions.

PDE:  -nabla^2 u = 100 * f(x,y)   on [0,1] x [0,1]
      f(x,y) = (rho0/100) * exp(-((x - mu_x)^2 + (y - mu_y)^2) / (2 * sigma^2))

Boundary conditions
-------------------
  Dirichlet:  u = 0       on x = 0  and  x = 1
  Neumann:    du/dy = 0   on y = 0  and  y = 1

Data source
-----------
  COMSOL export files:
    - Surface_Solution.txt   (forcing function)
    - Poisson_SOlution.txt   (reference solution)

Usage
-----
  python deeponet_2d_gaussian.py
  python deeponet_2d_gaussian.py --grid_size 51 --epochs 500
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from networks import DeepONet2D
from eval_utils import plot_training_history
from solver import solve_poisson_2d_mixed, get_boundary_indices


# ============================================================================
#  COMSOL data loading
# ============================================================================

def load_comsol_data(forcing_file, solution_file, N=31):
    """
    Load scattered COMSOL data and interpolate onto a regular N x N grid.

    Returns
    -------
    x, y     : 1-D arrays (N,)
    f_grid   : (N, N)  forcing field  (raw, i.e. the full COMSOL RHS)
    u_grid   : (N, N)  reference solution
    """
    f_data = np.loadtxt(forcing_file, comments='%')   # (n_pts, 3): X, Y, F
    u_data = np.loadtxt(solution_file, comments='%')   # (n_pts, 3): X, Y, U

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    f_grid = griddata(f_data[:, :2], f_data[:, 2], (X, Y), method='cubic')
    u_grid = griddata(u_data[:, :2], u_data[:, 2], (X, Y), method='cubic')

    # Fill NaN at edges with nearest-neighbour
    for grid, data in [(f_grid, f_data), (u_grid, u_data)]:
        mask = np.isnan(grid)
        if mask.any():
            fill = griddata(data[:, :2], data[:, 2],
                            (X[mask], Y[mask]), method='nearest')
            grid[mask] = fill

    return x, y, f_grid, u_grid


# ============================================================================
#  FD solver validation
# ============================================================================

def validate_fd_solver(x, y, f_comsol, u_comsol, f_scale=100.0):
    """
    Run the FD mixed-BC solver and compare with the COMSOL reference.

    Returns the FD solution.
    """
    f_raw = f_comsol / f_scale  # solver internally multiplies by f_scale

    bc_values = {'left': 0.0, 'right': 0.0, 'bottom': 0.0, 'top': 0.0}
    bc_types = {
        'left': 'dirichlet', 'right': 'dirichlet',
        'bottom': 'neumann',  'top': 'neumann',
    }

    u_fd = solve_poisson_2d_mixed(x, y, f_raw, bc_values, bc_types,
                                  f_scale=f_scale)

    max_err = np.abs(u_fd - u_comsol).max()
    rel_l2  = np.sqrt(np.sum((u_fd - u_comsol)**2) /
                      (np.sum(u_comsol**2) + 1e-12))
    print(f"FD vs COMSOL  |  max|err|={max_err:.6f}  rel-L2={rel_l2:.6f}")
    return u_fd


# ============================================================================
#  Physics-Informed Trainer (mixed BCs)
# ============================================================================

class GaussianPdeTrainer:
    """
    Physics-informed training loop for the Gaussian-forcing Poisson problem.

    Loss = L_data
         + bc_d_weight * L_dirichlet
         + bc_n_weight * L_neumann
         + res_weight  * L_residual
    """

    def __init__(self, model, x, y, f_raw, u_true, *,
                 optimizer, scheduler=None,
                 device=None, f_scale=100.0,
                 res_weight=1.0, bc_d_weight=10.0, bc_n_weight=5.0,
                 n_res_points=200):
        """
        Parameters
        ----------
        model      : DeepONet2D
        x, y       : 1-D grid arrays (N,)
        f_raw      : (Ny, Nx) raw source  (solver convention, peak ~ 10)
        u_true     : (Ny, Nx) reference solution
        f_scale    : PDE multiplier  (-nabla^2 u = f_scale * f_raw)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.f_scale = float(f_scale)
        self.res_weight = float(res_weight)
        self.bc_d_weight = float(bc_d_weight)
        self.bc_n_weight = float(bc_n_weight)
        self.n_res_points = n_res_points

        self.Nx, self.Ny = len(x), len(y)
        N2 = self.Nx * self.Ny

        # Mesh
        X, Y = np.meshgrid(x, y)
        self.xy_flat_np = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
        self.xy_grid = torch.tensor(self.xy_flat_np, dtype=torch.float32,
                                    device=self.device)

        # Branch input  (1 sample: just the flattened forcing)
        branch_np = f_raw.ravel().astype(np.float32)
        self.branch_input = torch.tensor(branch_np, dtype=torch.float32,
                                         device=self.device).unsqueeze(0)  # (1, N*N)

        # Normalise target to [0, 1] so the sharp Gaussian peak is not
        # swamped by the large flat near-zero region in the MSE average.
        self.u_scale = float(np.abs(u_true).max()) + 1e-8
        u_norm = (u_true / self.u_scale).ravel().astype(np.float32)
        self.u_target = torch.tensor(u_norm, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)  # (1, N*N)

        # Boundary bookkeeping
        bc_idx, _ = get_boundary_indices(self.Nx, self.Ny)

        # Dirichlet indices: x=0 (left) and x=1 (right) — includes corners
        self.dirichlet_idx = np.union1d(bc_idx['left'], bc_idx['right'])

        # Neumann indices: y=0 (bottom) and y=1 (top) — exclude corners
        neumann_all = np.union1d(bc_idx['bottom'], bc_idx['top'])
        self.neumann_idx = np.setdiff1d(neumann_all, self.dirichlet_idx)

        # Interior indices (everything that is not on any boundary)
        all_bnd = np.unique(np.concatenate([bc_idx[s] for s in bc_idx]))
        self.interior_idx = np.setdiff1d(np.arange(N2), all_bnd)

        # Forcing at interior points
        f_flat = f_raw.ravel().astype(np.float32)
        self.f_interior = torch.tensor(f_flat[self.interior_idx],
                                       dtype=torch.float32,
                                       device=self.device)

        # Dirichlet target (u = 0)
        self.u_bc_d = torch.zeros(1, len(self.dirichlet_idx),
                                  dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    def train_step(self):
        """One optimiser step."""
        self.model.train()
        loss_fn = nn.MSELoss()

        # --- Pass 1: all grid points (detached) — data + Dirichlet BC ---
        u_pred_all = self.model(self.branch_input,
                                self.xy_grid.detach())       # (1, N*N)
        data_loss = loss_fn(u_pred_all, self.u_target)
        dirichlet_loss = loss_fn(u_pred_all[:, self.dirichlet_idx],
                                 self.u_bc_d)

        # --- Pass 2: Neumann boundary points (with grad) ---
        xy_n = torch.tensor(self.xy_flat_np[self.neumann_idx],
                            dtype=torch.float32,
                            device=self.device).requires_grad_(True)
        u_n = self.model(self.branch_input, xy_n).squeeze(0)  # (n_neumann,)
        grad_n = torch.autograd.grad(u_n.sum(), xy_n,
                                     create_graph=True)[0]
        du_dy_n = grad_n[:, 1]
        neumann_loss = torch.mean(du_dy_n ** 2)

        # --- Pass 3: PDE residual at subsampled interior points ---
        # Skip if res_weight == 0 to avoid noisy gradient from subsampling
        if self.res_weight > 0:
            n_res = min(self.n_res_points, len(self.interior_idx))
            sel = np.random.choice(len(self.interior_idx), n_res, replace=False)
            grid_sel = self.interior_idx[sel]

            xy_r = torch.tensor(self.xy_flat_np[grid_sel],
                                dtype=torch.float32,
                                device=self.device).requires_grad_(True)
            u_r = self.model(self.branch_input, xy_r).squeeze(0)  # (n_res,)

            grads = torch.autograd.grad(u_r.sum(), xy_r,
                                        create_graph=True)[0]
            du_dx, du_dy = grads[:, 0], grads[:, 1]
            d2u_dx2 = torch.autograd.grad(du_dx.sum(), xy_r,
                                          create_graph=True)[0][:, 0]
            d2u_dy2 = torch.autograd.grad(du_dy.sum(), xy_r,
                                          create_graph=True)[0][:, 1]
            laplacian = d2u_dx2 + d2u_dy2
            pde_res = -laplacian - self.f_scale * self.f_interior[sel]
            residual_loss = torch.mean(pde_res ** 2)
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

    # ------------------------------------------------------------------
    def run(self, epochs=300, verbose_freq=10,
            log_dir="./output_2d_gaussian", save_every=0):
        """Full training loop.  Returns history dict."""
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

            # Save best checkpoint by data loss
            if stats["data_loss"] < best_data_loss:
                best_data_loss = stats["data_loss"]
                torch.save(self.model.state_dict(),
                           os.path.join(log_dir, "model_best.pth"))

            if ep % verbose_freq == 0 or ep == 1:
                lr_now = self.optimizer.param_groups[0]["lr"]
                elapsed = (time.time() - start) / 60.0
                print(f"Epoch {ep:4d}/{epochs} | "
                      f"loss={stats['loss']:.4e}  "
                      f"data={stats['data_loss']:.4e}  "
                      f"dir={stats['dirichlet_loss']:.4e}  "
                      f"neu={stats['neumann_loss']:.4e}  "
                      f"res={stats['residual_loss']:.4e}  "
                      f"lr={lr_now:.2e}  time={elapsed:.1f}min")

            if save_every and ep % save_every == 0:
                torch.save(self.model.state_dict(),
                           os.path.join(log_dir, f"model_ep{ep}.pth"))

        print(f"\nBest data_loss: {best_data_loss:.4e}  (saved to model_best.pth)")
        # Save final model + history
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
    def predict(self, f_array=None):
        """
        Predict the solution on the full grid.

        Parameters
        ----------
        f_array : (Ny, Nx) or None.  If None, uses the stored forcing.

        Returns
        -------
        u_pred : (Ny, Nx)
        """
        self.model.eval()
        if f_array is not None:
            b_in = torch.tensor(f_array.ravel().astype(np.float32),
                                dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        else:
            b_in = self.branch_input

        with torch.no_grad():
            pred = self.model(b_in, self.xy_grid.detach())
        return pred.squeeze(0).cpu().numpy().reshape(self.Ny, self.Nx) * self.u_scale


# ============================================================================
#  PINTO-style 3-panel comparison plot
# ============================================================================

def plot_three_panel(x, y, u_ref, u_pred, save_dir=None,
                     fname="comparison_3panel.png",
                     ref_label="Reference"):
    """
    3-panel comparison: Reference | DeepONet prediction | Absolute error.
    Matches the visual style of the PINTO result figure.
    """
    import matplotlib
    matplotlib.use("Agg")

    X, Y = np.meshgrid(x, y)
    err  = np.abs(u_ref - u_pred)
    mse  = np.mean((u_ref - u_pred) ** 2)
    rel  = np.sqrt(np.sum((u_ref - u_pred) ** 2) /
                   (np.sum(u_ref ** 2) + 1e-12))

    vmin, vmax = u_ref.min(), u_ref.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- panel 1: reference ---
    c0 = axes[0].contourf(X, Y, u_ref, levels=64, cmap="jet",
                          vmin=vmin, vmax=vmax)
    axes[0].set_title(ref_label, fontsize=12)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    plt.colorbar(c0, ax=axes[0])

    # --- panel 2: prediction ---
    c1 = axes[1].contourf(X, Y, u_pred, levels=64, cmap="jet",
                          vmin=vmin, vmax=vmax)
    axes[1].set_title(f"DeepONet  (MSE={mse:.3e}, rel-L2={rel:.3e})",
                      fontsize=12)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    plt.colorbar(c1, ax=axes[1])

    # --- panel 3: absolute error ---
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


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="2-D Poisson (Gaussian forcing, mixed BCs) via "
                    "Physics-Informed DeepONet")

    # data
    parser.add_argument("--forcing_file", type=str,
                        default="Surface_Solution.txt")
    parser.add_argument("--solution_file", type=str,
                        default="Poisson_SOlution.txt")
    parser.add_argument("--grid_size", type=int, default=31,
                        help="N for the N x N interpolation grid")

    # model — trunk does most of the work for u(x,y); branch is fixed for 1 sample
    parser.add_argument("--p_dim", type=int, default=512,
                        help="Latent dimension (inner-product size)")
    parser.add_argument("--branch_h", type=int, nargs="+",
                        default=[256, 256])
    parser.add_argument("--trunk_h", type=int, nargs="+",
                        default=[512, 512, 512])
    # Fourier feature encoding for trunk (x,y) inputs
    parser.add_argument("--use_fourier", action="store_true", default=True,
                        help="Encode trunk (x,y) with Fourier features")
    parser.add_argument("--no_fourier", dest="use_fourier", action="store_false")
    parser.add_argument("--n_fourier", type=int, default=8,
                        help="Number of Fourier frequency bands")

    # training
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    # warm-restart period; LR cycles: 0→T0, T0→3T0, 3T0→7T0, ...
    parser.add_argument("--restart_period", type=int, default=500,
                        help="Unused (kept for CLI compatibility)")
    # res_weight=0 by default: FD ground truth already satisfies the PDE,
    # so residual is redundant and its stochastic subsampling adds gradient noise
    parser.add_argument("--res_weight", type=float, default=0.0)
    parser.add_argument("--bc_d_weight", type=float, default=10.0)
    parser.add_argument("--bc_n_weight", type=float, default=5.0)
    parser.add_argument("--n_res_pts", type=int, default=200,
                        help="Interior points subsampled for residual")
    parser.add_argument("--f_scale", type=float, default=100.0,
                        help="Source scaling factor in PDE")

    # IO
    parser.add_argument("--log_dir", type=str,
                        default="./output_2d_gaussian")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    N = args.grid_size

    # ------------------------------------------------------------------
    # 1.  Load COMSOL data
    # ------------------------------------------------------------------
    print("Loading COMSOL data ...")
    x, y, f_comsol, u_comsol = load_comsol_data(
        args.forcing_file, args.solution_file, N=N)

    f_raw = f_comsol / args.f_scale      # raw forcing (peak ~ 10)
    print(f"Grid {N}x{N}  |  f_raw range [{f_raw.min():.4f}, {f_raw.max():.4f}]")
    print(f"u_comsol range [{u_comsol.min():.4f}, {u_comsol.max():.4f}]")

    # ------------------------------------------------------------------
    # 2.  Validate FD solver against COMSOL
    # ------------------------------------------------------------------
    print("\nValidating FD solver ...")
    u_fd = validate_fd_solver(x, y, f_comsol, u_comsol, f_scale=args.f_scale)

    # ------------------------------------------------------------------
    # 3.  Build model
    # ------------------------------------------------------------------
    branch_in_dim = N * N  # flattened forcing only (BCs are fixed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}  |  Branch input dim: {branch_in_dim}")

    model = DeepONet2D(
        branch_in_dim=branch_in_dim,
        p=args.p_dim,
        branch_hidden=tuple(args.branch_h),
        trunk_hidden=tuple(args.trunk_h),
        activation=nn.SiLU,   # SiLU flows gradients better than Tanh for this problem
        use_fourier=args.use_fourier,
        n_fourier=args.n_fourier,
    )
    if args.use_fourier:
        print(f"Fourier encoding: {args.n_fourier} bands → trunk input dim = {2 + 4*args.n_fourier}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Single smooth cosine decay over the full training budget.
    # Warm restarts were counter-productive: LR spikes at each restart
    # pushed the model out of good minima it had already found.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ------------------------------------------------------------------
    # 4.  Train
    # ------------------------------------------------------------------
    trainer = GaussianPdeTrainer(
        model=model, x=x, y=y,
        f_raw=f_raw, u_true=u_fd,   # train against FD solution
        optimizer=optimizer, scheduler=scheduler,
        device=device, f_scale=args.f_scale,
        res_weight=args.res_weight,
        bc_d_weight=args.bc_d_weight,
        bc_n_weight=args.bc_n_weight,
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
    # 5.  Evaluate
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" Evaluation")
    print(f"{'='*60}\n")

    # Load the best checkpoint (not the final, which may be at a higher LR point)
    best_ckpt = os.path.join(args.log_dir, "model_best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print("Loaded best checkpoint for evaluation.")
    u_pred = trainer.predict()

    # Errors vs COMSOL (primary reference)
    mse_cm = np.mean((u_pred - u_comsol) ** 2)
    rel_cm = np.sqrt(np.sum((u_pred - u_comsol) ** 2) /
                     (np.sum(u_comsol ** 2) + 1e-12))
    print(f"vs COMSOL      MSE={mse_cm:.6e}  rel-L2={rel_cm:.6e}")

    # Errors vs FD solver (secondary)
    mse_fd = np.mean((u_pred - u_fd) ** 2)
    rel_fd = np.sqrt(np.sum((u_pred - u_fd) ** 2) /
                     (np.sum(u_fd ** 2) + 1e-12))
    print(f"vs FD solver   MSE={mse_fd:.6e}  rel-L2={rel_fd:.6e}")

    # ------------------------------------------------------------------
    # 6.  Plots
    # ------------------------------------------------------------------
    plot_training_history(history, save_dir=args.log_dir)

    # Main 3-panel: COMSOL (interpolated) as reference
    plot_three_panel(x, y, u_comsol, u_pred,
                     ref_label="COMSOL (interpolated)",
                     save_dir=args.log_dir,
                     fname="comparison_3panel.png")

    print(f"\nAll outputs saved to  {os.path.abspath(args.log_dir)}/")
    print("Done.")


if __name__ == "__main__":
    main()
