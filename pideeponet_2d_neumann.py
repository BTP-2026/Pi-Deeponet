"""
pideeponet_2d_neumann.py
========================
Physics-Informed DeepONet for 2D Poisson with variable Neumann BCs.
NO data loss — training uses only PDE residual + BC losses.
COMSOL data is loaded for final evaluation only (never as a training target).

PDE:  -nabla^2 u = 100 * f(x,y)   on [0,1]^2
BCs:  u = 0           on x=0, x=1  (Dirichlet)
      du/dy = v0      on y=0, y=1  (Neumann, v0 varies per sample)

Training signals
----------------
  L_res  : PDE residual at interior collocation points (autograd Laplacian)
  L_d    : Dirichlet u=0 on left/right walls
  L_n    : Neumann du/dy = v0 on top/bottom walls

Evaluation
----------
  data_v0.txt loaded ONLY for final rel-L2 / plots vs COMSOL.

Usage
-----
  python pideeponet_2d_neumann.py
  python pideeponet_2d_neumann.py --epochs 20000 --output_scale 10.0
  python pideeponet_2d_neumann.py --w_res 1.0 --w_d 100.0 --w_n 10.0
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

def load_forcing(forcing_file, N=31):
    """Interpolate COMSOL forcing field onto an N×N regular grid."""
    f_data = np.loadtxt(forcing_file, comments='%')
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    f_grid = griddata(f_data[:, :2], f_data[:, 2], (X, Y), method='cubic')
    mask = np.isnan(f_grid)
    if mask.any():
        fill = griddata(f_data[:, :2], f_data[:, 2],
                        (X[mask], Y[mask]), method='nearest')
        f_grid[mask] = fill
    print(f"Forcing field: range [{f_grid.min():.4f}, {f_grid.max():.4f}]  "
          f"(grid {N}x{N})")
    return x, y, f_grid.astype(np.float32)


def load_comsol_solutions(data_file, x, y):
    """Load COMSOL multi-v0 solutions (evaluation only). Returns v0_values, u_grids."""
    v0_values = []
    with open(data_file, 'r') as fh:
        for line in fh:
            if not line.startswith('%'):
                break
            matches = re.findall(r'v0=([0-9.]+)', line)
            v0_values.extend(float(m) for m in matches)
    v0_values = np.array(v0_values)

    raw = np.loadtxt(data_file, comments='%')
    xy_pts = raw[:, :2]
    u_pts = raw[:, 2:]

    N = len(x)
    X, Y = np.meshgrid(x, y)
    u_grids = np.zeros((len(v0_values), N, N), dtype=np.float32)
    for i in range(len(v0_values)):
        grid = griddata(xy_pts, u_pts[:, i], (X, Y), method='cubic')
        mask = np.isnan(grid)
        if mask.any():
            fill = griddata(xy_pts, u_pts[:, i],
                            (X[mask], Y[mask]), method='nearest')
            grid[mask] = fill
        u_grids[i] = grid.astype(np.float32)

    print(f"COMSOL solutions: {len(v0_values)} samples  "
          f"v0=[{v0_values[0]:.1f},{v0_values[-1]:.1f}]  "
          f"u range [{u_grids.min():.4f}, {u_grids.max():.4f}]")
    return v0_values, u_grids


# ============================================================================
#  Physics-Informed Trainer
# ============================================================================

class PIDeepONetTrainer:
    """
    Physics-Informed DeepONet — no data loss.

    Loss = w_res(t) * L_res + w_d * L_d + w_n * L_n

    Per step, loop over all n_v0 samples; for each sample:
      - Compute PDE residual at interior points via autograd Laplacian.
      - Enforce Dirichlet u=0 on x=0, x=1.
      - Enforce Neumann du/dy=v0 on y=0, y=1.
    Average across samples, apply loss weights, backprop.

    Branch = [f_flat, v0]  (N*N + 1 dims)
    Prediction = S * model(branch, xy)   (S = fixed output scale)
    """

    def __init__(self, model, x, y, v0_values, f_raw, *,
                 optimizer, scheduler=None,
                 device=None, f_scale=100.0, output_scale=10.0,
                 w_res=1.0, w_d=100.0, w_n=10.0,
                 warmup_epochs=500):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.f_scale = float(f_scale)
        self.S = float(output_scale)
        self.w_res = float(w_res)
        self.w_d = float(w_d)
        self.w_n = float(w_n)
        self.warmup_epochs = int(warmup_epochs)

        self.Nx, self.Ny = len(x), len(y)
        self.n_v0 = len(v0_values)
        self.v0_values = v0_values

        N2 = self.Nx * self.Ny
        X, Y = np.meshgrid(x, y)
        xy_flat = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)

        # Full grid tensor (for predict)
        self.xy_grid = torch.tensor(xy_flat, dtype=torch.float32,
                                    device=self.device)

        # Branch inputs: [f_flat, v0] per sample
        f_flat = f_raw.ravel().astype(np.float32)
        branch_np = np.zeros((self.n_v0, N2 + 1), dtype=np.float32)
        for i, v0 in enumerate(v0_values):
            branch_np[i, :N2] = f_flat
            branch_np[i, N2] = float(v0)
        self.branch_inputs = torch.tensor(branch_np, dtype=torch.float32,
                                          device=self.device)  # (n_v0, N*N+1)

        # Boundary indices
        bc_idx, _ = get_boundary_indices(self.Nx, self.Ny)
        dirichlet_idx = np.union1d(bc_idx['left'], bc_idx['right'])
        neumann_all = np.union1d(bc_idx['bottom'], bc_idx['top'])
        neumann_idx = np.setdiff1d(neumann_all, dirichlet_idx)
        boundary_all = np.union1d(dirichlet_idx, neumann_all)
        interior_idx = np.setdiff1d(np.arange(N2), boundary_all)

        # Static coordinate tensors — pre-cached on device.
        # For Dirichlet no grad is needed; for Neumann/interior we call
        # .detach().requires_grad_(True) per step (leaf node, no copy).
        self.dirichlet_coords = torch.tensor(xy_flat[dirichlet_idx],
                                              dtype=torch.float32,
                                              device=self.device)
        self.neumann_coords  = torch.tensor(xy_flat[neumann_idx],
                                             dtype=torch.float32,
                                             device=self.device)
        self.interior_coords = torch.tensor(xy_flat[interior_idx],
                                             dtype=torch.float32,
                                             device=self.device)

        # RHS of PDE at interior points: f_scale * f_raw
        f_rhs_int = (f_scale * f_flat)[interior_idx]
        self.f_rhs_interior = torch.tensor(f_rhs_int, dtype=torch.float32,
                                           device=self.device)

        print(f"Collocation: {len(interior_idx)} interior  "
              f"| {len(dirichlet_idx)} Dirichlet  "
              f"| {len(neumann_idx)} Neumann")

    # ------------------------------------------------------------------
    def _w_res_current(self, epoch):
        """Linear warmup of w_res over warmup_epochs to avoid early instability."""
        if self.warmup_epochs <= 0:
            return self.w_res
        return self.w_res * min(float(epoch) / self.warmup_epochs, 1.0)

    # ------------------------------------------------------------------
    def train_step(self, epoch):
        self.model.train()
        total_res = torch.zeros(1, device=self.device)
        total_dir = torch.zeros(1, device=self.device)
        total_neu = torch.zeros(1, device=self.device)

        w_res = self._w_res_current(epoch)

        for i in range(self.n_v0):
            b_i = self.branch_inputs[i:i+1]   # (1, N*N+1)
            v0_i = float(self.v0_values[i])

            # ---- Dirichlet: u = 0 at x=0, x=1 ----
            u_dir = self.S * self.model(b_i, self.dirichlet_coords).squeeze(0)
            total_dir = total_dir + torch.mean(u_dir ** 2)

            # ---- Neumann: du/dy = v0 at y=0, y=1 ----
            xy_neu = self.neumann_coords.detach().requires_grad_(True)
            u_neu = self.S * self.model(b_i, xy_neu).squeeze(0)
            grad_neu = torch.autograd.grad(
                u_neu.sum(), xy_neu, create_graph=True)[0]       # (n_neu, 2)
            du_dy = grad_neu[:, 1]
            total_neu = total_neu + torch.mean((du_dy - v0_i) ** 2)

            # ---- PDE residual: -laplacian(u) - f_scale*f_raw = 0 ----
            xy_int = self.interior_coords.detach().requires_grad_(True)
            u_int = self.S * self.model(b_i, xy_int).squeeze(0)

            grads1 = torch.autograd.grad(
                u_int.sum(), xy_int, create_graph=True)[0]       # (n_int, 2)
            u_x = grads1[:, 0]
            u_y = grads1[:, 1]

            u_xx = torch.autograd.grad(
                u_x.sum(), xy_int, create_graph=True)[0][:, 0]
            u_yy = torch.autograd.grad(
                u_y.sum(), xy_int, create_graph=True)[0][:, 1]

            residual = -(u_xx + u_yy) - self.f_rhs_interior
            total_res = total_res + torch.mean(residual ** 2)

        total_res = total_res / self.n_v0
        total_dir = total_dir / self.n_v0
        total_neu = total_neu / self.n_v0

        loss = w_res * total_res + self.w_d * total_dir + self.w_n * total_neu

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "res_loss": total_res.item(),
            "dir_loss": total_dir.item(),
            "neu_loss": total_neu.item(),
            "w_res_eff": w_res,
        }

    # ------------------------------------------------------------------
    def run(self, epochs=10000, verbose_freq=50,
            log_dir="./output_pideeponet_neumann", save_every=0):
        """Full training loop. Returns history dict."""
        os.makedirs(log_dir, exist_ok=True)
        keys = ["loss", "res_loss", "dir_loss", "neu_loss"]
        history = {k: [] for k in keys}
        best_loss = float("inf")
        start = time.time()

        for ep in range(1, epochs + 1):
            stats = self.train_step(ep)
            for k in keys:
                history[k].append(stats[k])

            if self.scheduler is not None:
                self.scheduler.step()

            if stats["loss"] < best_loss:
                best_loss = stats["loss"]
                torch.save(self.model.state_dict(),
                           os.path.join(log_dir, "model_best.pth"))

            if ep % verbose_freq == 0 or ep == 1:
                lr_now = self.optimizer.param_groups[0]["lr"]
                elapsed = (time.time() - start) / 60.0
                print(f"Epoch {ep:5d}/{epochs} | "
                      f"loss={stats['loss']:.4e}  "
                      f"res={stats['res_loss']:.4e}  "
                      f"dir={stats['dir_loss']:.4e}  "
                      f"neu={stats['neu_loss']:.4e}  "
                      f"w_res={stats['w_res_eff']:.3f}  "
                      f"lr={lr_now:.2e}  time={elapsed:.1f}min")

            if save_every and ep % save_every == 0:
                torch.save(self.model.state_dict(),
                           os.path.join(log_dir, f"model_ep{ep}.pth"))

        print(f"\nBest total physics loss: {best_loss:.4e}")
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
    def predict(self, idx):
        """
        Predict solution for training sample at index idx.
        Returns (Ny, Nx) array in physical units (S * u_net).
        """
        self.model.eval()
        b_in = self.branch_inputs[idx:idx+1]
        with torch.no_grad():
            u_pred = self.S * self.model(b_in, self.xy_grid)
        return u_pred.squeeze(0).cpu().numpy().reshape(self.Ny, self.Nx)


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

    c0 = axes[0].contourf(X, Y, u_ref, levels=64, cmap="jet", vmin=vmin, vmax=vmax)
    axes[0].set_title(ref_label, fontsize=12)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    plt.colorbar(c0, ax=axes[0])

    c1 = axes[1].contourf(X, Y, u_pred, levels=64, cmap="jet", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"PI-DeepONet  (MSE={mse:.3e}, rel-L2={rel:.3e})", fontsize=12)
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
    ax.set_title("PI-DeepONet rel-L2 error vs COMSOL for each v0")
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
        description="Physics-Informed DeepONet — 2D Poisson with variable Neumann BCs "
                    "(no data loss)")

    # data
    parser.add_argument("--forcing_file", type=str, default="Surface_Solution.txt")
    parser.add_argument("--data_file",    type=str, default="data_v0.txt",
                        help="COMSOL solutions — used only for final evaluation")
    parser.add_argument("--grid_size",    type=int, default=31)

    # model — defaults tuned for CPU exploration; use --p_dim 256 --trunk_h 256 256 256 for final run
    parser.add_argument("--p_dim",       type=int, default=128)
    parser.add_argument("--branch_h",    type=int, nargs="+", default=[256, 256])
    parser.add_argument("--trunk_h",     type=int, nargs="+", default=[128, 128])
    parser.add_argument("--use_fourier", action="store_true", default=True)
    parser.add_argument("--no_fourier",  dest="use_fourier", action="store_false")
    parser.add_argument("--n_fourier",   type=int, default=8)

    # physics loss
    parser.add_argument("--output_scale",  type=float, default=25.0,
                        help="Fixed global output scale S; prediction = S * u_net  "
                             "(u_max≈26 from COMSOL, so S=25 keeps u_net≈O(1))")
    parser.add_argument("--w_res",         type=float, default=1.0)
    parser.add_argument("--w_d",           type=float, default=100.0)
    parser.add_argument("--w_n",           type=float, default=10.0)
    parser.add_argument("--f_scale",       type=float, default=100.0,
                        help="PDE coefficient: -nabla^2 u = f_scale * f_raw")
    parser.add_argument("--warmup_epochs", type=int, default=500,
                        help="Linear warmup epochs for w_res (0 to disable)")

    # training
    parser.add_argument("--epochs",      type=int,   default=10000)
    parser.add_argument("--lr",          type=float, default=1e-3)

    # IO
    parser.add_argument("--log_dir",    type=str, default="./output_pideeponet_neumann")
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--seed",       type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    N = args.grid_size

    # ------------------------------------------------------------------
    # 1.  Load forcing (training input) + COMSOL solutions (eval only)
    # ------------------------------------------------------------------
    print("Loading forcing field ...")
    x, y, f_comsol = load_forcing(args.forcing_file, N=N)
    f_raw = f_comsol / args.f_scale   # branch input: normalised forcing

    print("\nLoading COMSOL solutions (eval only) ...")
    v0_values, u_grids = load_comsol_solutions(args.data_file, x, y)

    # ------------------------------------------------------------------
    # 2.  Build model
    # ------------------------------------------------------------------
    branch_in_dim = N * N + 1   # flattened forcing + v0 scalar
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}  |  Branch input dim: {branch_in_dim}  "
          f"|  Output scale S={args.output_scale}")

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
        print(f"Fourier encoding: {args.n_fourier} bands → "
              f"trunk dim = {2 + 4*args.n_fourier}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ------------------------------------------------------------------
    # 3.  Train (physics-only)
    # ------------------------------------------------------------------
    trainer = PIDeepONetTrainer(
        model=model, x=x, y=y,
        v0_values=v0_values, f_raw=f_raw,
        optimizer=optimizer, scheduler=scheduler,
        device=device,
        f_scale=args.f_scale,
        output_scale=args.output_scale,
        w_res=args.w_res, w_d=args.w_d, w_n=args.w_n,
        warmup_epochs=args.warmup_epochs,
    )

    print(f"\n{'='*60}")
    print(f" PI-DeepONet Training  |  {len(v0_values)} samples  |  {args.epochs} epochs")
    print(f" Loss weights: w_res={args.w_res}  w_d={args.w_d}  w_n={args.w_n}")
    print(f" Residual warmup: {args.warmup_epochs} epochs")
    print(f"{'='*60}\n")

    history = trainer.run(
        epochs=args.epochs, verbose_freq=50,
        log_dir=args.log_dir, save_every=args.save_every,
    )

    # ------------------------------------------------------------------
    # 4.  Evaluate vs COMSOL
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" Evaluation vs COMSOL")
    print(f"{'='*60}\n")

    best_ckpt = os.path.join(args.log_dir, "model_best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print("Loaded best checkpoint for evaluation.")

    rel_errors = np.zeros(len(v0_values))
    for i, v0 in enumerate(v0_values):
        u_pred = trainer.predict(i)
        u_ref = u_grids[i]
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

    for v0_plot in [0.0, 5.0, 10.0, 15.0, 20.0]:
        idx = int(np.argmin(np.abs(v0_values - v0_plot)))
        v0_actual = v0_values[idx]
        u_pred = trainer.predict(idx)
        u_ref = u_grids[idx]
        plot_three_panel(x, y, u_ref, u_pred,
                         save_dir=args.log_dir,
                         fname=f"comparison_v0_{v0_actual:.1f}.png",
                         ref_label=f"COMSOL  v0={v0_actual:.1f}")

    print(f"\nAll outputs saved to  {os.path.abspath(args.log_dir)}/")
    print("Done.")


if __name__ == "__main__":
    main()
