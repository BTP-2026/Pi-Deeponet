"""
pideeponet_2d_dirichlet_v0.py
==============================
Physics-Informed DeepONet for 2D Poisson with variable Dirichlet BC at x=0.

PDE:   -nabla^2 u = f(x,y)   on [0,1]^2
BCs:   u = v0    on x=0      (Dirichlet, variable)
       u = 0     on x=1      (Dirichlet, fixed)
       du/dy = 0 on y=0, y=1 (Neumann, zero flux)

Data
----
  Surface_Solution.txt : 4300 raw COMSOL mesh points with f(x,y) — used as-is
  data_v0.txt          : 4300 points x 41 v0 solutions — evaluation only, never trained on

Training signals (physics-only, no data loss)
---------------------------------------------
  L_res : PDE residual -(u_xx + u_yy) - f = 0  at ~4068 interior points
  L_d0  : Dirichlet u = v0  at ~58 points on x=0
  L_d1  : Dirichlet u = 0   at ~58 points on x=1
  L_neu : Neumann du/dy = 0 at ~116 points on y=0, y=1
  Total loss = L_res + L_d0 + L_d1 + L_neu  (equal weights)

Architecture
------------
  Branch : v0 -> Fourier encoding (1+2*n_v0_freqs dims) -> MLP -> p
  Trunk  : (x,y) -> Fourier features (2+4*n_fourier dims) -> MLP -> p
  Output : S * (dot(branch, trunk) + bias)

Train/Test split
----------------
  Train : v0 values at every --train_step (e.g. 0, 2, 4, ..., 20)
  Test  : remaining v0 values (unseen during training)

Usage
-----
  python pideeponet_2d_dirichlet_v0.py
  python pideeponet_2d_dirichlet_v0.py --train_step 2 --epochs 10000
  python pideeponet_2d_dirichlet_v0.py --p_dim 256 --trunk_h 256 256 256 --n_fourier 16
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
import matplotlib.tri as mtri

from networks import DeepONet2D


# ============================================================================
#  Data loading
# ============================================================================

def load_data(forcing_file, data_file):
    """
    Load raw COMSOL mesh data — no interpolation.

    Returns
    -------
    xy        : (4300, 2) float32   mesh point coordinates (x, y)
    f_vals    : (4300,)  float32   forcing function f(x,y)
    v0_values : (41,)   float64   v0 parameter values
    u_comsol  : (4300, 41) float32 COMSOL solutions, column i = solution for v0_values[i]
    """
    surf = np.loadtxt(forcing_file, comments='%')
    xy = surf[:, :2].astype(np.float32)
    f_vals = surf[:, 2].astype(np.float32)
    print(f"Forcing: {len(xy)} pts  f in [{f_vals.min():.3f}, {f_vals.max():.3f}]")

    v0_values = []
    with open(data_file, 'r') as fh:
        for line in fh:
            if not line.startswith('%'):
                break
            v0_values.extend(float(m) for m in re.findall(r'v0=([0-9.]+)', line))
    v0_values = np.array(v0_values)

    raw = np.loadtxt(data_file, comments='%')
    u_comsol = raw[:, 2:].astype(np.float32)

    print(f"COMSOL: {len(v0_values)} samples  v0=[{v0_values[0]:.1f}, {v0_values[-1]:.1f}]"
          f"  u in [{u_comsol.min():.3f}, {u_comsol.max():.3f}]")

    return xy, f_vals, v0_values, u_comsol


# ============================================================================
#  Boundary identification
# ============================================================================

def identify_boundaries(xy, tol=1e-10):
    """
    Split raw mesh points into four index sets.
    Corners belong to Dirichlet (x wall), not Neumann (y wall).

    Returns dir_x0, dir_x1, neu, interior  — all numpy index arrays.
    """
    x, y = xy[:, 0], xy[:, 1]

    is_x0 = x < tol
    is_x1 = np.abs(x - 1.0) < tol
    is_y0 = y < tol
    is_y1 = np.abs(y - 1.0) < tol
    is_xwall = is_x0 | is_x1

    dir_x0 = np.where(is_x0)[0]
    dir_x1 = np.where(is_x1)[0]
    neu = np.where((is_y0 | is_y1) & ~is_xwall)[0]
    interior = np.where(~is_x0 & ~is_x1 & ~is_y0 & ~is_y1)[0]

    print(f"Points  x=0:{len(dir_x0)}  x=1:{len(dir_x1)}"
          f"  Neumann:{len(neu)}  Interior:{len(interior)}"
          f"  Sum:{len(dir_x0) + len(dir_x1) + len(neu) + len(interior)}")

    return dir_x0, dir_x1, neu, interior


# ============================================================================
#  Fourier encoding for v0
# ============================================================================

def encode_v0(v0_array, n_freqs=4):
    """
    Fourier encode v0 values.
    Output: [v0, sin(pi*1*v0/20), cos(pi*1*v0/20), ..., sin(pi*n*v0/20), cos(pi*n*v0/20)]
    Shape: (N, 1 + 2*n_freqs)
    """
    v0 = np.asarray(v0_array, dtype=np.float32).reshape(-1, 1)
    ks = np.arange(1, n_freqs + 1, dtype=np.float32)
    arg = np.pi * ks[None, :] * v0 / 20.0
    return np.concatenate([v0, np.sin(arg), np.cos(arg)], axis=1)


# ============================================================================
#  Physics-Informed Trainer
# ============================================================================

class PIDeepONetTrainer:
    """
    Physics-only training — no data loss.

    Loss = w_res(t)*L_res + L_d0 + L_d1 + L_neu   (equal weights except warmup on res)

    w_res(t) ramps linearly 0→1 over warmup_epochs.
    Training runs only on the v0 indices listed in train_indices.
    """

    def __init__(self, model, xy, f_vals, v0_values, optimizer, *,
                 scheduler=None, device=None, output_scale=20.0,
                 warmup_epochs=500, train_indices=None, n_v0_freqs=4):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.S = float(output_scale)
        self.warmup_epochs = int(warmup_epochs)
        self.v0_values = v0_values
        self.n_v0 = len(v0_values)
        self.train_indices = list(train_indices) if train_indices is not None else list(range(self.n_v0))

        dir_x0_idx, dir_x1_idx, neu_idx, interior_idx = identify_boundaries(xy)

        def _t(arr):
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        self.xy_all = _t(xy)
        self.dir_x0_coords = _t(xy[dir_x0_idx])
        self.dir_x1_coords = _t(xy[dir_x1_idx])
        self.neu_coords = _t(xy[neu_idx])
        self.interior_coords = _t(xy[interior_idx])
        self.f_interior = _t(f_vals[interior_idx])

        encoded = encode_v0(v0_values, n_freqs=n_v0_freqs)
        self.branch_inputs = _t(encoded)

    def _w_res_now(self, epoch):
        if self.warmup_epochs <= 0:
            return 1.0
        return min(float(epoch) / self.warmup_epochs, 1.0)

    def train_step(self, epoch):
        self.model.train()
        acc_res = acc_d0 = acc_d1 = acc_neu = torch.zeros(1, device=self.device)
        w_res = self._w_res_now(epoch)
        n = len(self.train_indices)

        for i in self.train_indices:
            b = self.branch_inputs[i:i+1]
            v0i = float(self.v0_values[i])

            # Dirichlet x=0: u = v0 → u_net target = v0 / S
            u_d0 = self.model(b, self.dir_x0_coords).squeeze(0)
            acc_d0 = acc_d0 + ((u_d0 - v0i / self.S) ** 2).mean()

            # Dirichlet x=1: u = 0 → u_net target = 0
            u_d1 = self.model(b, self.dir_x1_coords).squeeze(0)
            acc_d1 = acc_d1 + (u_d1 ** 2).mean()

            # Neumann y=0,1: du/dy = 0
            xy_neu = self.neu_coords.detach().requires_grad_(True)
            u_neu = self.model(b, xy_neu).squeeze(0)
            g_neu = torch.autograd.grad(u_neu.sum(), xy_neu, create_graph=True)[0]
            acc_neu = acc_neu + (g_neu[:, 1] ** 2).mean()

            # PDE residual: -(u_xx + u_yy) = f
            xy_int = self.interior_coords.detach().requires_grad_(True)
            u_int = self.model(b, xy_int).squeeze(0)
            g1 = torch.autograd.grad(u_int.sum(), xy_int, create_graph=True)[0]
            u_xx = torch.autograd.grad(g1[:, 0].sum(), xy_int, create_graph=True)[0][:, 0]
            u_yy = torch.autograd.grad(g1[:, 1].sum(), xy_int, create_graph=True)[0][:, 1]
            L_res = ((-(u_xx + u_yy) * self.S - self.f_interior) ** 2).mean()
            acc_res = acc_res + L_res

        L_d0 = acc_d0 / n
        L_d1 = acc_d1 / n
        L_neu = acc_neu / n
        L_res = acc_res / n

        loss = w_res * L_res + L_d0 + L_d1 + L_neu

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "res_loss": L_res.item(),
            "d0_loss": L_d0.item(),
            "d1_loss": L_d1.item(),
            "neu_loss": L_neu.item(),
            "w_res_eff": w_res,
        }

    def run(self, epochs=10000, verbose_freq=50, log_dir="./output_pideeponet_v0"):
        os.makedirs(log_dir, exist_ok=True)
        keys = ["loss", "res_loss", "d0_loss", "d1_loss", "neu_loss"]
        history = {k: [] for k in keys}
        best_loss = float("inf")
        t0 = time.time()

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
                lr = self.optimizer.param_groups[0]["lr"]
                ela = (time.time() - t0) / 60.0
                print(f"Epoch {ep:5d}/{epochs} | "
                      f"loss={stats['loss']:.4e}  "
                      f"res={stats['res_loss']:.4e}  "
                      f"d0={stats['d0_loss']:.4e}  "
                      f"d1={stats['d1_loss']:.4e}  "
                      f"neu={stats['neu_loss']:.4e}  "
                      f"w_res={stats['w_res_eff']:.3f}  "
                      f"lr={lr:.2e}  "
                      f"time={ela:.1f}min")

        torch.save(self.model.state_dict(),
                   os.path.join(log_dir, "model_final.pth"))
        print(f"\nBest physics loss: {best_loss:.4e}")

        try:
            import pandas as pd
            pd.DataFrame(history).to_csv(
                os.path.join(log_dir, "history.csv"), index=False)
        except ImportError:
            np.savez(os.path.join(log_dir, "history.npz"), **history)

        return history

    def predict(self, idx):
        """Return predicted u at all 4300 raw mesh points. Shape: (4300,)."""
        self.model.eval()
        with torch.no_grad():
            u = self.S * self.model(
                self.branch_inputs[idx:idx+1], self.xy_all)
        return u.squeeze(0).cpu().numpy()


# ============================================================================
#  Plotting
# ============================================================================

def _tri(xy):
    return mtri.Triangulation(xy[:, 0], xy[:, 1])


def plot_three_panel(xy, u_ref, u_pred, save_dir=None,
                     fname="comparison_3panel.png", label=""):
    triang = _tri(xy)
    err = np.abs(u_ref - u_pred)
    rel = np.sqrt(np.sum((u_ref - u_pred) ** 2) /
                  (np.sum(u_ref ** 2) + 1e-12))
    vmin, vmax = u_ref.min(), u_ref.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    c0 = axes[0].tricontourf(triang, u_ref, levels=64, cmap="jet",
                              vmin=vmin, vmax=vmax)
    axes[0].set_title(f"COMSOL  {label}")
    axes[0].set_aspect("equal")
    plt.colorbar(c0, ax=axes[0])

    c1 = axes[1].tricontourf(triang, u_pred, levels=64, cmap="jet",
                              vmin=vmin, vmax=vmax)
    axes[1].set_title(f"PI-DeepONet  rel-L2={rel*100:.2f}%")
    axes[1].set_aspect("equal")
    plt.colorbar(c1, ax=axes[1])

    c2 = axes[2].tricontourf(triang, err, levels=64, cmap="hot")
    axes[2].set_title(f"|Error|  max={err.max():.3e}")
    axes[2].set_aspect("equal")
    plt.colorbar(c2, ax=axes[2])

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.close(fig)


def plot_error_summary(v0_values, rel_errors, train_mask, save_dir=None):
    fig, ax = plt.subplots(figsize=(14, 4))

    colors = ['#1f77b4' if train_mask[i] else '#d62728'
              for i in range(len(v0_values))]
    ax.bar(v0_values, rel_errors * 100, color=colors, alpha=0.85, width=0.4)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#1f77b4', label='Train v0'),
        Patch(color='#d62728', label='Test v0  (unseen)'),
    ])
    ax.set_xlabel("v0 (Dirichlet BC at x=0)")
    ax.set_ylabel("Rel-L2 error (%)")

    train_mean = rel_errors[train_mask].mean() * 100
    test_mean = rel_errors[~train_mask].mean() * 100
    ax.set_title(f"PI-DeepONet rel-L2 vs COMSOL  |  Train mean={train_mean:.3f}%"
                 f"  Test mean={test_mean:.3f}%")
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "error_vs_v0.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.close(fig)


def plot_history(history, save_dir=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, v in history.items():
        ax.plot(np.log10(np.array(v) + 1e-15), label=k)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log10(loss)")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True)
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
    parser = argparse.ArgumentParser(
        description="PI-DeepONet: 2D Poisson, variable Dirichlet BC at x=0")

    parser.add_argument("--forcing_file", default="Surface_Solution.txt")
    parser.add_argument("--data_file", default="data_v0.txt")

    parser.add_argument("--p_dim", type=int, default=256)
    parser.add_argument("--branch_h", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--trunk_h", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--n_fourier", type=int, default=16,
                        help="Fourier frequencies for trunk spatial encoding")
    parser.add_argument("--n_v0_freqs", type=int, default=4,
                        help="Fourier frequencies for branch v0 encoding; branch_in = 1+2*n")
    parser.add_argument("--output_scale", type=float, default=20.0)
    parser.add_argument("--warmup_epochs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--log_dir", type=str, default="./output_pideeponet_v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint (.pth) to resume from (weights only; LR resets)")
    parser.add_argument("--train_step", type=float, default=2.0,
                        help="Use v0 values at multiples of this step for training; rest are test")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading data ...")
    xy, f_vals, v0_values, u_comsol = load_data(args.forcing_file, args.data_file)

    tol = 0.01
    train_mask = np.array([np.abs(v % args.train_step) < tol for v in v0_values])
    train_indices = np.where(train_mask)[0].tolist()
    test_indices = np.where(~train_mask)[0].tolist()
    print(f"\nTrain v0 ({len(train_indices)}): {v0_values[train_indices]}")
    print(f"Test  v0 ({len(test_indices)}):  {v0_values[test_indices]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    branch_in_dim = 1 + 2 * args.n_v0_freqs
    model = DeepONet2D(
        branch_in_dim=branch_in_dim,
        p=args.p_dim,
        branch_hidden=tuple(args.branch_h),
        trunk_hidden=tuple(args.trunk_h),
        activation=nn.SiLU,
        use_fourier=True,
        n_fourier=args.n_fourier,
    )
    n_params = sum(p.numel() for p in model.parameters())
    trunk_in = 2 + 4 * args.n_fourier
    print(f"\nDevice: {device}  |  params: {n_params:,}")
    print(f"Branch: {branch_in_dim} -> {args.branch_h}  (v0 Fourier n={args.n_v0_freqs})")
    print(f"Trunk:  {trunk_in} -> {args.trunk_h}  (spatial Fourier n={args.n_fourier})")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f"Resumed weights from: {args.resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    trainer = PIDeepONetTrainer(
        model=model, xy=xy, f_vals=f_vals, v0_values=v0_values,
        optimizer=optimizer, scheduler=scheduler,
        device=device, output_scale=args.output_scale,
        warmup_epochs=args.warmup_epochs, train_indices=train_indices,
        n_v0_freqs=args.n_v0_freqs,
    )

    print(f"\n{'=' * 60}")
    print(f" Training  |  {len(train_indices)} train samples  |  {args.epochs} epochs")
    print(f" Loss = L_res + L_d0 + L_d1 + L_neu  (equal weights)")
    print(f" warmup={args.warmup_epochs}  lr={args.lr}  S={args.output_scale}")
    print(f"{'=' * 60}\n")

    history = trainer.run(epochs=args.epochs, verbose_freq=50, log_dir=args.log_dir)

    # --- Evaluation vs COMSOL ---
    print(f"\n{'=' * 60}")
    print(f" Evaluation vs COMSOL (at raw 4300 mesh points)")
    print(f"{'=' * 60}\n")

    best_ckpt = os.path.join(args.log_dir, "model_best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device,
                                         weights_only=True))
        print("Loaded best checkpoint.\n")

    rel_errors = np.zeros(len(v0_values))
    for i in range(len(v0_values)):
        u_pred = trainer.predict(i)
        u_ref = u_comsol[:, i]
        rel_errors[i] = np.sqrt(np.sum((u_pred - u_ref) ** 2) /
                                (np.sum(u_ref ** 2) + 1e-12))

    print(f"{'v0':>6}  {'rel-L2 (%)':>12}  {'split'}")
    print("-" * 30)
    for i, (v0, err) in enumerate(zip(v0_values, rel_errors)):
        split = "TRAIN" if train_mask[i] else "test"
        print(f"{v0:6.1f}  {err * 100:12.4f}  {split}")

    train_mean = rel_errors[train_mask].mean() * 100
    test_mean = rel_errors[~train_mask].mean() * 100
    train_max = rel_errors[train_mask].max() * 100
    test_max = rel_errors[~train_mask].max() * 100
    print(f"\nTrain — Mean: {train_mean:.4f}%  Max: {train_max:.4f}%")
    print(f"Test  — Mean: {test_mean:.4f}%  Max: {test_max:.4f}%")

    # --- Plots ---
    plot_history(history, save_dir=args.log_dir)
    plot_error_summary(v0_values, rel_errors, train_mask, save_dir=args.log_dir)

    for v0_plot in (0.0, 5.0, 10.0, 15.0, 20.0):
        idx = int(np.argmin(np.abs(v0_values - v0_plot)))
        split_tag = "TRAIN" if train_mask[idx] else "test"
        plot_three_panel(
            xy, u_comsol[:, idx], trainer.predict(idx),
            save_dir=args.log_dir,
            fname=f"comparison_v0_{v0_values[idx]:.1f}.png",
            label=f"v0={v0_values[idx]:.1f} [{split_tag}]",
        )

    print(f"\nAll outputs saved to {os.path.abspath(args.log_dir)}/")
    print("Done.")


if __name__ == "__main__":
    main()
