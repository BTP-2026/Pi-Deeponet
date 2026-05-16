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

Architecture
------------
  Branch : v0 scalar (dim=1) -> MLP -> p
  Trunk  : (x,y) -> Fourier features (34 dims) -> MLP -> p
  Output : S * (dot(branch, trunk) + bias)

Usage
-----
  python pideeponet_2d_dirichlet_v0_cuda.py --device cuda
  python pideeponet_2d_dirichlet_v0_cuda.py --device auto --epochs 15000 --p_dim 256 --trunk_h 256 256 256
  python pideeponet_2d_dirichlet_v0_cuda.py --w_res 5.0 --w_d 100.0 --w_neu 1.0
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
from tqdm.auto import tqdm

from networks import DeepONet2D


# ============================================================================
#  Device / CUDA setup
# ============================================================================

def resolve_device(requested="auto", allow_tf32=True):
    """
    Resolve a requested runtime device and print enough diagnostics to make
    CUDA/CPU mismatches obvious at startup.
    """
    requested = str(requested).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")

    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build and check your NVIDIA driver."
        )

    device = "cuda" if (requested == "cuda" or (requested == "auto" and cuda_available)) else "cpu"

    print("\nRuntime device check")
    print(f"  torch: {torch.__version__}")
    print(f"  torch CUDA build: {torch.version.cuda}")
    print(f"  torch.cuda.is_available(): {cuda_available}")

    if device == "cuda":
        torch.cuda.set_device(0)
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            precision = "tf32" if allow_tf32 else "ieee"
            torch.backends.cuda.matmul.fp32_precision = precision
            torch.backends.cudnn.conv.fp32_precision = precision
            torch.backends.cudnn.rnn.fp32_precision = precision
        else:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"  selected GPU: {torch.cuda.get_device_name(0)}")
        print(f"  compute capability: {props.major}.{props.minor}")
        print(f"  GPU memory: {total_gb:.2f} GiB")
        print(f"  TF32 enabled: {allow_tf32}")
    else:
        print("  selected device: cpu")

    return device


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
    xy     = surf[:, :2].astype(np.float32)
    f_vals = surf[:, 2].astype(np.float32)
    print(f"Forcing: {len(xy)} pts  f in [{f_vals.min():.3f}, {f_vals.max():.3f}]")

    v0_values = []
    with open(data_file) as fh:
        for line in fh:
            if not line.startswith('%'):
                break
            v0_values.extend(float(m) for m in re.findall(r'v0=([0-9.]+)', line))
    v0_values = np.array(v0_values)

    raw      = np.loadtxt(data_file, comments='%')
    u_comsol = raw[:, 2:].astype(np.float32)   # (4300, 41)
    print(f"COMSOL: {len(v0_values)} samples  "
          f"v0=[{v0_values[0]:.1f}, {v0_values[-1]:.1f}]  "
          f"u in [{u_comsol.min():.3f}, {u_comsol.max():.3f}]")
    return xy, f_vals, v0_values, u_comsol


def identify_boundaries(xy, tol=1e-10):
    """
    Split raw mesh points into four index sets.
    Corners belong to Dirichlet (x wall), not Neumann (y wall).

    Returns dir_x0, dir_x1, neu, interior  — all numpy index arrays.
    """
    x, y  = xy[:, 0], xy[:, 1]
    is_x0 = x < tol
    is_x1 = np.abs(x - 1.0) < tol
    is_y0 = y < tol
    is_y1 = np.abs(y - 1.0) < tol
    is_xwall = is_x0 | is_x1

    dir_x0   = np.where(is_x0)[0]
    dir_x1   = np.where(is_x1)[0]
    neu      = np.where((is_y0 | is_y1) & ~is_xwall)[0]
    interior = np.where(~(is_xwall | is_y0 | is_y1))[0]

    print(f"Points  x=0:{len(dir_x0)}  x=1:{len(dir_x1)}  "
          f"Neumann:{len(neu)}  Interior:{len(interior)}  "
          f"Sum:{len(dir_x0)+len(dir_x1)+len(neu)+len(interior)}")
    return dir_x0, dir_x1, neu, interior


# ============================================================================
#  Physics-Informed Trainer
# ============================================================================

class PIDeepONetTrainer:
    """
    Physics-only training — no data loss.

    Loss = w_res(t)*L_res + w_d*(L_d0 + L_d1) + w_neu*L_neu

    w_res(t) ramps linearly from 0 to w_res over warmup_epochs.
    """

    def __init__(self, model, xy, f_vals, v0_values, *,
                 optimizer, scheduler=None,
                 device=None, output_scale=20.0,
                 w_res=1.0, w_d=100.0, w_neu=10.0,
                 warmup_epochs=500, v0_zero_weight=1.0,
                 train_indices=None, num_batches=1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.S            = float(output_scale)
        self.w_res        = float(w_res)
        self.w_d          = float(w_d)
        self.w_neu        = float(w_neu)
        self.warmup_epochs   = int(warmup_epochs)
        self.v0_zero_weight  = float(v0_zero_weight)
        self.v0_values    = v0_values
        self.train_indices = np.arange(len(v0_values)) if train_indices is None else np.array(train_indices, dtype=int)
        if self.train_indices.size == 0:
            raise ValueError("At least one v0 sample must be selected for training.")
        self.train_v0_values = v0_values[self.train_indices]
        self.n_v0         = len(self.train_indices)
        self.num_batches  = max(1, int(num_batches))
        # Match TF: dataset.batch(ceil(N / batch)) — minibatch size is ceil(N / num_batches)
        self.minibatch_size = int(np.ceil(self.n_v0 / self.num_batches))

        dir_x0_idx, dir_x1_idx, neu_idx, interior_idx = identify_boundaries(xy)

        def _t(arr):
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        # Pre-cached coordinate tensors
        self.xy_all          = _t(xy)
        self.dir_x0_coords   = _t(xy[dir_x0_idx])
        self.dir_x1_coords   = _t(xy[dir_x1_idx])
        self.neu_coords      = _t(xy[neu_idx])
        self.interior_coords = _t(xy[interior_idx])
        self.f_interior      = _t(f_vals[interior_idx])   # PDE RHS at interior pts

        # Branch inputs: one scalar v0 per sample
        self.branch_inputs = _t(v0_values.astype(np.float32).reshape(-1, 1))  # (all_v0, 1)
        self.train_branch_inputs = self.branch_inputs[self.train_indices]

    # ------------------------------------------------------------------
    def _w_res_now(self, epoch):
        if self.warmup_epochs <= 0:
            return self.w_res
        return self.w_res * min(float(epoch) / self.warmup_epochs, 1.0)

    # ------------------------------------------------------------------
    def _forward_per_sample_xy(self, branch_input, xy_batched):
        """
        Forward DeepONet with per-sample coordinate tensors.

        Standard model.forward(branch, xy) shares xy across the batch, which
        makes per-sample autograd of (u_xx, u_yy) impossible in one call —
        gradients collapse along the batch dim. Here xy_batched is shaped
        (B, n_pts, 2) so each sample has its own xy leaf; the resulting
        u[b, j] depends ONLY on xy_batched[b, j], so a single autograd.grad
        call returns properly per-sample gradients of shape (B, n_pts, 2).

        Trunk is a pointwise MLP, so we just flatten -> trunk -> reshape.
        """
        B, n_pts, _ = xy_batched.shape
        b = self.model.branch(branch_input)                         # (B, p)
        xy_flat = xy_batched.reshape(B * n_pts, 2)
        if self.model.fourier_enc is not None:
            xy_flat = self.model.fourier_enc(xy_flat)
        t_flat = self.model.trunk(xy_flat)                          # (B*n_pts, p)
        t = t_flat.view(B, n_pts, -1)                               # (B, n_pts, p)
        out = (b.unsqueeze(1) * t).sum(dim=-1)                      # (B, n_pts)
        return out + self.model.bias

    # ------------------------------------------------------------------
    def _minibatch_step(self, batch_positions, w_res):
        """
        Vectorized forward+backward over the mini-batch dimension B.
        One optimizer.step() per mini-batch (TF dataset.batch() semantics).
        Memory scales with B because all B forward passes share one autograd
        graph — no per-sample Python loop, no serialized graphs.
        """
        B = len(batch_positions)
        idx_t = torch.as_tensor(batch_positions, dtype=torch.long, device=self.device)
        b_input = self.train_branch_inputs.index_select(0, idx_t)   # (B, 1)
        v0_targets = torch.as_tensor(
            self.train_v0_values[batch_positions], dtype=torch.float32, device=self.device
        ).view(B, 1)                                                # (B, 1)

        # Per-sample weights — only v0=0 sample gets v0_zero_weight, others 1.0.
        orig_idx = self.train_indices[batch_positions]
        weights_np = np.where(orig_idx == 0, self.v0_zero_weight, 1.0).astype(np.float32)
        w_t = torch.as_tensor(weights_np, device=self.device)
        w_t = w_t / w_t.sum()                                       # (B,)

        self.optimizer.zero_grad()

        # ---- Dirichlet x=0 : u = v0 (no coord-grad needed) ----
        u_d0 = self.S * self.model(b_input, self.dir_x0_coords)     # (B, n_d0)
        L_d0_per = torch.mean((u_d0 - v0_targets) ** 2, dim=1)      # (B,)
        L_d0 = (w_t * L_d0_per).sum()

        # ---- Dirichlet x=1 : u = 0 ----
        u_d1 = self.S * self.model(b_input, self.dir_x1_coords)     # (B, n_d1)
        L_d1_per = torch.mean(u_d1 ** 2, dim=1)                     # (B,)
        L_d1 = (w_t * L_d1_per).sum()

        # ---- Neumann y=0,y=1 : du/dy = 0 ----
        n_neu = self.neu_coords.shape[0]
        xy_neu_b = self.neu_coords.unsqueeze(0).expand(B, n_neu, 2).contiguous()
        xy_neu_b.requires_grad_(True)
        u_neu = self.S * self._forward_per_sample_xy(b_input, xy_neu_b)  # (B, n_neu)
        g_neu = torch.autograd.grad(u_neu.sum(), xy_neu_b, create_graph=True)[0]
        L_neu_per = torch.mean(g_neu[..., 1] ** 2, dim=1)           # (B,)
        L_neu = (w_t * L_neu_per).sum()

        # ---- PDE residual : -(u_xx + u_yy) - f = 0 ----
        n_int = self.interior_coords.shape[0]
        xy_int_b = self.interior_coords.unsqueeze(0).expand(B, n_int, 2).contiguous()
        xy_int_b.requires_grad_(True)
        u_int = self.S * self._forward_per_sample_xy(b_input, xy_int_b)  # (B, n_int)
        g1 = torch.autograd.grad(u_int.sum(), xy_int_b, create_graph=True)[0]   # (B, n_int, 2)
        u_xx = torch.autograd.grad(g1[..., 0].sum(), xy_int_b, create_graph=True)[0][..., 0]
        u_yy = torch.autograd.grad(g1[..., 1].sum(), xy_int_b, create_graph=True)[0][..., 1]
        res = -(u_xx + u_yy) - self.f_interior.unsqueeze(0)         # (B, n_int)
        L_res_per = torch.mean(res ** 2, dim=1)                     # (B,)
        L_res = (w_t * L_res_per).sum()

        loss = w_res * L_res + self.w_d * (L_d0 + L_d1) + self.w_neu * L_neu
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return (loss.item(), L_res.item(), L_d0.item(), L_d1.item(), L_neu.item())

    # ------------------------------------------------------------------
    def train_step(self, epoch):
        """
        One epoch = full shuffle of train_indices, split into mini-batches of
        size ceil(n_v0 / num_batches), one optimizer step per mini-batch.

        This mirrors the TensorFlow pipeline:
            dataset.shuffle(buffer_size=N).batch(ceil(N / num_batches))
        """
        self.model.train()
        w_res = self._w_res_now(epoch)

        perm = np.random.permutation(self.n_v0)
        chunks = [perm[k:k + self.minibatch_size]
                  for k in range(0, self.n_v0, self.minibatch_size)]

        agg_loss = agg_res = agg_d0 = agg_d1 = agg_neu = 0.0
        for chunk in chunks:
            l, r, d0, d1, n = self._minibatch_step(chunk, w_res)
            agg_loss += l; agg_res += r; agg_d0 += d0; agg_d1 += d1; agg_neu += n

        n_steps = max(1, len(chunks))
        return {
            "loss":     agg_loss / n_steps,
            "res_loss": agg_res / n_steps,
            "d0_loss":  agg_d0 / n_steps,
            "d1_loss":  agg_d1 / n_steps,
            "neu_loss": agg_neu / n_steps,
            "w_res_eff": w_res,
            "n_minibatches": n_steps,
        }

    # ------------------------------------------------------------------
    def run(self, epochs=10000, verbose_freq=50, log_dir="./output_pideeponet_v0",
            show_progress=True):
        os.makedirs(log_dir, exist_ok=True)
        keys = ["loss", "res_loss", "d0_loss", "d1_loss", "neu_loss"]
        history = {k: [] for k in keys}
        best_loss = float("inf")
        t0 = time.time()
        epoch_iter = range(1, epochs + 1)
        progress = tqdm(epoch_iter, total=epochs, desc="Training",
                        unit="epoch", dynamic_ncols=True) if show_progress else epoch_iter

        for ep in progress:
            stats = self.train_step(ep)
            for k in keys:
                history[k].append(stats[k])
            if self.scheduler:
                self.scheduler.step()

            if stats["loss"] < best_loss:
                best_loss = stats["loss"]
                torch.save(self.model.state_dict(),
                           os.path.join(log_dir, "model_best.pth"))

            lr = self.optimizer.param_groups[0]["lr"]
            if show_progress:
                progress.set_postfix(
                    loss=f"{stats['loss']:.3e}",
                    res=f"{stats['res_loss']:.3e}",
                    lr=f"{lr:.2e}",
                )

            if not show_progress and (ep % verbose_freq == 0 or ep == 1):
                ela = (time.time() - t0) / 60.0
                print(f"Epoch {ep:5d}/{epochs} | "
                      f"loss={stats['loss']:.4e}  "
                      f"res={stats['res_loss']:.4e}  "
                      f"d0={stats['d0_loss']:.4e}  "
                      f"d1={stats['d1_loss']:.4e}  "
                      f"neu={stats['neu_loss']:.4e}  "
                      f"w_res={stats['w_res_eff']:.3f}  "
                      f"lr={lr:.2e}  time={ela:.1f}min")

        torch.save(self.model.state_dict(), os.path.join(log_dir, "model_final.pth"))
        print(f"\nBest physics loss: {best_loss:.4e}")
        try:
            import pandas as pd
            pd.DataFrame(history).to_csv(os.path.join(log_dir, "history.csv"), index=False)
        except ImportError:
            np.savez(os.path.join(log_dir, "history.npz"), **history)
        return history

    # ------------------------------------------------------------------
    def predict(self, idx):
        """Return predicted u at all 4300 raw mesh points. Shape: (4300,)."""
        self.model.eval()
        with torch.no_grad():
            u = self.S * self.model(self.branch_inputs[idx:idx+1], self.xy_all)
        return u.squeeze(0).cpu().numpy()


# ============================================================================
#  Plotting  (unstructured mesh — Delaunay triangulation)
# ============================================================================

def _tri(xy):
    return mtri.Triangulation(xy[:, 0], xy[:, 1])


def plot_three_panel(xy, u_ref, u_pred, save_dir, fname, label=""):
    triang = _tri(xy)
    err    = np.abs(u_ref - u_pred)
    rel    = np.sqrt(np.sum((u_ref - u_pred)**2) / (np.sum(u_ref**2) + 1e-12))
    vmin, vmax = u_ref.min(), u_ref.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    c0 = axes[0].tricontourf(triang, u_ref,  levels=64, cmap="jet", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"COMSOL  {label}"); axes[0].set_aspect("equal")
    plt.colorbar(c0, ax=axes[0])

    c1 = axes[1].tricontourf(triang, u_pred, levels=64, cmap="jet", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"PI-DeepONet  rel-L2={rel*100:.2f}%"); axes[1].set_aspect("equal")
    plt.colorbar(c1, ax=axes[1])

    c2 = axes[2].tricontourf(triang, err,   levels=64, cmap="hot")
    axes[2].set_title(f"|Error|  max={err.max():.3e}"); axes[2].set_aspect("equal")
    plt.colorbar(c2, ax=axes[2])

    plt.tight_layout()
    path = os.path.join(save_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)


def plot_error_summary(v0_values, rel_errors, save_dir):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(v0_values, rel_errors * 100, color="steelblue", alpha=0.8)
    ax.axhline(1.0, color="red", linestyle="--", lw=1, label="1% threshold")
    ax.set_xlabel("v0 (Dirichlet BC at x=0)")
    ax.set_ylabel("Rel-L2 error (%)")
    ax.set_title("PI-DeepONet rel-L2 vs COMSOL per v0")
    ax.legend(); plt.tight_layout()
    path = os.path.join(save_dir, "error_vs_v0.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)


def plot_history(history, save_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, v in history.items():
        ax.plot(np.log10(np.array(v) + 1e-15), label=k)
    ax.set_xlabel("Epoch"); ax.set_ylabel("log10(loss)")
    ax.set_title("Training History"); ax.legend(); ax.grid(True)
    plt.tight_layout()
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

    parser.add_argument("--forcing_file",  default="Surface_Solution.txt")
    parser.add_argument("--data_file",     default="data_v0.txt")

    parser.add_argument("--p_dim",         type=int,   default=128)
    parser.add_argument("--branch_h",      type=int,   nargs="+", default=[64, 64])
    parser.add_argument("--trunk_h",       type=int,   nargs="+", default=[128, 128])
    parser.add_argument("--n_fourier",     type=int,   default=8)
    parser.add_argument("--output_scale",  type=float, default=20.0,
                        help="S: prediction = S * u_net  (u_max~20 -> S=20 keeps u_net~O(1))")

    parser.add_argument("--w_res",         type=float, default=1.0)
    parser.add_argument("--w_d",           type=float, default=100.0,
                        help="Applied to both Dirichlet walls (x=0 and x=1)")
    parser.add_argument("--w_neu",         type=float, default=10.0,
                        help="Zero-Neumann on y=0, y=1")
    parser.add_argument("--warmup_epochs", type=int,   default=500)

    parser.add_argument("--epochs",          type=int,   default=10000)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--log_dir",         type=str,   default="./output_pideeponet_v0_cuda")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--resume",          type=str,   default=None,
                        help="Path to checkpoint (.pth) to resume from (weights only; LR resets)")
    parser.add_argument("--v0_zero_weight",  type=float, default=1.0,
                        help="Extra loss weight for v0=0 sample (default 1.0 = no upsampling)")
    parser.add_argument("--device",          choices=["auto", "cpu", "cuda"], default="auto",
                        help="Runtime device. Use 'cuda' to require GPU, 'auto' to prefer GPU if available.")
    parser.add_argument("--disable_tf32",    action="store_true",
                        help="Disable TensorFloat-32 matmul/CuDNN acceleration on NVIDIA Ampere+ GPUs.")
    parser.add_argument("--no_progress",     action="store_true",
                        help="Disable the live tqdm epoch progress bar.")
    parser.add_argument("--train_step",      type=float, default=None,
                        help="Step size between training v0 values (e.g. 1.0 = integers 0,1,...,20; "
                             "0.5 = all 41 samples; default None = all samples).")
    parser.add_argument("--num_batches",     type=int, default=1,
                        help="Number of mini-batches per epoch (matches TF create_data_pipeline `batch` kwarg). "
                             "Mini-batch size is ceil(n_v0 / num_batches). Default 1 = full batch.")

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device, allow_tf32=not args.disable_tf32)

    # ---- Data ----
    print("Loading data ...")
    xy, f_vals, v0_values, u_comsol = load_data(args.forcing_file, args.data_file)

    if args.train_step is not None:
        tol = args.train_step * 0.01
        train_mask = np.array([
            abs(v % args.train_step) < tol or
            abs(v % args.train_step - args.train_step) < tol
            for v in v0_values
        ])
        train_indices = np.where(train_mask)[0].tolist()
        test_indices  = np.where(~train_mask)[0].tolist()
        print(f"Train step={args.train_step}: {len(train_indices)} train v0: {v0_values[train_indices]}")
        print(f"                              {len(test_indices)} test  v0 (unseen)")
    else:
        train_mask    = np.ones(len(v0_values), dtype=bool)
        train_indices = None
        test_indices  = []
        print(f"Training on all {len(v0_values)} v0 samples")

    # ---- Model ----
    model = DeepONet2D(
        branch_in_dim=1,
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
    print(f"Branch: 1 -> {args.branch_h} -> {args.p_dim}")
    print(f"Trunk:  {trunk_in} -> {args.trunk_h} -> {args.p_dim}  (Fourier n={args.n_fourier})")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f"Resumed weights from: {args.resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- Train ----
    trainer = PIDeepONetTrainer(
        model=model, xy=xy, f_vals=f_vals, v0_values=v0_values,
        optimizer=optimizer, scheduler=scheduler,
        device=device,
        output_scale=args.output_scale,
        w_res=args.w_res, w_d=args.w_d, w_neu=args.w_neu,
        warmup_epochs=args.warmup_epochs,
        v0_zero_weight=args.v0_zero_weight,
        train_indices=train_indices,
        num_batches=args.num_batches,
    )

    print(f"\n{'='*60}")
    n_train_samples = len(v0_values) if train_indices is None else len(train_indices)
    print(f" Training  |  {n_train_samples} samples  |  {args.epochs} epochs")
    print(f" num_batches={trainer.num_batches}  minibatch_size={trainer.minibatch_size}  "
          f"steps_per_epoch={int(np.ceil(trainer.n_v0 / trainer.minibatch_size))}")
    print(f" w_res={args.w_res}  w_d={args.w_d}  w_neu={args.w_neu}  S={args.output_scale}")
    print(f" warmup={args.warmup_epochs}  lr={args.lr}  v0_zero_weight={args.v0_zero_weight}")
    print(f"{'='*60}\n")

    history = trainer.run(epochs=args.epochs, verbose_freq=50, log_dir=args.log_dir,
                          show_progress=not args.no_progress)

    # ---- Evaluate vs COMSOL ----
    print(f"\n{'='*60}")
    print(f" Evaluation vs COMSOL (at raw 4300 mesh points)")
    print(f"{'='*60}\n")

    best_ckpt = os.path.join(args.log_dir, "model_best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        print("Loaded best checkpoint.")

    # Evaluate all 41 v0 values, tag each as TRAIN or test
    all_indices = list(range(len(v0_values)))
    rel_errors  = np.zeros(len(v0_values))
    for i in all_indices:
        u_pred = trainer.predict(i)
        u_ref  = u_comsol[:, i]
        rel_errors[i] = np.sqrt(np.sum((u_pred - u_ref)**2) / (np.sum(u_ref**2) + 1e-12))

    print(f"\n{'v0':>6}  {'rel-L2 (%)':>12}   {'split':>5}")
    print("-" * 30)
    for i in all_indices:
        tag = "TRAIN" if train_mask[i] else " test"
        print(f"{v0_values[i]:6.1f}  {rel_errors[i]*100:12.4f}   {tag}")

    train_errs = rel_errors[train_mask]
    print(f"\nTrain — Mean: {train_errs.mean()*100:.4f}%  Max: {train_errs.max()*100:.4f}%")
    if test_indices:
        test_errs = rel_errors[~train_mask]
        print(f"Test  — Mean: {test_errs.mean()*100:.4f}%  Max: {test_errs.max()*100:.4f}%")

    # ---- Plots ----
    plot_history(history, save_dir=args.log_dir)
    plot_error_summary(v0_values, rel_errors, save_dir=args.log_dir)
    for v0_plot in [0.0, 5.0, 10.0, 15.0, 20.0]:
        idx = int(np.argmin(np.abs(v0_values - v0_plot)))
        plot_three_panel(xy, u_comsol[:, idx], trainer.predict(idx),
                         save_dir=args.log_dir,
                         fname=f"comparison_v0_{v0_values[idx]:.1f}.png",
                         label=f"v0={v0_values[idx]:.1f}")

    print(f"\nAll outputs saved to {os.path.abspath(args.log_dir)}/")
    print("Done.")


if __name__ == "__main__":
    main()
