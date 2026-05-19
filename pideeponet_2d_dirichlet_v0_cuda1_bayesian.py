"""
pideeponet_2d_dirichlet_v0_cuda1_bayesian.py
=============================================
Physics-Informed DeepONet for 2D Poisson with variable Dirichlet BC at x=0,
**multiple forcing functions**, and **residual-based active learning (RAD)**.

PDE:   -nabla^2 u = f_i(x,y)   on [0,1]^2
BCs:   u = v0_i  on x=0        (Dirichlet, variable)
       u = 0     on x=1        (Dirichlet, fixed)
       du/dy = 0 on y=0, y=1   (Neumann, zero flux)

Each sample i has its own Gaussian forcing f_i centred at (x0_i, y0_i)
and Dirichlet value v0_i.

Active Learning (RAD)
---------------------
Instead of evaluating the PDE residual on all ~28k interior points every
step, we sample n_collocation points from an adaptive distribution:

    p(x_j) ∝ |r(x_j)|^k + c

where r is the PDE residual averaged across training samples. The weights
are recomputed every resample_every epochs on the full interior mesh.
(Wu et al., "A comprehensive study of non-adaptive and residual-based
adaptive sampling for PINNs", CMAME 2023.)

Data
----
  multiple-forcing-f.txt : 28570 COMSOL mesh pts x 41 forcing functions
  multiple-forcing-u.txt : 28570 COMSOL mesh pts x 41 solutions (eval only)

Architecture
------------
  Branch : (v0, x0, y0) -> MLP -> p          (dim=3)
  Trunk  : (x,y) -> Fourier features -> MLP -> p
  Output : dot(branch, trunk) + bias
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
    Load raw COMSOL mesh data with multiple forcing functions.

    Returns
    -------
    xy        : (N, 2)   float32  mesh point coordinates
    f_all     : (N, 41)  float32  per-sample forcing values
    v0_values : (41,)    float64  Dirichlet v0 per sample
    x0_values : (41,)    float64  Gaussian centre x0 per sample
    y0_values : (41,)    float64  Gaussian centre y0 per sample
    u_comsol  : (N, 41)  float32  COMSOL solutions (eval only)
    """
    v0_values, x0_values, y0_values = [], [], []
    with open(data_file) as fh:
        for line in fh:
            if not line.startswith('%'):
                break
            for m in re.finditer(
                    r'v0=([0-9.]+)\s*V,\s*x0=([0-9.]+)\s*m,\s*y0=([0-9.]+)\s*m', line):
                v0_values.append(float(m.group(1)))
                x0_values.append(float(m.group(2)))
                y0_values.append(float(m.group(3)))
    v0_values = np.array(v0_values)
    x0_values = np.array(x0_values)
    y0_values = np.array(y0_values)

    print("Loading forcing file (may take a moment for large meshes)...")
    surf = np.loadtxt(forcing_file, comments='%')
    xy    = surf[:, :2].astype(np.float32)
    f_all = surf[:, 2:].astype(np.float32)          # (N, 41)

    print("Loading solution file...")
    raw      = np.loadtxt(data_file, comments='%')
    u_comsol = raw[:, 2:].astype(np.float32)        # (N, 41)

    print(f"Mesh: {len(xy)} pts")
    print(f"Forcing: {f_all.shape[1]} functions  "
          f"f in [{f_all.min():.3f}, {f_all.max():.3f}]")
    print(f"Samples: {len(v0_values)}  "
          f"v0=[{v0_values[0]:.1f}, {v0_values[-1]:.1f}]  "
          f"x0=[{x0_values[0]:.2f}, {x0_values[-1]:.2f}]  "
          f"y0=[{y0_values[0]:.2f}, {y0_values[-1]:.2f}]  "
          f"u in [{u_comsol.min():.3f}, {u_comsol.max():.3f}]")
    return xy, f_all, v0_values, x0_values, y0_values, u_comsol


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

class BayesianLossWeights(nn.Module):
    """
    Learn physics-loss weights with Bayesian uncertainty weighting.

    Each trainable log-variance s_i gives precision_i = exp(-s_i), and the
    total objective is sum_i precision_i * L_i + s_i.
    """

    def __init__(self, init_res=1.0, init_d=100.0, init_neu=10.0):
        super().__init__()
        self.log_var_res = nn.Parameter(self._init_log_var(init_res))
        self.log_var_d = nn.Parameter(self._init_log_var(init_d))
        self.log_var_neu = nn.Parameter(self._init_log_var(init_neu))

    @staticmethod
    def _init_log_var(initial_weight):
        initial_weight = max(float(initial_weight), 1e-12)
        return torch.tensor(-np.log(initial_weight), dtype=torch.float32)

    def forward(self, L_res, L_d, L_neu):
        return (
            torch.exp(-self.log_var_res) * L_res + self.log_var_res +
            torch.exp(-self.log_var_d) * L_d + self.log_var_d +
            torch.exp(-self.log_var_neu) * L_neu + self.log_var_neu
        )

    def weights(self):
        with torch.no_grad():
            return {
                "w_res": float(torch.exp(-self.log_var_res).detach().cpu()),
                "w_d": float(torch.exp(-self.log_var_d).detach().cpu()),
                "w_neu": float(torch.exp(-self.log_var_neu).detach().cpu()),
                "log_var_res": float(self.log_var_res.detach().cpu()),
                "log_var_d": float(self.log_var_d.detach().cpu()),
                "log_var_neu": float(self.log_var_neu.detach().cpu()),
            }


class PIDeepONetTrainer:
    """
    Physics-only training — no data loss.

    Loss weights are learned with Bayesian uncertainty weighting:
        precision_i = exp(-log_var_i)
        Loss = sum_i precision_i*L_i + log_var_i

    The PDE residual still uses a 0->1 warmup multiplier before Bayesian
    balancing so early training is not dominated by second derivatives.
    """

    def __init__(self, model, xy, f_all, v0_values, x0_values, y0_values, *,
                 optimizer, scheduler=None,
                 device=None, output_scale=1.0,
                 loss_balancer=None,
                 warmup_epochs=500, v0_zero_weight=1.0,
                 train_indices=None, num_batches=1,
                 n_collocation=0, resample_every=500,
                 rad_k=1.0, rad_c=1.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_balancer = loss_balancer.to(self.device) if loss_balancer is not None else BayesianLossWeights().to(self.device)
        self.S            = float(output_scale)
        self.warmup_epochs   = int(warmup_epochs)
        self.v0_zero_weight  = float(v0_zero_weight)
        self.v0_values    = v0_values
        self.train_indices = np.arange(len(v0_values)) if train_indices is None else np.array(train_indices, dtype=int)
        if self.train_indices.size == 0:
            raise ValueError("At least one v0 sample must be selected for training.")
        self.train_v0_values = v0_values[self.train_indices]
        self.n_v0         = len(self.train_indices)
        self.num_batches  = max(1, int(num_batches))
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

        # Only store TRAIN data — no test data in the trainer
        train_f = f_all[interior_idx, :][:, self.train_indices]       # (n_interior, n_train)
        self.f_interior = _t(train_f.T)                               # (n_train, n_interior)

        branch_data = np.column_stack([v0_values, x0_values, y0_values]).astype(np.float32)
        self.train_branch_inputs = _t(branch_data[self.train_indices]) # (n_train, 3)

        # RAD active learning
        n_int = self.interior_coords.shape[0]
        self.use_rad = n_collocation > 0 and n_collocation < n_int
        self.n_collocation   = n_collocation if self.use_rad else n_int
        self.resample_every  = int(resample_every)
        self.rad_k           = float(rad_k)
        self.rad_c           = float(rad_c)
        self.rad_weights = torch.ones(n_int, device=self.device) / n_int
        if self.use_rad:
            print(f"RAD active learning: {self.n_collocation}/{n_int} collocation pts, "
                  f"resample every {self.resample_every} epochs, k={self.rad_k}, c={self.rad_c}")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _recompute_rad_weights(self):
        """
        Recompute RAD sampling weights over all interior points.

        For each training sample we evaluate |-(u_xx + u_yy) - f|^k on
        the full interior mesh, then average across samples.  The final
        probability vector is p_j = (mean_resid_j^k + c) / Z.

        Autograd is needed for the Laplacian but we process one sample at
        a time and never build a training graph, so memory stays low.
        """
        self.model.eval()
        n_int = self.interior_coords.shape[0]
        accum = torch.zeros(n_int, device=self.device)

        for pos in range(len(self.train_indices)):
            xy_leaf = self.interior_coords.clone().requires_grad_(True)   # (n_int, 2)
            b_in = self.train_branch_inputs[pos : pos + 1]               # (1, 3)

            with torch.enable_grad():
                if self.model.fourier_enc is not None:
                    xy_enc = self.model.fourier_enc(xy_leaf)
                else:
                    xy_enc = xy_leaf
                t_out = self.model.trunk(xy_enc)                          # (n_int, p)
                b_out = self.model.branch(b_in)                           # (1, p)
                u = (b_out * t_out).sum(dim=-1) + self.model.bias         # (n_int,)
                u = self.S * u

                g1 = torch.autograd.grad(u.sum(), xy_leaf, create_graph=True)[0]
                u_xx = torch.autograd.grad(g1[:, 0].sum(), xy_leaf, retain_graph=True)[0][:, 0]
                u_yy = torch.autograd.grad(g1[:, 1].sum(), xy_leaf)[0][:, 1]

            f_i = self.f_interior[pos]                                    # (n_int,)
            res = -(u_xx + u_yy) - f_i
            accum += res.abs().pow(self.rad_k)

        accum /= len(self.train_indices)
        self.rad_weights = accum + self.rad_c
        self.rad_weights /= self.rad_weights.sum()
        self.model.train()

    # ------------------------------------------------------------------
    def _w_res_now(self, epoch):
        if self.warmup_epochs <= 0:
            return 1.0
        return min(float(epoch) / self.warmup_epochs, 1.0)

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
        b_input = self.train_branch_inputs.index_select(0, idx_t)   # (B, 3)
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

        # ---- PDE residual : -(u_xx + u_yy) - f_i = 0  (per-sample f) ----
        col_idx = self._current_col_idx                              # (n_col,) set in train_step
        xy_col = self.interior_coords[col_idx]                       # (n_col, 2)
        n_col = xy_col.shape[0]
        xy_int_b = xy_col.unsqueeze(0).expand(B, n_col, 2).contiguous()
        xy_int_b.requires_grad_(True)
        u_int = self.S * self._forward_per_sample_xy(b_input, xy_int_b)  # (B, n_col)
        g1 = torch.autograd.grad(u_int.sum(), xy_int_b, create_graph=True)[0]   # (B, n_col, 2)
        u_xx = torch.autograd.grad(g1[..., 0].sum(), xy_int_b, create_graph=True)[0][..., 0]
        u_yy = torch.autograd.grad(g1[..., 1].sum(), xy_int_b, create_graph=True)[0][..., 1]
        f_batch = self.f_interior[batch_positions][:, col_idx]        # (B, n_col)
        res = -(u_xx + u_yy) - f_batch                              # (B, n_col)
        L_res_per = torch.mean(res ** 2, dim=1)                     # (B,)
        L_res = (w_t * L_res_per).sum()

        L_res_warm = w_res * L_res
        L_d = L_d0 + L_d1
        loss = self.loss_balancer(L_res_warm, L_d, L_neu)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        bayes_stats = self.loss_balancer.weights()
        return (loss.item(), L_res.item(), L_d0.item(), L_d1.item(), L_neu.item(), bayes_stats)

    # ------------------------------------------------------------------
    def train_step(self, epoch):
        """
        One epoch = full shuffle of train_indices, split into mini-batches of
        size ceil(n_v0 / num_batches), one optimizer step per mini-batch.

        RAD active learning: every resample_every epochs (and at epoch 1),
        recompute the sampling weights on the full interior mesh.  Each
        epoch, draw n_collocation indices from the current distribution.
        """
        # RAD: recompute weights periodically
        rad_resampled = False
        if self.use_rad and (epoch == 1 or epoch % self.resample_every == 0):
            self._recompute_rad_weights()
            rad_resampled = True

        # Draw collocation indices for this epoch
        if self.use_rad:
            self._current_col_idx = torch.multinomial(
                self.rad_weights, self.n_collocation, replacement=False)
        else:
            self._current_col_idx = torch.arange(
                self.interior_coords.shape[0], device=self.device)

        self.model.train()
        w_res = self._w_res_now(epoch)

        perm = np.random.permutation(self.n_v0)
        chunks = [perm[k:k + self.minibatch_size]
                  for k in range(0, self.n_v0, self.minibatch_size)]

        agg_loss = agg_res = agg_d0 = agg_d1 = agg_neu = 0.0
        last_bayes_stats = self.loss_balancer.weights()
        for chunk in chunks:
            l, r, d0, d1, n, last_bayes_stats = self._minibatch_step(chunk, w_res)
            agg_loss += l; agg_res += r; agg_d0 += d0; agg_d1 += d1; agg_neu += n

        n_steps = max(1, len(chunks))
        stats = {
            "loss":     agg_loss / n_steps,
            "res_loss": agg_res / n_steps,
            "d0_loss":  agg_d0 / n_steps,
            "d1_loss":  agg_d1 / n_steps,
            "neu_loss": agg_neu / n_steps,
            "res_warmup": w_res,
            "n_minibatches": n_steps,
            "rad_resampled": rad_resampled,
        }
        stats.update(last_bayes_stats)
        return stats

    # ------------------------------------------------------------------
    def run(self, epochs=10000, verbose_freq=50, log_dir="./output_pideeponet_v0",
            show_progress=True):
        os.makedirs(log_dir, exist_ok=True)
        keys = [
            "loss", "res_loss", "d0_loss", "d1_loss", "neu_loss",
            "res_warmup",
            "w_res", "w_d", "w_neu",
            "log_var_res", "log_var_d", "log_var_neu",
        ]
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
                torch.save(self.loss_balancer.state_dict(),
                           os.path.join(log_dir, "loss_balancer_best.pth"))

            lr = self.optimizer.param_groups[0]["lr"]
            if show_progress:
                progress.set_postfix(
                    loss=f"{stats['loss']:.3e}",
                    res=f"{stats['res_loss']:.3e}",
                    lr=f"{lr:.2e}",
                )

            if not show_progress and (ep % verbose_freq == 0 or ep == 1):
                ela = (time.time() - t0) / 60.0
                rad_tag = "  [RAD resampled]" if stats.get("rad_resampled") else ""
                print(f"Epoch {ep:5d}/{epochs} | "
                      f"loss={stats['loss']:.4e}  "
                      f"res={stats['res_loss']:.4e}  "
                      f"d0={stats['d0_loss']:.4e}  "
                      f"d1={stats['d1_loss']:.4e}  "
                      f"neu={stats['neu_loss']:.4e}  "
                      f"w_res={stats['w_res']:.3e}  "
                      f"w_d={stats['w_d']:.3e}  "
                      f"w_neu={stats['w_neu']:.3e}  "
                      f"res_warmup={stats['res_warmup']:.3f}  "
                      f"lr={lr:.2e}  time={ela:.1f}min"
                      f"{rad_tag}")

        torch.save(self.model.state_dict(), os.path.join(log_dir, "model_final.pth"))
        torch.save(self.loss_balancer.state_dict(), os.path.join(log_dir, "loss_balancer_final.pth"))
        print(f"\nBest physics loss: {best_loss:.4e}")
        try:
            import pandas as pd
            pd.DataFrame(history).to_csv(os.path.join(log_dir, "history.csv"), index=False)
        except ImportError:
            np.savez(os.path.join(log_dir, "history.npz"), **history)
        return history

    # ------------------------------------------------------------------
    def predict(self, branch_input):
        """
        Predict u at all mesh points for a single sample.

        Parameters
        ----------
        branch_input : array-like, shape (3,)
            (v0, x0, y0) for the sample to predict.

        Returns
        -------
        u : ndarray, shape (N,)
        """
        self.model.eval()
        b = torch.as_tensor(branch_input, dtype=torch.float32,
                            device=self.device).unsqueeze(0)          # (1, 3)
        with torch.no_grad():
            u = self.S * self.model(b, self.xy_all)
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


def plot_error_summary_train_test(v0_values, rel_errors,
                                  train_indices, test_indices, save_dir):
    """Bar chart with train (blue) and test (orange) errors side by side."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    w = 0.35
    train_v0 = v0_values[train_indices]
    test_v0  = v0_values[test_indices]
    train_err = rel_errors[train_indices] * 100
    test_err  = rel_errors[test_indices] * 100

    ax.bar(train_v0 - w/2, train_err, w, color="steelblue", alpha=0.85,
           label=f"Train ({len(train_indices)}) — mean {train_err.mean():.4f}%")
    ax.bar(test_v0 + w/2, test_err, w, color="darkorange", alpha=0.85,
           label=f"Test ({len(test_indices)}) — mean {test_err.mean():.4f}%")
    ax.axhline(1.0, color="red", ls="--", lw=1, label="1% threshold")
    ax.set_xlabel("v0 (Dirichlet BC at x=0)")
    ax.set_ylabel("Rel-L2 error (%)")
    ax.set_title("PI-DeepONet rel-L2 vs COMSOL — Train / Test split")
    ax.legend(); plt.tight_layout()
    path = os.path.join(save_dir, "error_vs_v0_train_test.png")
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

    parser.add_argument("--forcing_file",  default="multiple-forcing-f.txt")
    parser.add_argument("--data_file",     default="multiple-forcing-u.txt")

    parser.add_argument("--p_dim",         type=int,   default=128)
    parser.add_argument("--branch_h",      type=int,   nargs="+", default=[64, 64])
    parser.add_argument("--trunk_h",       type=int,   nargs="+", default=[128, 128])
    parser.add_argument("--n_fourier",     type=int,   default=8)
    parser.add_argument("--output_scale",  type=float, default=1.0,
                        help="S: prediction = S * u_net  (default 1.0 = no output scaling)")

    parser.add_argument("--bayes_init_res", type=float, default=1.0,
                        help="Initial Bayesian inverse-variance weight for PDE residual")
    parser.add_argument("--bayes_init_d",   type=float, default=100.0,
                        help="Initial Bayesian inverse-variance weight for both Dirichlet walls")
    parser.add_argument("--bayes_init_neu", type=float, default=10.0,
                        help="Initial Bayesian inverse-variance weight for zero-Neumann walls")
    parser.add_argument("--warmup_epochs", type=int,   default=500)

    parser.add_argument("--epochs",          type=int,   default=10000)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--log_dir",         type=str,   default="./output_multi_forcing")
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
    parser.add_argument("--train_v0",        type=float, default=None,
                        help="Train on only the nearest available v0 value, e.g. --train_v0 0.")
    parser.add_argument("--train_step",      type=float, default=None,
                        help="Train on v0 values at this step size (e.g. 1.0 picks v0=0,1,2,...,20). "
                             "Remaining samples become the test set.")
    parser.add_argument("--num_batches",     type=int, default=8,
                        help="Number of mini-batches per epoch. "
                             "Mini-batch size is ceil(n_v0 / num_batches). "
                             "Default 8 to fit 28k mesh pts on GPU.")

    # RAD active learning
    parser.add_argument("--n_collocation",  type=int, default=0,
                        help="Number of collocation points sampled per epoch via RAD. "
                             "0 = use all interior points (no active learning).")
    parser.add_argument("--resample_every", type=int, default=500,
                        help="Recompute RAD sampling weights every N epochs.")
    parser.add_argument("--rad_k",          type=float, default=1.0,
                        help="Exponent k in the RAD weight formula |r|^k.")
    parser.add_argument("--rad_c",          type=float, default=1.0,
                        help="Floor constant c in the RAD weight formula (prevents zero prob).")

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device, allow_tf32=not args.disable_tf32)

    # ---- Data ----
    print("Loading data ...")
    xy, f_all, v0_values, x0_values, y0_values, u_comsol = load_data(
        args.forcing_file, args.data_file)
    train_indices = None
    test_indices = None
    if args.train_step is not None:
        step = float(args.train_step)
        train_v0_targets = np.arange(v0_values.min(), v0_values.max() + step / 2, step)
        train_indices = []
        for tv in train_v0_targets:
            idx = int(np.argmin(np.abs(v0_values - tv)))
            if idx not in train_indices:
                train_indices.append(idx)
        test_indices = sorted(set(range(len(v0_values))) - set(train_indices))
        print(f"Train/test split (step={step}): "
              f"{len(train_indices)} train, {len(test_indices)} test")
        print(f"  Train v0: {v0_values[train_indices]}")
        print(f"  Test  v0: {v0_values[test_indices]}")
    elif args.train_v0 is not None:
        nearest_idx = int(np.argmin(np.abs(v0_values - args.train_v0)))
        train_indices = [nearest_idx]
        print(f"Training subset: requested v0={args.train_v0:g}; "
              f"using nearest available v0={v0_values[nearest_idx]:.1f} "
              f"(index {nearest_idx})")

    # ---- Model ----
    branch_in_dim = 3  # (v0, x0, y0)
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
    print(f"Branch: {branch_in_dim} (v0,x0,y0) -> {args.branch_h} -> {args.p_dim}")
    print(f"Trunk:  {trunk_in} -> {args.trunk_h} -> {args.p_dim}  (Fourier n={args.n_fourier})")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f"Resumed weights from: {args.resume}")

    loss_balancer = BayesianLossWeights(
        init_res=args.bayes_init_res,
        init_d=args.bayes_init_d,
        init_neu=args.bayes_init_neu,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_balancer.parameters()),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- Train ----
    trainer = PIDeepONetTrainer(
        model=model, xy=xy, f_all=f_all,
        v0_values=v0_values, x0_values=x0_values, y0_values=y0_values,
        optimizer=optimizer, scheduler=scheduler,
        device=device,
        output_scale=args.output_scale,
        loss_balancer=loss_balancer,
        warmup_epochs=args.warmup_epochs,
        v0_zero_weight=args.v0_zero_weight,
        train_indices=train_indices,
        num_batches=args.num_batches,
        n_collocation=args.n_collocation,
        resample_every=args.resample_every,
        rad_k=args.rad_k,
        rad_c=args.rad_c,
    )

    print(f"\n{'='*60}")
    n_train_samples = len(v0_values) if train_indices is None else len(train_indices)
    n_test_samples = 0 if test_indices is None else len(test_indices)
    split_info = f"  ({n_test_samples} held-out test)" if n_test_samples else ""
    print(f" Training  |  {n_train_samples} samples{split_info}  |  {args.epochs} epochs")
    print(f" num_batches={trainer.num_batches}  minibatch_size={trainer.minibatch_size}  "
          f"steps_per_epoch={int(np.ceil(trainer.n_v0 / trainer.minibatch_size))}")
    print(f" Bayesian init weights: res={args.bayes_init_res}  "
          f"d={args.bayes_init_d}  neu={args.bayes_init_neu}  S={args.output_scale}")
    print(f" warmup={args.warmup_epochs}  lr={args.lr}  v0_zero_weight={args.v0_zero_weight}")
    if trainer.use_rad:
        print(f" RAD active learning: n_collocation={trainer.n_collocation}  "
              f"resample_every={trainer.resample_every}  k={trainer.rad_k}  c={trainer.rad_c}")
    else:
        print(f" RAD: disabled (using all {trainer.n_collocation} interior points)")
    print(f"{'='*60}\n")

    history = trainer.run(epochs=args.epochs, verbose_freq=50, log_dir=args.log_dir,
                          show_progress=not args.no_progress)
    final_weights = trainer.loss_balancer.weights()
    print("\nFinal Bayesian loss weights")
    print(f"  w_res={final_weights['w_res']:.6e}")
    print(f"  w_d={final_weights['w_d']:.6e}")
    print(f"  w_neu={final_weights['w_neu']:.6e}")

    # ---- Evaluate vs COMSOL ----
    best_ckpt = os.path.join(args.log_dir, "model_best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        print("\nLoaded best checkpoint.")

    # Build branch inputs for ALL samples (outside the trainer — eval only)
    all_branch = np.column_stack([v0_values, x0_values, y0_values]).astype(np.float32)

    def _eval_set(indices, set_name):
        """Evaluate a set of samples and print results."""
        idx_list = list(indices)
        v0s = v0_values[idx_list]
        x0s = x0_values[idx_list]
        y0s = y0_values[idx_list]
        errs = np.zeros(len(idx_list))
        for j, i in enumerate(idx_list):
            u_pred = trainer.predict(all_branch[i])
            u_ref  = u_comsol[:, i]
            errs[j] = np.sqrt(
                np.sum((u_pred - u_ref)**2) / (np.sum(u_ref**2) + 1e-12))

        print(f"\n{'='*60}")
        print(f" {set_name} evaluation vs COMSOL  ({len(idx_list)} samples)")
        print(f"{'='*60}")
        print(f"\n{'v0':>6}  {'x0':>6}  {'y0':>6}  {'rel-L2 (%)':>12}")
        print("-" * 36)
        for v0, x0, y0, err in zip(v0s, x0s, y0s, errs):
            print(f"{v0:6.1f}  {x0:6.2f}  {y0:6.2f}  {err*100:12.4f}")
        print(f"\n{set_name} — Mean rel-L2: {errs.mean()*100:.4f}%  "
              f"Max: {errs.max()*100:.4f}%")
        return v0s, errs

    # Evaluate all samples, then split for reporting
    all_indices = list(range(len(v0_values)))
    all_rel_errors = np.zeros(len(v0_values))
    for i in all_indices:
        u_pred = trainer.predict(all_branch[i])
        u_ref  = u_comsol[:, i]
        all_rel_errors[i] = np.sqrt(
            np.sum((u_pred - u_ref)**2) / (np.sum(u_ref**2) + 1e-12))

    if test_indices is not None:
        train_v0s, train_errs = _eval_set(train_indices, "TRAIN")
        test_v0s, test_errs = _eval_set(test_indices, "TEST")
    else:
        eval_indices = train_indices if train_indices is not None else all_indices
        _eval_set(eval_indices, "ALL")

    # ---- Plots ----
    plot_history(history, save_dir=args.log_dir)

    if test_indices is not None:
        plot_error_summary_train_test(
            v0_values, all_rel_errors,
            train_indices, test_indices,
            save_dir=args.log_dir)
    else:
        eval_idx = train_indices if train_indices is not None else all_indices
        plot_error_summary(v0_values[eval_idx], all_rel_errors[eval_idx],
                           save_dir=args.log_dir)

    v0_plots = [0.0, 5.0, 10.0, 15.0, 20.0]
    if test_indices is not None:
        test_v0_set = set(v0_values[test_indices])
        v0_plots += [0.5, 5.5, 10.5, 15.5]
    for v0_plot in sorted(set(v0_plots)):
        idx = int(np.argmin(np.abs(v0_values - v0_plot)))
        tag = ""
        if test_indices is not None:
            tag = " [TEST]" if idx in test_indices else " [TRAIN]"
        plot_three_panel(xy, u_comsol[:, idx], trainer.predict(all_branch[idx]),
                         save_dir=args.log_dir,
                         fname=f"comparison_v0_{v0_values[idx]:.1f}.png",
                         label=f"v0={v0_values[idx]:.1f}{tag}")

    print(f"\nAll outputs saved to {os.path.abspath(args.log_dir)}/")
    print("Done.")


if __name__ == "__main__":
    main()
