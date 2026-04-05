---
name: Gaussian Poisson DeepONet Project
description: Architecture, training results, and key decisions for the 2D Poisson DeepONet with Gaussian forcing
type: project
---

## Problem

2D Poisson PDE: `-∇²u = 100·f(x,y)` on [0,1]×[0,1]
Forcing: fixed tail-end Gaussian `f(x,y) = (rho0/100) * exp(-((x-mu_x)²+(y-mu_y)²)/(2σ²))`
BCs: Dirichlet u=0 on x=0,1 | Neumann du/dy=0 on y=0,1
Data: COMSOL exports `Surface_Solution.txt` (forcing), `Poisson_SOlution.txt` (solution)

## Files

- `deeponet_2d_gaussian.py` — main training + evaluation script
- `networks.py` — shared MLP, FourierFeatureEncoder, DeepONet2D
- `eval_utils.py` — shared plotting utilities (compute_errors, plot_comparison, etc.)
- `solver.py` — FD solver with mixed BC support (`solve_poisson_2d_mixed`)
- Output dir: `./output_2d_gaussian_v2/` (best run)

## Current Best Architecture (as of 2026-04-05)

| Component | Config |
|-----------|--------|
| Branch | MLP(961 → 256 → 256 → 512), SiLU |
| Fourier encoder | n_freqs=8 → trunk input dim=34 |
| Trunk | MLP(34 → 512 → 512 → 512 → 512), SiLU |
| p_dim | 512 |
| Output | dot(branch, trunk) + bias |
| Total params | ~1.25M |

## Best Training Config

- epochs: 20,000
- lr: 1e-3
- scheduler: `CosineAnnealingLR(T_max=20000, eta_min=1e-6)` — smooth decay, NO warm restarts
- Loss: `1.0*L_data + 10.0*L_dirichlet + 5.0*L_neumann + 0.0*L_residual`
- Target normalized to [0,1]: `u_scale = u_true.max()` to prevent Gaussian peak being swamped by near-zero region in MSE
- Best checkpoint saved as `model_best.pth` (tracks min data_loss during training)
- Evaluation loads best checkpoint, not final

## Results Comparison

| Run | Epochs | vs COMSOL rel-L2 | Notes |
|-----|--------|-------------------|-------|
| Original (no Fourier, Tanh) | 40k | ~33% | Unnormalized target |
| SiLU + target norm + warm restarts | 40k | 7.9% | Warm restarts hurt |
| **Fourier + p512 + cosine decay** | **20k** | **0.47%** | Current best |

**Why:** 17× accuracy improvement with half the epochs.

## Key Technical Decisions

### Fourier Feature Encoding (biggest win)
Maps `(x,y) → [x, y, sin(2πkx), cos(2πkx), sin(2πky), cos(2πky)]` for k=1..8
Trunk input: 34-dim instead of 2-dim
**Why:** MLPs suffer from spectral bias — they struggle to learn high-frequency content from raw coordinates. Explicit Fourier features give the trunk the frequencies needed to resolve the sharp Gaussian peak without extra epochs.

### No Warm Restarts
`CosineAnnealingWarmRestarts` with T_mult=2 was destructive here.
At restart at epoch 5000, LR spiked back to 1e-3 and ejected the model from a good minimum it had found at epoch ~1250. Loss went from 1.2e-2 → 8.8e-2 after restart.
Fix: `CosineAnnealingLR(T_max=epochs)` — single smooth decay over full budget.

### Target Normalization
`u_scale = u_true.max()`, train on `u_true / u_scale`, unscale at inference.
**Why:** Gaussian peak (~25 points at max) was swamped by ~900 near-zero boundary points in MSE average. Normalizing gives the peak proper gradient weight.

### Residual Loss Disabled (res_weight=0)
FD ground truth already satisfies the PDE, so residual loss is redundant and its stochastic subsampling adds gradient noise.

### FD Solver Validated
`solve_poisson_2d_mixed` in solver.py: 0.08% rel-L2 vs COMSOL at 31×31 grid.
Training target is FD solution (not raw COMSOL scatter), evaluation is vs both.

## Training Instability Note
Neumann loss occasionally spikes (autograd-computed du/dy at boundary points). These spikes are transient and the model recovers within ~10 epochs. Best-checkpoint saving handles this gracefully.
