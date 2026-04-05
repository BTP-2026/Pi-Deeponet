---
name: Codebase Structure
description: File layout, module responsibilities, and how the pieces fit together in the BTP DeepONet project
type: project
---

## Repository: /home/aditya/dev/btp

### Shared Modules (used by both problems)

**`networks.py`**
- `MLP(in_dim, out_dim, hidden, activation)` — generic feedforward network
- `FourierFeatureEncoder(n_freqs)` — maps (x,y) → 34-dim Fourier features; out_dim = 2 + 4*n_freqs
- `DeepONet2D(branch_in_dim, p, branch_hidden, trunk_hidden, activation, use_fourier, n_fourier)` — branch+trunk DeepONet; forward(branch_input, xy_grid) → (B, n_pts)

**`eval_utils.py`**
- `compute_errors(u_pred, u_true)` → per-sample MSE and rel-L2
- `plot_comparison(x, y, u_true, u_pred, ...)` — 3-panel side-by-side
- `plot_cross_sections(x, y, u_true, u_pred, ...)` — cross-sections at x=0.5, y=0.5
- `plot_training_history(history, ...)` — log-scale loss curves

**`solver.py`**
- `solve_poisson_2d_mixed(x, y, f, bc_values, bc_types, f_scale)` — FD solver with mixed Dirichlet/Neumann BCs; validated at 0.08% rel-L2 vs COMSOL
- `get_boundary_indices(Nx, Ny)` — returns dict of boundary flat indices

### Problem-Specific Files

**`deeponet_2d_gaussian.py`** — Gaussian-forcing Poisson problem
- Imports from `networks`, `eval_utils`, `solver`
- `load_comsol_data(forcing_file, solution_file, N)` — reads Surface_Solution.txt + Poisson_SOlution.txt, interpolates scattered→regular N×N grid via griddata
- `validate_fd_solver(...)` — compares FD vs COMSOL
- `GaussianPdeTrainer` — training loop with data + Dirichlet + Neumann + residual losses
- `plot_three_panel(...)` — PINTO-style 3-panel with jet colormap
- Output: `./output_2d_gaussian_v2/`

**`deeponet_2d_poisson.py`** — original Poisson problem (random forcing, imports from shared modules)

### Data Files
- `Surface_Solution.txt` — COMSOL forcing function (scattered ~4300 pts, 3 cols: X Y F)
- `Poisson_SOlution.txt` — COMSOL solution (scattered ~4300 pts, 3 cols: X Y U)
- `Gaussian.xlsx` — original data reference (superseded by .txt files)

### Output Directories
- `./output_2d_gaussian/` — older runs (pre-Fourier)
- `./output_2d_gaussian_v2/` — current best run (Fourier + p512 + cosine decay)
  - `model_best.pth` — best checkpoint by data_loss
  - `model_final.pth` — final checkpoint
  - `comparison_3panel.png` — main result plot
  - `training_history.png` — loss curves

### Excluded from work
- `Poisson-test/` directory — do not touch
