"""
pi_probe.py
===========
Manual tuning probes for pideeponet_2d_neumann.py.

Each probe runs the PI-DeepONet for a short number of epochs with a specific
set of hyperparameters. After each probe, read the loss breakdown to decide
what to change next.

Usage
-----
  python pi_probe.py --probe 1              # baseline, 500 epochs
  python pi_probe.py --probe 2              # reduce Neumann weight
  python pi_probe.py --probe 3              # increase residual weight
  python pi_probe.py --probe 4              # longer confirm run, 2000 epochs
  python pi_probe.py --probe 1 --epochs 200 # quick look at loss balance

Outputs: ./pi_probe_results/probe_N/  (history.csv, model_best.pth, plots)

Manual tuning guide
-------------------
After each probe look at the LAST printed loss line:
  - If dir_loss > 0.01       → increase w_d (try 2x)
  - If neu_loss > 0.1        → increase w_n (try 2x)
  - If res_loss flat/large   → increase w_res; check warmup is done
  - If all three < 0.001     → training is converging, run probe 4 to confirm

Key facts about this problem:
  - u_max ≈ 26  →  S=25 keeps u_net ≈ O(1)
  - Neumann targets range 0–20, so L_n dominates early unless w_n is small
  - Residual warmup is 500 epochs; res_loss only counts fully after that
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from networks import DeepONet2D
from pideeponet_2d_neumann import (
    load_forcing, load_comsol_solutions, PIDeepONetTrainer,
    plot_three_panel, plot_error_summary,
)
from eval_utils import plot_training_history

# ============================================================================
#  Probe definitions
# ============================================================================
#
# Each probe encodes one hypothesis. Run them in order; stop early if a probe
# clearly converges and confirm with probe 4.
#
# Architecture is fixed to p=128 / trunk=[128,128] for all probes (fast).
# Switch to p=256 / trunk=[256,256,256] only for the final production run.

PROBES = {
    1: dict(
        label="Baseline — check initial loss balance",
        note=(
            "Observe the loss breakdown at epoch 1 and epoch 500.\n"
            "Expect: neu_loss >> dir_loss early because v0 ranges 0–20.\n"
            "Action: if dir_loss is not falling by ep 200, move to probe 2."
        ),
        output_scale=25.0, w_res=1.0, w_d=100.0, w_n=10.0,
        warmup_epochs=500, lr=1e-3, epochs=500,
    ),
    2: dict(
        label="Reduce Neumann weight — prevent early dominance",
        note=(
            "Lower w_n so Dirichlet and Neumann contribute equally early.\n"
            "Action: if both dir and neu converge, try probe 3 to add residual pressure."
        ),
        output_scale=25.0, w_res=1.0, w_d=100.0, w_n=3.0,
        warmup_epochs=500, lr=1e-3, epochs=500,
    ),
    3: dict(
        label="Increase residual weight — enforce PDE harder",
        note=(
            "After BCs are roughly satisfied (dir<0.01, neu<0.1), push residual.\n"
            "Action: if res_loss drops significantly vs probe 2, this is the winner config."
        ),
        output_scale=25.0, w_res=5.0, w_d=100.0, w_n=3.0,
        warmup_epochs=500, lr=1e-3, epochs=500,
    ),
    4: dict(
        label="Longer confirmation — 2000 epochs with best config",
        note=(
            "Run the winner config longer to see if losses keep falling.\n"
            "Also evaluates vs COMSOL at the end — this is the quality signal.\n"
            "If mean rel-L2 < 10%, scale up to p=256 / trunk=[256,256,256] for final."
        ),
        output_scale=25.0, w_res=5.0, w_d=100.0, w_n=3.0,
        warmup_epochs=500, lr=1e-3, epochs=2000,
    ),
}


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe",   type=int, required=True, choices=PROBES.keys())
    parser.add_argument("--epochs",  type=int, default=None,
                        help="Override probe default epoch count")
    parser.add_argument("--forcing_file", type=str, default="Surface_Solution.txt")
    parser.add_argument("--data_file",    type=str, default="data_v0.txt")
    parser.add_argument("--grid_size",    type=int, default=31)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    cfg = PROBES[args.probe]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    log_dir = f"./pi_probe_results/probe_{args.probe}"
    os.makedirs(log_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print(f" Probe {args.probe}: {cfg['label']}")
    print(f" Epochs: {epochs}  |  S={cfg['output_scale']}  "
          f"w_res={cfg['w_res']}  w_d={cfg['w_d']}  w_n={cfg['w_n']}")
    print(f" Output: {log_dir}")
    print(f"{'='*60}\n")

    N = args.grid_size
    x, y, f_comsol = load_forcing(args.forcing_file, N=N)
    f_raw = f_comsol / 100.0
    v0_values, u_grids = load_comsol_solutions(args.data_file, x, y)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepONet2D(
        branch_in_dim=N * N + 1,
        p=128,
        branch_hidden=(256, 256),
        trunk_hidden=(128, 128),
        activation=nn.SiLU,
        use_fourier=True,
        n_fourier=8,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: p=128 trunk=[128,128]  |  {n_params:,} params  |  device={device}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    trainer = PIDeepONetTrainer(
        model=model, x=x, y=y,
        v0_values=v0_values, f_raw=f_raw,
        optimizer=optimizer, scheduler=scheduler,
        device=device, f_scale=100.0,
        output_scale=cfg["output_scale"],
        w_res=cfg["w_res"], w_d=cfg["w_d"], w_n=cfg["w_n"],
        warmup_epochs=cfg["warmup_epochs"],
    )

    history = trainer.run(epochs=epochs, verbose_freq=50, log_dir=log_dir)

    # ---- Evaluate vs COMSOL ----
    print(f"\n{'='*60}")
    print(f" Evaluation vs COMSOL")
    print(f"{'='*60}\n")

    best_ckpt = os.path.join(log_dir, "model_best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device,
                                         weights_only=True))

    rel_errors = []
    for i, v0 in enumerate(v0_values):
        u_pred = trainer.predict(i)
        u_ref = u_grids[i]
        rel = float(np.sqrt(np.sum((u_pred - u_ref) ** 2) /
                            (np.sum(u_ref ** 2) + 1e-12)))
        rel_errors.append(rel)
    rel_errors = np.array(rel_errors)

    print(f"Mean rel-L2 vs COMSOL: {rel_errors.mean()*100:.2f}%  "
          f"Max: {rel_errors.max()*100:.2f}%")

    # ---- Plots ----
    plot_training_history(history, save_dir=log_dir)
    plot_error_summary(v0_values, rel_errors, save_dir=log_dir)
    for v0_plot in [0.0, 10.0, 20.0]:
        idx = int(np.argmin(np.abs(v0_values - v0_plot)))
        plot_three_panel(x, y, u_grids[idx], trainer.predict(idx),
                         save_dir=log_dir,
                         fname=f"comparison_v0_{v0_values[idx]:.1f}.png",
                         ref_label=f"COMSOL v0={v0_values[idx]:.1f}")

    # ---- What to do next ----
    last = {k: history[k][-1] for k in history}
    print(f"\n{'='*60}")
    print(f" Loss at final epoch: "
          f"res={last['res_loss']:.3e}  dir={last['dir_loss']:.3e}  "
          f"neu={last['neu_loss']:.3e}")
    print(f"\n Notes for probe {args.probe}:")
    for line in cfg["note"].splitlines():
        print(f"   {line}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
