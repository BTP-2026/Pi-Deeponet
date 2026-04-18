"""
hparam_search.py
================
Structured hyperparameter search for deeponet_2d_neumann.py

Phase 1  : All configs, 2000 epochs — finds the best region quickly.
Phase 2  : Top-3 configs from Phase 1, 15000 epochs — confirms the winner.

Results
-------
  hparam_results/
    phase1/config_XX/   model_best.pth  history.csv  config.json
    phase2/config_XX/   model_best.pth  history.csv  config.json
    results.csv         all Phase-1 + Phase-2 summary rows
    log.md              human-readable progress log (appended live)
"""

import os
import json
import time
import csv
import itertools
import numpy as np
import torch
import torch.nn as nn

from deeponet_2d_neumann import load_neumann_data, NeumannPdeTrainer
from networks import DeepONet2D


# ============================================================================
#  Search space
# ============================================================================

PHASE1_EPOCHS = 2000
PHASE2_EPOCHS = 15000
GRID_SIZE     = 31
F_SCALE       = 100.0
DATA_FILE     = "data_v0.txt"
FORCING_FILE  = "Surface_Solution.txt"
RESULTS_DIR   = "./hparam_results"
SEED          = 42

# All Phase-1 configurations to try
PHASE1_CONFIGS = []
_id = 1

# --- Group A: vary bc weights (fixed lr=1e-3, p=512, standard arch) ---
for bc_d, bc_n in itertools.product([5.0, 10.0, 20.0], [1.0, 5.0, 10.0]):
    PHASE1_CONFIGS.append({
        "id": f"{_id:02d}",
        "group": "A_weights",
        "lr": 1e-3,
        "bc_d_weight": bc_d,
        "bc_n_weight": bc_n,
        "p_dim": 512,
        "branch_h": [256, 256],
        "trunk_h": [512, 512, 512],
        "n_fourier": 8,
        "batch_size": 16,
    })
    _id += 1

# --- Group B: vary learning rate (baseline bc weights) ---
for lr in [5e-4, 2e-4, 5e-3]:
    PHASE1_CONFIGS.append({
        "id": f"{_id:02d}",
        "group": "B_lr",
        "lr": lr,
        "bc_d_weight": 10.0,
        "bc_n_weight": 5.0,
        "p_dim": 512,
        "branch_h": [256, 256],
        "trunk_h": [512, 512, 512],
        "n_fourier": 8,
        "batch_size": 16,
    })
    _id += 1

# --- Group C: vary architecture ---
for p_dim, trunk_h in [(256, [512, 512, 512]), (512, [512, 512, 512, 512]),
                        (512, [256, 256, 256])]:
    PHASE1_CONFIGS.append({
        "id": f"{_id:02d}",
        "group": "C_arch",
        "lr": 1e-3,
        "bc_d_weight": 10.0,
        "bc_n_weight": 5.0,
        "p_dim": p_dim,
        "branch_h": [256, 256],
        "trunk_h": trunk_h,
        "n_fourier": 8,
        "batch_size": 16,
    })
    _id += 1

# --- Group D: vary Fourier bands ---
for n_fourier in [4, 16]:
    PHASE1_CONFIGS.append({
        "id": f"{_id:02d}",
        "group": "D_fourier",
        "lr": 1e-3,
        "bc_d_weight": 10.0,
        "bc_n_weight": 5.0,
        "p_dim": 512,
        "branch_h": [256, 256],
        "trunk_h": [512, 512, 512],
        "n_fourier": n_fourier,
        "batch_size": 16,
    })
    _id += 1


# ============================================================================
#  Utilities
# ============================================================================

def log(msg, log_path):
    """Append msg to the markdown log and print it."""
    print(msg)
    with open(log_path, 'a') as f:
        f.write(msg + "\n")


def append_csv(path, row: dict, write_header=False):
    mode = 'w' if write_header else 'a'
    with open(path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_model(cfg, branch_in_dim, device):
    model = DeepONet2D(
        branch_in_dim=branch_in_dim,
        p=cfg["p_dim"],
        branch_hidden=tuple(cfg["branch_h"]),
        trunk_hidden=tuple(cfg["trunk_h"]),
        activation=nn.SiLU,
        use_fourier=True,
        n_fourier=cfg["n_fourier"],
    )
    return model.to(device)


def evaluate(trainer, f_raw, v0_values, u_grids):
    """Return mean and max rel-L2 across all v0 samples."""
    rel_errors = []
    for i, v0 in enumerate(v0_values):
        u_pred = trainer.predict(f_raw, v0)
        u_ref  = u_grids[i]
        rel = float(np.sqrt(np.sum((u_pred - u_ref)**2) /
                            (np.sum(u_ref**2) + 1e-12)))
        rel_errors.append(rel)
    return np.mean(rel_errors), np.max(rel_errors), np.array(rel_errors)


def run_config(cfg, epochs, out_dir, x, y, v0_values, u_grids, f_raw,
               log_path, device, verbose_freq=200):
    """Train one configuration and return evaluation metrics."""
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(out_dir, "config.json"), 'w') as f:
        json.dump(cfg, f, indent=2)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    N = GRID_SIZE
    branch_in_dim = N * N + 1
    model = build_model(cfg, branch_in_dim, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    trainer = NeumannPdeTrainer(
        model=model, x=x, y=y,
        v0_values=v0_values, u_grids=u_grids, f_raw=f_raw,
        optimizer=optimizer, scheduler=scheduler,
        device=device, f_scale=F_SCALE,
        res_weight=0.0,
        bc_d_weight=cfg["bc_d_weight"],
        bc_n_weight=cfg["bc_n_weight"],
        batch_size=cfg.get("batch_size", None),
    )

    t0 = time.time()
    history = trainer.run(
        epochs=epochs, verbose_freq=verbose_freq,
        log_dir=out_dir, save_every=0,
    )
    elapsed = time.time() - t0

    # Load best checkpoint for evaluation
    best_ckpt = os.path.join(out_dir, "model_best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    mean_rel, max_rel, per_v0 = evaluate(trainer, f_raw, v0_values, u_grids)

    # Save per-v0 errors
    np.save(os.path.join(out_dir, "per_v0_errors.npy"), per_v0)

    return {
        "mean_rel_l2_pct": round(mean_rel * 100, 4),
        "max_rel_l2_pct":  round(max_rel  * 100, 4),
        "best_data_loss":  round(min(history["data_loss"]), 6),
        "elapsed_min":     round(elapsed / 60, 2),
    }


# ============================================================================
#  Main search loop
# ============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path   = os.path.join(RESULTS_DIR, "log.md")
    csv_path   = os.path.join(RESULTS_DIR, "results.csv")

    # Wipe log for a fresh run
    with open(log_path, 'w') as f:
        f.write("# Hyperparameter Search Log\n\n")
        f.write(f"Device: {device}  |  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Load data once
    log("## Loading data ...", log_path)
    x, y, v0_values, u_grids, f_comsol = load_neumann_data(
        DATA_FILE, FORCING_FILE, N=GRID_SIZE)
    f_raw = f_comsol / F_SCALE

    # ----------------------------------------------------------------
    # Phase 1
    # ----------------------------------------------------------------
    log(f"\n## Phase 1  ({len(PHASE1_CONFIGS)} configs × {PHASE1_EPOCHS} epochs)\n",
        log_path)
    log(f"{'ID':>4}  {'Group':>12}  {'lr':>8}  {'bc_d':>6}  {'bc_n':>6}  "
        f"{'p':>5}  {'n_f':>4}  {'mean%':>8}  {'max%':>8}  {'min_data_loss':>14}  {'time':>6}",
        log_path)
    log("-" * 100, log_path)

    phase1_results = []
    csv_header_written = False

    for cfg in PHASE1_CONFIGS:
        out_dir = os.path.join(RESULTS_DIR, "phase1", f"config_{cfg['id']}")
        log(f"  Running config {cfg['id']} ({cfg['group']}) ...", log_path)

        metrics = run_config(
            cfg, PHASE1_EPOCHS, out_dir,
            x, y, v0_values, u_grids, f_raw,
            log_path, device, verbose_freq=500,
        )

        row = {**cfg, "phase": 1, "epochs": PHASE1_EPOCHS, **metrics}
        row["branch_h"] = str(cfg["branch_h"])
        row["trunk_h"]  = str(cfg["trunk_h"])

        append_csv(csv_path, row, write_header=not csv_header_written)
        csv_header_written = True

        phase1_results.append({"cfg": cfg, **metrics})

        log(f"{cfg['id']:>4}  {cfg['group']:>12}  {cfg['lr']:>8.0e}  "
            f"{cfg['bc_d_weight']:>6.1f}  {cfg['bc_n_weight']:>6.1f}  "
            f"{cfg['p_dim']:>5}  {cfg['n_fourier']:>4}  "
            f"{metrics['mean_rel_l2_pct']:>8.4f}  {metrics['max_rel_l2_pct']:>8.4f}  "
            f"{metrics['best_data_loss']:>14.6e}  {metrics['elapsed_min']:>5.1f}m",
            log_path)

    # Sort by mean rel-L2
    phase1_results.sort(key=lambda r: r["mean_rel_l2_pct"])

    log(f"\n### Phase 1 — Top 5 configs by mean rel-L2\n", log_path)
    for rank, r in enumerate(phase1_results[:5], 1):
        c = r["cfg"]
        log(f"  #{rank}  config_{c['id']}  group={c['group']}  "
            f"lr={c['lr']:.0e}  bc_d={c['bc_d_weight']}  bc_n={c['bc_n_weight']}  "
            f"p={c['p_dim']}  nf={c['n_fourier']}  "
            f"mean={r['mean_rel_l2_pct']:.4f}%  max={r['max_rel_l2_pct']:.4f}%",
            log_path)

    # ----------------------------------------------------------------
    # Phase 2 — top 3 configs, full training
    # ----------------------------------------------------------------
    top3 = phase1_results[:3]

    log(f"\n## Phase 2  (Top-3 configs × {PHASE2_EPOCHS} epochs)\n", log_path)
    log(f"{'ID':>4}  {'Group':>12}  {'lr':>8}  {'bc_d':>6}  {'bc_n':>6}  "
        f"{'p':>5}  {'n_f':>4}  {'mean%':>8}  {'max%':>8}  {'time':>6}",
        log_path)
    log("-" * 90, log_path)

    phase2_results = []
    for r in top3:
        cfg = r["cfg"]
        # Use full batch for Phase 2
        cfg2 = {**cfg, "batch_size": None}
        out_dir = os.path.join(RESULTS_DIR, "phase2", f"config_{cfg['id']}")
        log(f"  Running config {cfg['id']} (phase2) ...", log_path)

        metrics = run_config(
            cfg2, PHASE2_EPOCHS, out_dir,
            x, y, v0_values, u_grids, f_raw,
            log_path, device, verbose_freq=500,
        )

        row = {**cfg2, "phase": 2, "epochs": PHASE2_EPOCHS, **metrics}
        row["branch_h"] = str(cfg2["branch_h"])
        row["trunk_h"]  = str(cfg2["trunk_h"])
        append_csv(csv_path, row, write_header=False)

        phase2_results.append({"cfg": cfg2, **metrics})

        log(f"{cfg['id']:>4}  {cfg['group']:>12}  {cfg['lr']:>8.0e}  "
            f"{cfg['bc_d_weight']:>6.1f}  {cfg['bc_n_weight']:>6.1f}  "
            f"{cfg['p_dim']:>5}  {cfg['n_fourier']:>4}  "
            f"{metrics['mean_rel_l2_pct']:>8.4f}  {metrics['max_rel_l2_pct']:>8.4f}  "
            f"{metrics['elapsed_min']:>5.1f}m",
            log_path)

    phase2_results.sort(key=lambda r: r["mean_rel_l2_pct"])
    best = phase2_results[0]
    bc = best["cfg"]

    log(f"\n## Final Best Config\n", log_path)
    log(f"  config_{bc['id']}  group={bc['group']}", log_path)
    log(f"  lr={bc['lr']}  bc_d_weight={bc['bc_d_weight']}  bc_n_weight={bc['bc_n_weight']}", log_path)
    log(f"  p_dim={bc['p_dim']}  trunk_h={bc['trunk_h']}  n_fourier={bc['n_fourier']}", log_path)
    log(f"  Mean rel-L2: {best['mean_rel_l2_pct']:.4f}%  "
        f"Max rel-L2: {best['max_rel_l2_pct']:.4f}%", log_path)
    log(f"\n  Best checkpoint: hparam_results/phase2/config_{bc['id']}/model_best.pth", log_path)
    log(f"\nSearch complete.  {time.strftime('%Y-%m-%d %H:%M:%S')}", log_path)

    print(f"\nLog saved to {log_path}")
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
