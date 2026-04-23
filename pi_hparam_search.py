"""
pi_hparam_search.py
===================
Hyperparameter search for pideeponet_2d_neumann.py.

Phase 1 : ~14 configs × 1000 epochs — identifies good region.
Phase 2 : Top-3 configs × 5000 epochs — confirms winner.

PI-specific axes searched:
  A) output_scale       : {5, 10, 20, 50}
  B) loss weights (w_d, w_n) : several ratios at best S
  C) learning rate      : {1e-3, 5e-4} on best config

Results: ./pi_hparam_results/log.md  +  results.csv
"""

import os
import json
import time
import csv
import numpy as np
import torch
import torch.nn as nn

from pideeponet_2d_neumann import (
    load_forcing, load_comsol_solutions, PIDeepONetTrainer
)
from networks import DeepONet2D

# ============================================================================
#  Search space
# ============================================================================

PHASE1_EPOCHS = 1000
PHASE2_EPOCHS = 5000
GRID_SIZE     = 31
F_SCALE       = 100.0
FORCING_FILE  = "Surface_Solution.txt"
DATA_FILE     = "data_v0.txt"
RESULTS_DIR   = "./pi_hparam_results"
SEED          = 42

# Fixed arch (best from previous Gaussian search)
BASE = dict(
    p_dim=512, branch_h=[256, 256], trunk_h=[512, 512, 512], n_fourier=8,
    w_res=1.0, warmup_epochs=500,
)

PHASE1_CONFIGS = []
_id = 1

# --- Group A: vary output_scale ---
for S in [5.0, 10.0, 20.0, 50.0]:
    PHASE1_CONFIGS.append({**BASE,
        "id": f"{_id:02d}", "group": "A_scale",
        "output_scale": S, "w_d": 100.0, "w_n": 10.0, "lr": 1e-3,
    }); _id += 1

# --- Group B: vary w_d / w_n (with S=10, fixed from Group A baseline) ---
for w_d, w_n in [(50.0, 5.0), (100.0, 10.0), (200.0, 20.0),
                  (100.0, 1.0), (100.0, 50.0)]:
    PHASE1_CONFIGS.append({**BASE,
        "id": f"{_id:02d}", "group": "B_weights",
        "output_scale": 10.0, "w_d": w_d, "w_n": w_n, "lr": 1e-3,
    }); _id += 1

# --- Group C: vary lr (S=10, best weights from B) ---
for lr in [5e-4, 3e-4]:
    PHASE1_CONFIGS.append({**BASE,
        "id": f"{_id:02d}", "group": "C_lr",
        "output_scale": 10.0, "w_d": 100.0, "w_n": 10.0, "lr": lr,
    }); _id += 1

# --- Group D: vary warmup ---
for wu in [0, 200]:
    PHASE1_CONFIGS.append({**BASE,
        "id": f"{_id:02d}", "group": "D_warmup",
        "output_scale": 10.0, "w_d": 100.0, "w_n": 10.0, "lr": 1e-3,
        "warmup_epochs": wu,
    }); _id += 1


# ============================================================================
#  Utilities
# ============================================================================

def log(msg, log_path):
    print(msg)
    with open(log_path, 'a') as f:
        f.write(msg + "\n")


def append_csv(path, row, write_header=False):
    mode = 'w' if write_header else 'a'
    with open(path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_model(cfg, branch_in_dim, device):
    return DeepONet2D(
        branch_in_dim=branch_in_dim,
        p=cfg["p_dim"],
        branch_hidden=tuple(cfg["branch_h"]),
        trunk_hidden=tuple(cfg["trunk_h"]),
        activation=nn.SiLU,
        use_fourier=True,
        n_fourier=cfg["n_fourier"],
    ).to(device)


def evaluate(trainer, u_grids):
    """Mean and max rel-L2 across all v0 samples using best checkpoint."""
    rel_errors = []
    for i, u_ref in enumerate(u_grids):
        u_pred = trainer.predict(i)
        rel = float(np.sqrt(np.sum((u_pred - u_ref) ** 2) /
                            (np.sum(u_ref ** 2) + 1e-12)))
        rel_errors.append(rel)
    return np.mean(rel_errors), np.max(rel_errors)


def run_config(cfg, epochs, out_dir, x, y, v0_values, u_grids, f_raw,
               log_path, device, verbose_freq=200):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), 'w') as f:
        json.dump({k: v for k, v in cfg.items()
                   if not isinstance(v, list)}, f, indent=2)

    np.random.seed(SEED); torch.manual_seed(SEED)

    model = build_model(cfg, GRID_SIZE * GRID_SIZE + 1, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    trainer = PIDeepONetTrainer(
        model=model, x=x, y=y,
        v0_values=v0_values, f_raw=f_raw,
        optimizer=optimizer, scheduler=scheduler,
        device=device, f_scale=F_SCALE,
        output_scale=cfg["output_scale"],
        w_res=cfg["w_res"], w_d=cfg["w_d"], w_n=cfg["w_n"],
        warmup_epochs=cfg["warmup_epochs"],
    )

    t0 = time.time()
    trainer.run(epochs=epochs, verbose_freq=verbose_freq, log_dir=out_dir)
    elapsed = time.time() - t0

    best_ckpt = os.path.join(out_dir, "model_best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device,
                                         weights_only=True))

    mean_rel, max_rel = evaluate(trainer, u_grids)
    return {
        "mean_rel_l2_pct": round(mean_rel * 100, 4),
        "max_rel_l2_pct":  round(max_rel  * 100, 4),
        "elapsed_min":     round(elapsed / 60, 2),
    }


# ============================================================================
#  Main
# ============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, "log.md")
    csv_path = os.path.join(RESULTS_DIR, "results.csv")

    with open(log_path, 'w') as f:
        f.write("# PI-DeepONet Hyperparameter Search\n\n")
        f.write(f"Device: {device}  |  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    log("Loading data ...", log_path)
    x, y, f_comsol = load_forcing(FORCING_FILE, N=GRID_SIZE)
    f_raw = f_comsol / F_SCALE
    v0_values, u_grids = load_comsol_solutions(DATA_FILE, x, y)

    # ----------------------------------------------------------------
    # Phase 1
    # ----------------------------------------------------------------
    log(f"\n## Phase 1  ({len(PHASE1_CONFIGS)} configs × {PHASE1_EPOCHS} epochs)\n",
        log_path)
    header = (f"{'ID':>4}  {'Group':>10}  {'S':>5}  {'w_d':>6}  {'w_n':>5}  "
              f"{'lr':>7}  {'wu':>4}  {'mean%':>8}  {'max%':>8}  {'time':>6}")
    log(header, log_path)
    log("-" * len(header), log_path)

    phase1_results = []
    csv_header_written = False

    for cfg in PHASE1_CONFIGS:
        out_dir = os.path.join(RESULTS_DIR, "phase1", f"config_{cfg['id']}")
        log(f"  Running config {cfg['id']} ({cfg['group']}) ...", log_path)

        metrics = run_config(cfg, PHASE1_EPOCHS, out_dir,
                             x, y, v0_values, u_grids, f_raw,
                             log_path, device, verbose_freq=200)

        row = {**{k: v for k, v in cfg.items() if not isinstance(v, list)},
               "phase": 1, "epochs": PHASE1_EPOCHS, **metrics}
        append_csv(csv_path, row, write_header=not csv_header_written)
        csv_header_written = True
        phase1_results.append({"cfg": cfg, **metrics})

        log(f"{cfg['id']:>4}  {cfg['group']:>10}  {cfg['output_scale']:>5.0f}  "
            f"{cfg['w_d']:>6.0f}  {cfg['w_n']:>5.0f}  {cfg['lr']:>7.0e}  "
            f"{cfg['warmup_epochs']:>4}  "
            f"{metrics['mean_rel_l2_pct']:>8.4f}  {metrics['max_rel_l2_pct']:>8.4f}  "
            f"{metrics['elapsed_min']:>5.1f}m", log_path)

    phase1_results.sort(key=lambda r: r["mean_rel_l2_pct"])
    log(f"\n### Phase 1 — Top 5\n", log_path)
    for rank, r in enumerate(phase1_results[:5], 1):
        c = r["cfg"]
        log(f"  #{rank}  config_{c['id']}  {c['group']}  S={c['output_scale']}  "
            f"w_d={c['w_d']}  w_n={c['w_n']}  lr={c['lr']:.0e}  "
            f"wu={c['warmup_epochs']}  "
            f"mean={r['mean_rel_l2_pct']:.4f}%  max={r['max_rel_l2_pct']:.4f}%",
            log_path)

    # ----------------------------------------------------------------
    # Phase 2 — top 3
    # ----------------------------------------------------------------
    top3 = phase1_results[:3]
    log(f"\n## Phase 2  (Top-3 × {PHASE2_EPOCHS} epochs)\n", log_path)
    log(header, log_path)
    log("-" * len(header), log_path)

    phase2_results = []
    for r in top3:
        cfg = r["cfg"]
        out_dir = os.path.join(RESULTS_DIR, "phase2", f"config_{cfg['id']}")
        log(f"  Running config {cfg['id']} (phase2) ...", log_path)

        metrics = run_config(cfg, PHASE2_EPOCHS, out_dir,
                             x, y, v0_values, u_grids, f_raw,
                             log_path, device, verbose_freq=500)

        row = {**{k: v for k, v in cfg.items() if not isinstance(v, list)},
               "phase": 2, "epochs": PHASE2_EPOCHS, **metrics}
        append_csv(csv_path, row, write_header=False)
        phase2_results.append({"cfg": cfg, **metrics})

        log(f"{cfg['id']:>4}  {cfg['group']:>10}  {cfg['output_scale']:>5.0f}  "
            f"{cfg['w_d']:>6.0f}  {cfg['w_n']:>5.0f}  {cfg['lr']:>7.0e}  "
            f"{cfg['warmup_epochs']:>4}  "
            f"{metrics['mean_rel_l2_pct']:>8.4f}  {metrics['max_rel_l2_pct']:>8.4f}  "
            f"{metrics['elapsed_min']:>5.1f}m", log_path)

    phase2_results.sort(key=lambda r: r["mean_rel_l2_pct"])
    best = phase2_results[0]["cfg"]

    log(f"\n## Best Config\n", log_path)
    log(f"  config_{best['id']}  group={best['group']}", log_path)
    log(f"  output_scale={best['output_scale']}  w_res={best['w_res']}  "
        f"w_d={best['w_d']}  w_n={best['w_n']}", log_path)
    log(f"  lr={best['lr']}  warmup_epochs={best['warmup_epochs']}", log_path)
    log(f"  p_dim={best['p_dim']}  n_fourier={best['n_fourier']}", log_path)
    log(f"  Mean rel-L2: {phase2_results[0]['mean_rel_l2_pct']:.4f}%  "
        f"Max rel-L2: {phase2_results[0]['max_rel_l2_pct']:.4f}%", log_path)
    log(f"\n  Checkpoint: pi_hparam_results/phase2/config_{best['id']}/model_best.pth",
        log_path)
    log(f"\nSearch complete. {time.strftime('%Y-%m-%d %H:%M:%S')}", log_path)


if __name__ == "__main__":
    main()
