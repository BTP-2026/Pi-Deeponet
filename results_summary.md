# PI-DeepONet Training Results Summary

All runs use `pideeponet_2d_dirichlet_v0.py`.  
PDE: -∇²u = f(x,y), BCs: u=v0 @ x=0, u=0 @ x=1, du/dy=0 @ y walls.  
Data: 4300 raw COMSOL mesh points, no interpolation.

---

## Architecture Glossary

| Symbol | Meaning |
|---|---|
| p | Branch/trunk coupling dimension |
| trunk_h | Hidden layer sizes in trunk MLP |
| n_fourier | Fourier frequencies for (x,y) trunk encoding; trunk_in = 2+4*n_fourier |
| n_v0_freqs | Fourier frequencies for v0 branch encoding; branch_in = 1+2*n_v0_freqs |
| train_step | Step size between training v0 values; 0.5 = all 41 samples |
| w_d | Dirichlet BC loss weight (x=0 and x=1) |
| w_neu | Neumann BC loss weight (y walls) |
| S | Output scale (prediction = S * u_net) |

---

## Run Results

| Run | Epochs | p | trunk_h | n_fourier | branch_in | n_v0_freqs | train_step | train_v0 | test_v0 | w_d | w_neu | Mean rel-L2 (train) | Mean rel-L2 (test) | Max rel-L2 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run1 | 5000 | 128 | [128,128] | 8 | 1 | 0 (raw) | 0.5 | 41 | — | 100 | 10 | 1.2611% | — | 3.7117% (v0=0) | Baseline; LR decayed to 1e-6 at end |
| run2 | 10000 | 128 | [128,128] | 8 | 1 | 0 (raw) | 0.5 | 41 | — | 500 | 10 | 0.2352% | — | 0.5994% (v0=0.5) | Resume run1; higher w_d; best result so far |
| run3 | 10000 | 256 | [256,256,256] | 16 | 9 | 4 | 2.0 | 11 | 30 | 1 | 1 | 33.33% | 33.23% | 40.29% (v0=20) | Equal weights caused Dirichlet BC failure; d0/d1 loss ~36 at end |
| run4 | 10000 | 256 | [256,256,256] | 16 | 9 | 4 | 2.0 | 11 | 30 | 500 | 10 | 4.0287% | 3.9965% | 4.4688% (v0=3.0) | run3 arch + proper loss weights; train≈test → good generalization but worse abs accuracy vs run2 |

---

## Run Configurations (CLI commands)

### run1
```bash
python3 pideeponet_2d_dirichlet_v0.py \
  --epochs 5000 --p_dim 128 --branch_h 64 64 --trunk_h 128 128 \
  --n_fourier 8 --n_v0_freqs 0 --output_scale 20.0 \
  --w_d 100 --w_neu 10 --warmup_epochs 500 --train_step 0.5 \
  --lr 1e-3 --log_dir output_v0_run1
```

### run2
```bash
python3 pideeponet_2d_dirichlet_v0.py \
  --epochs 10000 --p_dim 128 --branch_h 64 64 --trunk_h 128 128 \
  --n_fourier 8 --n_v0_freqs 0 --output_scale 20.0 \
  --w_d 500 --w_neu 10 --warmup_epochs 0 --train_step 0.5 \
  --lr 1e-3 --resume output_v0_run1/model_best.pth --log_dir output_v0_run2
```

### run3
```bash
python3 pideeponet_2d_dirichlet_v0.py \
  --epochs 10000 --p_dim 256 --branch_h 64 64 --trunk_h 256 256 256 \
  --n_fourier 16 --n_v0_freqs 4 --output_scale 20.0 \
  --w_d 1 --w_neu 1 --warmup_epochs 500 --train_step 2.0 \
  --lr 1e-3 --log_dir output_v0_run3
```

### run4
```bash
python3 pideeponet_2d_dirichlet_v0.py \
  --epochs 10000 --p_dim 256 --branch_h 64 64 --trunk_h 256 256 256 \
  --n_fourier 16 --n_v0_freqs 4 --output_scale 20.0 \
  --w_d 500 --w_neu 10 --warmup_epochs 500 --train_step 2.0 \
  --lr 1e-3 --log_dir output_v0_run4
```

---

## Key Lessons

1. **Loss weights are essential**: w_d=1 caused Dirichlet loss to dominate d0/d1 ~36 at end of run3 → 33% rel-L2. w_d=500 enforces BCs properly.
2. **LR restart helps**: run2 resumed run1's checkpoint with fresh LR cycle → 5.4× improvement (1.26% → 0.235%).
3. **Removing f from branch**: f is fixed across all samples; branch should encode only v0.
4. **No interpolation**: raw 4300 COMSOL mesh points used directly.
5. **v0=0 is hardest**: zero Dirichlet everywhere + pure f-driven solution → historically worst error.
