---
name: Training and Architecture Feedback
description: Lessons learned from training runs — what works and what to avoid
type: feedback
---

Do NOT use CosineAnnealingWarmRestarts for single-sample DeepONet problems.

**Why:** Warm restarts spike LR at each cycle boundary, ejecting the model from good minima it has already found. On this problem, the model hit data_loss=1.2e-2 at epoch 1250, then degraded to 8.8e-2 after the first restart at epoch 5000. Use `CosineAnnealingLR(T_max=epochs)` instead for a single smooth decay.

**How to apply:** Any time training a DeepONet on a fixed single-sample problem (branch input fixed, trunk learning u(x,y)), use CosineAnnealingLR, not WarmRestarts.

---

Always use Fourier Feature Encoding on trunk (x,y) inputs for problems with sharp spatial features.

**Why:** Raw (x,y) → MLP trunk suffers from spectral bias and cannot efficiently resolve sharp Gaussian peaks. Encoding to [x, y, sin(2πkx), cos(2πkx), sin(2πky), cos(2πky)] for k=1..8 (34-dim total) gave 17× accuracy improvement with half the epochs.

**How to apply:** `use_fourier=True, n_fourier=8` in DeepONet2D constructor.

---

Always normalize the target when the solution has a sharp localized peak surrounded by near-zero values.

**Why:** MSE over 961 grid points was dominated by ~900 near-zero boundary points, suppressing the Gaussian peak (~25 points). Normalizing to [0,1] gives the peak proper gradient weight.

**How to apply:** `u_scale = u_true.max()`, train on normalized target, `trainer.predict()` unscales automatically.

---

Always save the best checkpoint (by data_loss), not just the final model.

**Why:** With cosine annealing ending at eta_min=1e-6, the final model IS the best. But with any schedule that doesn't strictly decrease, the final checkpoint may not be optimal.

**How to apply:** Track `best_data_loss` each epoch, save `model_best.pth`. Load it for evaluation.

---

User does not want to vary the Gaussian centre position in the Gaussian-forcing Poisson problem.

**Why:** The problem is a fixed forcing function. The PINTO reference image showing multiple panels is NOT showing different Gaussian centres — it shows the case with non-zero Neumann BC values, which is not needed.

**How to apply:** Keep Gaussian centre fixed, Neumann BC value = 0. Generate 3-panel plot: Reference | DeepONet | Absolute Error.
