"""
plot_run4_all_v0.py
===================
Bar chart of rel-L2 error for all 41 v0 values (step 0.5, 0 to 20).
Train v0 shown in blue, test v0 in red.
Data parsed directly from run4/run.log — no model reload needed.
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

LOG_FILE = "output_v0_run4/run.log"
OUT_FILE = "output_v0_run4/all_v0_errors.png"

# ── parse log ────────────────────────────────────────────────────────────────
pattern = re.compile(r'^\s*([\d.]+)\s+([\d.]+)\s+(TRAIN|test)')
v0_list, err_list, split_list = [], [], []

in_eval = False
with open(LOG_FILE) as fh:
    for line in fh:
        if "Evaluation vs COMSOL" in line:
            in_eval = True
        if in_eval:
            m = pattern.match(line)
            if m:
                v0_list.append(float(m.group(1)))
                err_list.append(float(m.group(2)))
                split_list.append(m.group(3))

v0s    = np.array(v0_list)
errs   = np.array(err_list)
splits = np.array(split_list)

train_mask = splits == "TRAIN"
train_mean = errs[train_mask].mean()
test_mean  = errs[~train_mask].mean()
train_max  = errs[train_mask].max()
test_max   = errs[~train_mask].max()

# ── plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))

colors = ['#1f77b4' if t else '#d62728' for t in train_mask]
ax.bar(v0s, errs, color=colors, width=0.38, alpha=0.88)

# threshold lines
ax.axhline(1.0, color='green',  linestyle='--', lw=1.4, label='1% threshold')
ax.axhline(4.0, color='orange', linestyle='--', lw=1.2, alpha=0.7,
           label=f'~run4 mean ({errs.mean():.2f}%)')

# mean annotations
ax.axhline(train_mean, color='#1f77b4', linestyle=':', lw=1.2, alpha=0.6)
ax.axhline(test_mean,  color='#d62728', linestyle=':', lw=1.2, alpha=0.6)

ax.set_xlabel('v0  (Dirichlet BC at x = 0)', fontsize=12)
ax.set_ylabel('Rel-L2 error  (%)', fontsize=12)
ax.set_title(
    f'Run4 — PI-DeepONet rel-L2 vs COMSOL  |  All 41 v0 values\n'
    f'Train: mean={train_mean:.3f}%  max={train_max:.3f}%     '
    f'Test:  mean={test_mean:.3f}%  max={test_max:.3f}%',
    fontsize=12
)
ax.set_xticks(v0s[::2])
ax.set_xlim(-0.5, 20.5)
ax.set_ylim(0, max(errs) * 1.18)

legend_handles = [
    Patch(color='#1f77b4', label=f'Train  (n=11, mean={train_mean:.2f}%)'),
    Patch(color='#d62728', label=f'Test   (n=30, mean={test_mean:.2f}%)'),
    plt.Line2D([0], [0], color='green', linestyle='--', lw=1.4, label='1% target'),
]
ax.legend(handles=legend_handles, fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.35)

# label each bar with its value
for x, y, s in zip(v0s, errs, splits):
    ax.text(x, y + 0.04, f'{y:.2f}', ha='center', va='bottom',
            fontsize=5.5, color='#333333', rotation=90)

fig.tight_layout()
fig.savefig(OUT_FILE, dpi=160)
print(f"Saved -> {OUT_FILE}")
plt.close(fig)
