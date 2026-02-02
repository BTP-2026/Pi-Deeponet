# train_deeponet_poisson_from_solver_fixed.py
# Full script: generate dataset from compute_numerical_solution on x with 1000 pts,
# f = 1, random Dirichlet BCs per sample, train DeepONet with losses data + boundary + residual.
#
# Key fixes:
#  - residual uses f_scale=100.0 (matches solver)
#  - residual term multiplied by res_weight
#  - uL and uR are concatenated to branch input (branch input dim = nx + 2)
#
# WARNING: nx=1000 + autograd second derivatives per-sample is computationally heavy.

import os
import time
import numpy as np
import torch
import torch.nn as nn

def gaussian_forcing(x, mu, sigma, amplitude=1.0):
    """
    Gaussian forcing function:
    f(x) = A * exp(-(x-mu)^2 / (2*sigma^2))
    """
    return amplitude * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))

# -----------------------
# Numerical solver (as provided; fixed h)
# -----------------------
def compute_numerical_solution(x, f, N, u_left=0.0, u_right=0.0):
    """
    Solve -u'' = f on uniform grid x of length N with Dirichlet u(0)=u_left, u(1)=u_right
    using finite difference: central second difference matrix for interior points.
    Returns u vector length N.
    """
    # grid spacing
    h = x[1] - x[0]
    # interior size
    n_in = N - 2
    # discrete Laplacian matrix K (n_in x n_in)
    K = 2.0 * np.eye(n_in) - np.eye(n_in, k=1) - np.eye(n_in, k=-1)
    # right-hand side scaled as given in your snippet (kept the factor 100)
    b = (h ** 2) * 100.0 * f[1:-1].astype(np.float64)
    # adjust for Dirichlet BCs: u_0 and u_{N-1}
    b[0] += u_left
    b[-1] += u_right
    u_interior = np.linalg.solve(K, b)
    return np.concatenate(([u_left], u_interior, [u_right]))

# -----------------------
# DeepONet model
# -----------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256), activation=nn.Tanh):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    def __init__(self, branch_in_dim, p=256, branch_hidden=(256,256), trunk_hidden=(256,256)):
        """
        branch_in_dim: dimension of branch input (nx + 2 if concatenating uL/uR)
        p: inner-product feature dimension
        """
        super().__init__()
        self.branch = MLP(in_dim=branch_in_dim, out_dim=p, hidden=branch_hidden)
        self.trunk = MLP(in_dim=1, out_dim=p, hidden=trunk_hidden)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input, x_grid):
        # branch_input: (B, branch_in_dim), x_grid: (nx,1)
        b = self.branch(branch_input)                # (B,p)
        t = self.trunk(x_grid)                       # (nx,p)
        out = torch.sum(b.unsqueeze(1) * t.unsqueeze(0), dim=-1)  # (B,nx)
        return out + self.bias

# -----------------------
# Trainer (simple, using N-sample splitting per epoch)
# -----------------------
class PdeModelTorch:
    def __init__(self, model, x, f, u, uL, uR, optimizer,
                 loss_fn=nn.MSELoss(), batches=10, device=None,
                 f_scale=100.0, res_weight=10.0):
        """
        model: DeepONet instance (expects branch input dims matching how we construct branch vectors)
        x: (nx,)
        f: (N, nx)
        u: (N, nx)
        uL: (N,)
        uR: (N,)
        f_scale: float, scaler used in numerical solver (100 in your compute_numerical_solution)
        res_weight: float, multiply residual loss by this weight when summing losses
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.x = x.astype(np.float32)
        self.f = f.astype(np.float32)
        self.u = u.astype(np.float32)
        self.uL = uL.astype(np.float32)
        self.uR = uR.astype(np.float32)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batches = int(batches)
        self.N, self.nx = self.f.shape
        # trunk receives x grid as (nx,1) tensor
        self.x_grid = torch.tensor(self.x.reshape(-1,1), dtype=torch.float32, device=self.device)
        self.f_scale = float(f_scale)
        self.res_weight = float(res_weight)

    def _get_epoch_splits(self):
        idx = np.arange(self.N)
        np.random.shuffle(idx)
        return np.array_split(idx, self.batches)

    def _u_xx_for_sample(self, branch_input_single):
        """
        Compute d2u/dx2 for one sample using autograd.
        branch_input_single: (branch_in_dim,) tensor (on device)
        """
        # ensure batch dim
        if branch_input_single.dim() == 1:
            b_in = branch_input_single.unsqueeze(0)
        else:
            b_in = branch_input_single
        # require grad on x grid for derivatives
        x_req = self.x_grid.clone().detach().requires_grad_(True)
        u_pred = self.model(b_in.to(self.device), x_req)  # (1, nx)
        u_vec = u_pred.view(-1)                           # (nx,)
        # first derivative du/dx
        du_dx = torch.autograd.grad(u_vec, x_req, grad_outputs=torch.ones_like(u_vec), create_graph=True)[0].view(-1)
        # second derivative
        d2u_dx2 = torch.autograd.grad(du_dx, x_req, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0].view(-1)
        return d2u_dx2

    def _make_branch_input(self, f_batch, uL_batch, uR_batch):
        """
        Concatenate uL and uR to branch input.
        f_batch: (B, nx)
        uL_batch, uR_batch: (B,) or (B,1)
        returns: branch_input (B, nx+2)
        """
        B = f_batch.shape[0]
        uL_col = uL_batch.view(B,1)
        uR_col = uR_batch.view(B,1)
        branch = torch.cat([f_batch, uL_col, uR_col], dim=1)   # (B, nx+2)
        return branch

    def train_step_indices(self, indices):
        """
        One optimizer step on subset of samples indexed by indices.
        """
        self.model.train()
        idx = np.array(indices, dtype=int)
        f_batch = torch.tensor(self.f[idx], dtype=torch.float32, device=self.device)   # (B,nx)
        u_batch = torch.tensor(self.u[idx], dtype=torch.float32, device=self.device)   # (B,nx)
        uL_batch = torch.tensor(self.uL[idx], dtype=torch.float32, device=self.device) # (B,)
        uR_batch = torch.tensor(self.uR[idx], dtype=torch.float32, device=self.device) # (B,)

        # make branch input (concatenate uL/uR)
        branch_in = self._make_branch_input(f_batch, uL_batch, uR_batch)   # (B, nx+2)

        # forward (data + boundary)
        u_pred = self.model(branch_in, self.x_grid.detach())   # (B,nx)
        data_loss = self.loss_fn(u_pred, u_batch)

        left_pred = u_pred[:,0]; right_pred = u_pred[:,-1]
        bound_loss = self.loss_fn(left_pred, uL_batch) + self.loss_fn(right_pred, uR_batch)

        # residual per sample (expensive): compute -u_xx - f_scale * f
        res_list = []
        for i in range(branch_in.shape[0]):
            # branch_in[i] contains f_i concat uL_i, uR_i
            u_xx = self._u_xx_for_sample(branch_in[i])   # (nx,)
            # f_i is the first nx entries of branch_in
            f_i = branch_in[i, :self.nx].view(-1)
            ge = -u_xx - (self.f_scale * f_i)
            res_list.append(torch.mean(ge**2))
        residual_loss = torch.stack(res_list).mean()

        loss = data_loss + bound_loss + self.res_weight * residual_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "data_loss": data_loss.item(),
                "bound_loss": bound_loss.item(), "residual_loss": residual_loss.item()}

    def run(self, epochs=200, verbose_freq=10, log_dir="./output", save_every=0):
        os.makedirs(log_dir, exist_ok=True)
        history = {"loss": [], "data_loss": [], "bound_loss": [], "residual_loss": []}
        start = time.time()
        for ep in range(1, epochs+1):
            splits = self._get_epoch_splits()
            ep_stats = {"loss":0.0,"data_loss":0.0,"bound_loss":0.0,"residual_loss":0.0}
            steps = 0
            for inds in splits:
                if len(inds) == 0:
                    continue
                stats = self.train_step_indices(inds)
                for k in ep_stats:
                    ep_stats[k] += stats[k]
                steps += 1
            # average per step
            for k in ep_stats:
                ep_stats[k] /= max(1, steps)
                history[k].append(ep_stats[k])
            if ep % verbose_freq == 0 or ep == 1:
                elapsed = (time.time() - start) / 60.0
                print(f"Epoch {ep}/{epochs} loss={ep_stats['loss']:.4e} data={ep_stats['data_loss']:.4e} bound={ep_stats['bound_loss']:.4e} res={ep_stats['residual_loss']:.4e} time={elapsed:.2f}min")
            if save_every and (ep % save_every == 0):
                torch.save(self.model.state_dict(), os.path.join(log_dir, f"model_ep{ep}.pth"))
        torch.save(self.model.state_dict(), os.path.join(log_dir, "model_final.pth"))
        return history

    def predict(self, f_array, uL_array, uR_array, batch_size=4):
        """
        Predict solutions for many samples. Must provide matching uL,uR arrays.
        f_array: (M, nx)
        uL_array, uR_array: (M,)
        """
        self.model.eval()
        out = []
        f_array = f_array.astype(np.float32)
        uL_array = uL_array.astype(np.float32)
        uR_array = uR_array.astype(np.float32)
        M = f_array.shape[0]
        for i in range(0, M, batch_size):
            fb = torch.tensor(f_array[i:i+batch_size], dtype=torch.float32, device=self.device)
            uLb = torch.tensor(uL_array[i:i+batch_size], dtype=torch.float32, device=self.device)
            uRb = torch.tensor(uR_array[i:i+batch_size], dtype=torch.float32, device=self.device)
            branch_in = self._make_branch_input(fb, uLb, uRb)
            with torch.no_grad():
                up = self.model(branch_in, self.x_grid.detach())
            out.append(up.cpu().numpy())
        return np.vstack(out)

# -----------------------
# Dataset generation using provided solver
# -----------------------
if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # grid
    nx = 1000
    x = np.linspace(0.0, 1.0, nx).astype(np.float64)

    # dataset size
    N_samples = 100   # change as needed

    # forcing f(x) = 1 for all samples (shape N x nx)
    #f_all = np.ones((N_samples, nx), dtype=np.float64)

    mu_all = np.random.uniform(0.2, 0.8, size=N_samples)
    sigma_all = np.random.uniform(0.03, 0.12, size=N_samples)
    amp_all = np.random.uniform(0.5, 1.5, size=N_samples)

    # build forcing dataset
    f_all = np.zeros((N_samples, nx), dtype=np.float64)
    for i in range(N_samples):
        f_all[i, :] = gaussian_forcing(
            x,
            mu=mu_all[i],
            sigma=sigma_all[i],
            amplitude=amp_all[i]
        )

    f_all /= np.max(np.abs(f_all), axis=1, keepdims=True)

    # random boundary values per sample (you can set to zeros if you want)
    uL = np.random.uniform(-1.0, 1.0, size=(N_samples,)).astype(np.float64)
    uR = np.random.uniform(-1.0, 1.0, size=(N_samples,)).astype(np.float64)

    # compute numerical solutions using your solver
    u_all = np.zeros_like(f_all, dtype=np.float64)
    print("Generating dataset with numerical solver (this may take a moment)...")
    for i in range(N_samples):
        u_all[i,:] = compute_numerical_solution(x, f_all[i,:], N=nx, u_left=float(uL[i]), u_right=float(uR[i]))
    print("Dataset generation complete.")

    # convert to float32 for training
    N_total = f_all.shape[0]
    train_ratio = 0.75
    N_train = int(train_ratio * N_total)

    perm = np.random.permutation(N_total)
    train_idx = perm[:N_train]
    test_idx  = perm[N_train:]

    # Train arrays
    f_train = f_all[train_idx].astype(np.float32)
    u_train = u_all[train_idx].astype(np.float32)
    uL_train = uL[train_idx].astype(np.float32)
    uR_train = uR[train_idx].astype(np.float32)

    # Test arrays
    f_test = f_all[test_idx].astype(np.float32)
    u_test = u_all[test_idx].astype(np.float32)
    uL_test = uL[test_idx].astype(np.float32)
    uR_test = uR[test_idx].astype(np.float32)

    print(f"Dataset: total={N_total}, train={N_train}, test={N_total-N_train}")

    # model: branch_in_dim = nx + 2 (f + uL + uR)
    branch_in_dim = nx + 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = DeepONet(branch_in_dim=branch_in_dim, p=256, branch_hidden=(256,256), trunk_hidden=(256,256)).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # trainer with correct f_scale and a tuned res_weight (adjust res_weight if needed)
    trainer = PdeModelTorch(model=model, x=x.astype(np.float32), f=f_train, u=u_train, uL=uL_train, uR=uR_train,
                            optimizer=optimizer, loss_fn=nn.MSELoss(), batches=10, device=device,
                            f_scale=100.0, res_weight=10.0)

    # train (be prepared for long runtime with nx=1000)
    history = trainer.run(epochs=200, verbose_freq=10, log_dir="./output_deeponet_poisson_fixed", save_every=0)

    # example prediction for first 3 samples (pass uL/uR explicitly)
    """ preds = trainer.predict(f_train[:3], uL_train[:3], uR_train[:3], batch_size=1)
    try:
        import matplotlib.pyplot as plt
        for i in range(min(3, preds.shape[0])):
            plt.figure(figsize=(8,3))
            plt.plot(x, u_train[i], label="true")
            plt.plot(x, preds[i], '--', label="pred")
            plt.title(f"Sample {i}, uL={uL_train[i]:.3f}, uR={uR_train[i]:.3f}")
            plt.legend()
        plt.show()
    except Exception:
        print("matplotlib not available or plotting failed; predictions computed.") """
    
    # Predictions: pass matching BC arrays
    preds_train = trainer.predict(f_train, uL_train, uR_train, batch_size=4)  # (N_train, nx)
    preds_test  = trainer.predict(f_test,  uL_test,  uR_test,  batch_size=4)  # (N_test, nx)

    # per-sample MSEs
    mse_train = np.mean((preds_train - u_train)**2, axis=1)
    mse_test  = np.mean((preds_test  - u_test )**2, axis=1)
    print(f"Train MSE mean: {mse_train.mean():.6e}, Test MSE mean: {mse_test.mean():.6e}")

    # plotting 2x3 layout: top row -> first 3 train samples, bottom row -> first 2 test samples and empty
    """ import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14,7))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # top: 3 train examples (indices 0,1,2 in train arrays)
    for i in range(3):
        ax = axes[0, i]
        ax.plot(x, u_train[i], label='True', color='tab:blue')
        ax.plot(x, preds_train[i], '--', label='PI-Deeponet', color='tab:orange')
        ax.scatter([0.0, 1.0], [uL_train[i], uR_train[i]], color='k', s=60, zorder=10, label='BCs' if i==0 else '')
        ax.set_title(f"Train Sample {i}\nMSE: {mse_train[i]:.6f}\nBC: ({uL_train[i]:.2f}, {uR_train[i]:.2f})")
        ax.set_xlabel('x'); ax.set_ylabel('u(x)'); ax.grid(True)
        if i==0: ax.legend(loc='upper right')

    # bottom: 2 test examples (indices 0,1 in test arrays)
    for j in range(2):
        ax = axes[1, j]
        ax.plot(x, u_test[j], label='True', color='tab:blue')
        ax.plot(x, preds_test[j], '--', label='PI-Deeponet', color='tab:orange')
        ax.scatter([0.0, 1.0], [uL_test[j], uR_test[j]], color='k', s=60, zorder=10)
        ax.set_title(f"Test Sample {j}\nMSE: {mse_test[j]:.6f}\nBC: ({uL_test[j]:.2f}, {uR_test[j]:.2f})")
        ax.set_xlabel('x'); ax.set_ylabel('u(x)'); ax.grid(True)
        if j==0: ax.legend(loc='upper right')

    # last subplot blank
    axes[1,2].axis('off')

    plt.suptitle("Seen BCs (Training) and Unseen BCs (Testing)", fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.96])
    out_fig = "/mnt/data/deeponet_poisson_train_test.png"
    plt.savefig(out_fig, dpi=200)
    plt.close(fig)
    print("Saved comparison figure to:", out_fig) """

    import matplotlib.pyplot as plt

    def plot_sample(x, f, u_true, u_pred, uL, uR, title=""):
        fig, ax1 = plt.subplots(figsize=(9, 4))

        # ---- Solution plot ----
        ax1.plot(x, u_true, label="True u(x)", color="tab:blue", linewidth=2)
        ax1.plot(x, u_pred, "--", label="Predicted u(x)", color="tab:orange", linewidth=2)
        ax1.scatter([0.0, 1.0], [uL, uR], color="k", s=60, zorder=10, label="BCs")
        ax1.set_xlabel("x")
        ax1.set_ylabel("u(x)")
        ax1.grid(True)

        # ---- Forcing (secondary axis) ----
        ax2 = ax1.twinx()
        ax2.plot(x, f, color="tab:red", alpha=0.35, linewidth=2, label="Forcing f(x)")
        ax2.set_ylabel("f(x)")

        # ---- Combined legend ----
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.title(title)
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------
    # Plot ONE training sample
    # -------------------------------------------------
    i = 0
    plot_sample(
        x,
        f_train[i],
        u_train[i],
        preds_train[i],
        uL_train[i],
        uR_train[i],
        title=f"TRAIN sample {i} | MSE={mse_train[i]:.2e}"
    )

    # -------------------------------------------------
    # Plot ONE test (unseen) sample
    # -------------------------------------------------
    j = 0
    plot_sample(
        x,
        f_test[j],
        u_test[j],
        preds_test[j],
        uL_test[j],
        uR_test[j],
        title=f"TEST sample {j} (unseen) | MSE={mse_test[j]:.2e}"
    )

    # -------------------------------------------------
    # Grid plot: multiple TEST samples
    # -------------------------------------------------
    n_show = 3
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 3 * n_show))

    for i in range(n_show):
        ax = axes[i]

        # solution
        ax.plot(x, u_test[i], label="True u(x)", color="tab:blue", linewidth=2)
        ax.plot(x, preds_test[i], "--", label="Pred u(x)", color="tab:orange", linewidth=2)
        ax.scatter([0, 1], [uL_test[i], uR_test[i]], color="k", s=40)

        # forcing
        ax2 = ax.twinx()
        ax2.plot(x, f_test[i], color="tab:red", alpha=0.3, linewidth=2)

        ax.set_title(f"Test sample {i} | MSE={mse_test[i]:.2e}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.grid(True)

        if i == 0:
            ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()