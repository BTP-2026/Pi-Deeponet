import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py


# =============================================================================
# Differentiation Matrix Operators (from Poisson-solver-2D)
# =============================================================================

def Diff_mat_1D(Nx):
    # First derivative: central difference with forward/backward at boundaries
    D_1d = sp.diags([-1, 1], [-1, 1], shape=(Nx, Nx))  # Division by (2*dx) required later
    D_1d = sp.lil_matrix(D_1d)
    D_1d[0, [0, 1, 2]] = [-3, 4, -1]                    # 2nd order forward difference
    D_1d[Nx-1, [Nx-3, Nx-2, Nx-1]] = [1, -4, 3]        # 2nd order backward difference

    # Second derivative: central difference with proper boundary treatment
    D2_1d = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx))  # Division by dx^2 required
    D2_1d = sp.lil_matrix(D2_1d)
    D2_1d[0, [0, 1, 2, 3]] = [2, -5, 4, -1]                   # 2nd order forward
    D2_1d[Nx-1, [Nx-4, Nx-3, Nx-2, Nx-1]] = [-1, 4, -5, 2]   # 2nd order backward

    return D_1d.tocsr(), D2_1d.tocsr()


def Diff_mat_2D(Nx, Ny):
    Dx_1d, D2x_1d = Diff_mat_1D(Nx)
    Dy_1d, D2y_1d = Diff_mat_1D(Ny)

    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)

    # 2D operators via Kronecker product
    Dx_2d = sp.kron(Iy, Dx_1d)
    Dy_2d = sp.kron(Dy_1d, Ix)
    D2x_2d = sp.kron(Iy, D2x_1d)
    D2y_2d = sp.kron(D2y_1d, Ix)

    return Dx_2d.tocsr(), Dy_2d.tocsr(), D2x_2d.tocsr(), D2y_2d.tocsr()


# =============================================================================
# Boundary Index Functions
# =============================================================================

def get_boundary_indices(Nx, Ny):
    indices = {
        'left': [],      # x = 0 (i = 0)
        'right': [],     # x = 1 (i = Nx-1)
        'bottom': [],    # y = 0 (j = 0)
        'top': []        # y = 1 (j = Ny-1)
    }

    for j in range(Ny):
        indices['left'].append(j * Nx + 0)           # i = 0
        indices['right'].append(j * Nx + (Nx - 1))   # i = Nx-1

    for i in range(Nx):
        indices['bottom'].append(0 * Nx + i)         # j = 0
        indices['top'].append((Ny - 1) * Nx + i)     # j = Ny-1

    # Convert to numpy arrays
    for key in indices:
        indices[key] = np.array(indices[key], dtype=int)

    # All boundary indices combined
    all_boundary = np.unique(np.concatenate([indices[k] for k in indices]))

    return indices, all_boundary


# =============================================================================
# 2D Poisson Solver
# =============================================================================

def solve_poisson_2d(x, y, f, bc_values):
    Nx = len(x)
    Ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Get differentiation matrices
    Dx_2d, Dy_2d, D2x_2d, D2y_2d = Diff_mat_2D(Nx, Ny)

    # Build negative Laplacian: -∇² = -(d²/dx² + d²/dy²)
    # We solve: -∇²u = f, which gives: -L * u = f, or L * u = -f
    # Equivalently, use -L as the system matrix
    L_sys = -(D2x_2d / dx**2 + D2y_2d / dy**2)

    # Convert to lil_matrix for efficient row modification
    L_sys = sp.lil_matrix(L_sys)

    # Build RHS vector
    if np.isscalar(f):
        b = f * np.ones(Nx * Ny)
    else:
        b = f.ravel().copy()

    # Apply source scaling (factor of 100 like in 1D case)
    b = 100.0 * b

    # Get boundary indices
    bc_indices, all_boundary = get_boundary_indices(Nx, Ny)

    # Identity matrix for Dirichlet BC
    I_sp = sp.eye(Nx * Ny).tocsr()

    # Process boundary values - convert scalars to arrays
    bc_arrays = {}
    for side, val in bc_values.items():
        if np.isscalar(val):
            if side in ['left', 'right']:
                bc_arrays[side] = val * np.ones(Ny)
            else:
                bc_arrays[side] = val * np.ones(Nx)
        else:
            bc_arrays[side] = np.asarray(val)

    # Apply Dirichlet boundary conditions
    # Replace rows in L_sys with identity rows, set b to boundary values
    for side in ['left', 'right', 'bottom', 'top']:
        idx = bc_indices[side]
        L_sys[idx, :] = I_sp[idx, :]  # Replace with identity rows
        b[idx] = bc_arrays[side]       # Set RHS to boundary values

    # Convert back to CSR for efficient solving
    L_sys = L_sys.tocsr()

    # Solve the system
    u = spsolve(L_sys, b)

    # Reshape to 2D grid
    u = u.reshape(Ny, Nx)

    return u


# =============================================================================
# Visualization
# =============================================================================

def plot_3d_solution(x, y, u, title="2D Poisson Solution", save_path=None):
    """Create a 3D surface plot and contour plot of the solution."""
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(14, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X, Y, u, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    ax1.set_title('3D Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

    # 2D contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(X, Y, u, levels=50, cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour Plot')
    ax2.set_aspect('equal')
    fig.colorbar(contour, ax=ax2)

    # Cross-section at y=0.5
    ax3 = fig.add_subplot(133)
    mid_y = len(y) // 2
    ax3.plot(x, u[mid_y, :], 'b-', linewidth=2, label=f'y={y[mid_y]:.2f}')
    mid_x = len(x) // 2
    ax3.plot(y, u[:, mid_x], 'r--', linewidth=2, label=f'x={x[mid_x]:.2f}')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('u')
    ax3.set_title('Cross-sections')
    ax3.legend()
    ax3.grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


# =============================================================================
# Dataset Class
# =============================================================================

class Poisson2DDataset:
    """Dataset class for 2D Poisson equation solutions."""

    def __init__(self):
        self.x = None
        self.y = None
        self.solutions = []
        self.n_samples = 0

    def __getitem__(self, idx):
        if idx >= self.n_samples or idx < 0:
            raise IndexError(f"Index {idx} out of range")
        return self.solutions[idx]

    def __len__(self):
        return self.n_samples

    def add_sample(self, f_source, bc_left, bc_right, bc_bottom, bc_top, u_solution):
        """Add a sample to the dataset."""
        sample = {
            'boundary_conditions': {
                'bc_left': bc_left,
                'bc_right': bc_right,
                'bc_bottom': bc_bottom,
                'bc_top': bc_top
            },
            'u_solution': u_solution,
            'x': self.x,
            'y': self.y,
            'f_source': f_source
        }
        self.solutions.append(sample)
        self.n_samples += 1

    def save_to_hdf5(self, filepath):
        """Save dataset to HDF5 file."""
        all_bc_left = []
        all_bc_right = []
        all_bc_bottom = []
        all_bc_top = []
        all_u_solutions = []
        all_f_source = []

        for sample in self.solutions:
            bc = sample['boundary_conditions']
            all_bc_left.append(bc['bc_left'])
            all_bc_right.append(bc['bc_right'])
            all_bc_bottom.append(bc['bc_bottom'])
            all_bc_top.append(bc['bc_top'])
            all_u_solutions.append(sample['u_solution'])
            all_f_source.append(sample['f_source'])

        with h5py.File(filepath, 'w') as f:
            common = f.create_group('common')
            common.create_dataset('x', data=self.x)
            common.create_dataset('y', data=self.y)

            solutions = f.create_group('solutions')
            solutions.create_dataset('bc_left', data=np.array(all_bc_left))
            solutions.create_dataset('bc_right', data=np.array(all_bc_right))
            solutions.create_dataset('bc_bottom', data=np.array(all_bc_bottom))
            solutions.create_dataset('bc_top', data=np.array(all_bc_top))
            solutions.create_dataset('u_solutions', data=np.array(all_u_solutions))
            solutions.create_dataset('f_source', data=np.array(all_f_source))

            metadata = f.create_group('metadata')
            metadata.attrs['n_samples'] = self.n_samples
            metadata.attrs['nx'] = len(self.x)
            metadata.attrs['ny'] = len(self.y)
            metadata.attrs['dim'] = 2

        print(f"Saved 2D dataset with {self.n_samples} samples to {filepath}")

    @classmethod
    def load_from_hdf5(cls, filepath):
        """Load dataset from HDF5 file."""
        dataset = cls()

        with h5py.File(filepath, 'r') as f:
            dataset.x = f['common/x'][:]
            dataset.y = f['common/y'][:]

            bc_left = f['solutions/bc_left'][:]
            bc_right = f['solutions/bc_right'][:]
            bc_bottom = f['solutions/bc_bottom'][:]
            bc_top = f['solutions/bc_top'][:]
            u_solutions = f['solutions/u_solutions'][:]
            f_source = f['solutions/f_source'][:]

            for i in range(len(u_solutions)):
                dataset.add_sample(
                    f_source[i], bc_left[i], bc_right[i],
                    bc_bottom[i], bc_top[i], u_solutions[i]
                )

        print(f"Loaded 2D dataset with {dataset.n_samples} samples")
        return dataset


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_random_bc(Nx, Ny, bc_type='constant', low=1.0, high=10.0):
    """
    Generate random boundary conditions.

    Args:
        bc_type: 'constant' - same value on all 4 sides
                 'two_sides' - left/right same, top/bottom different
                 'four_sides' - different value on each side
                 'varying' - smoothly varying along boundaries (interpolated corners)
    """
    if bc_type == 'constant':
        # Same value on all sides
        val = np.random.uniform(low, high)
        return {
            'left': val * np.ones(Ny),
            'right': val * np.ones(Ny),
            'bottom': val * np.ones(Nx),
            'top': val * np.ones(Nx)
        }
    elif bc_type == 'two_sides':
        # Left/right have same value, top/bottom have different value
        val_lr = np.random.uniform(low, high)  # left and right same
        val_bottom = np.random.uniform(low, high)
        val_top = np.random.uniform(low, high)
        return {
            'left': val_lr * np.ones(Ny),
            'right': val_lr * np.ones(Ny),
            'bottom': val_bottom * np.ones(Nx),
            'top': val_top * np.ones(Nx)
        }
    elif bc_type == 'four_sides':
        # Different random constant on each side
        val_left = np.random.uniform(low, high)
        val_right = np.random.uniform(low, high)
        val_bottom = np.random.uniform(low, high)
        val_top = np.random.uniform(low, high)
        return {
            'left': val_left * np.ones(Ny),
            'right': val_right * np.ones(Ny),
            'bottom': val_bottom * np.ones(Nx),
            'top': val_top * np.ones(Nx)
        }
    else:  # varying
        # Random corner values with smooth interpolation along edges
        corners = np.random.uniform(low, high, 4)  # [bl, br, tl, tr]
        x_frac = np.linspace(0, 1, Nx)
        y_frac = np.linspace(0, 1, Ny)

        bc_bottom = corners[0] + (corners[1] - corners[0]) * x_frac
        bc_top = corners[2] + (corners[3] - corners[2]) * x_frac
        bc_left = corners[0] + (corners[2] - corners[0]) * y_frac
        bc_right = corners[1] + (corners[3] - corners[1]) * y_frac

        return {
            'left': bc_left,
            'right': bc_right,
            'bottom': bc_bottom,
            'top': bc_top
        }


def gen_2d_datasets(n_samples, N=51, bc_type='constant', parent_dir="./"):
    """Generate 2D Poisson dataset."""
    PATH = os.path.join(parent_dir, "data_2d_poisson")
    os.makedirs(PATH, exist_ok=True)

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    dataset = Poisson2DDataset()
    dataset.x = x
    dataset.y = y

    print(f"Generating {n_samples} 2D Poisson datasets...")
    print(f"Grid size: {N} x {N}")

    for i in range(n_samples):
        # Generate boundary conditions
        bc = generate_random_bc(N, N, bc_type=bc_type)

        # Constant source function
        f = np.ones((N, N))

        # Solve
        u = solve_poisson_2d(x, y, f, bc)

        # Plot first few samples
        if i < 3:
            bc_val = bc['left'][N//2]
            title = f"Sample {i}: BC ~ {bc_val:.2f}"
            save_path = os.path.join(PATH, f"sample_{i}_solution.png")
            plot_3d_solution(x, y, u, title=title, save_path=save_path)

        dataset.add_sample(f, bc['left'], bc['right'], bc['bottom'], bc['top'], u)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples")

    # Save
    hdf5_path = os.path.join(PATH, f"data_2d_{n_samples}.h5")
    dataset.save_to_hdf5(hdf5_path)

    return dataset


def test_solver():
    """Test the solver with zero BC (should produce dome shape)."""
    print("Testing 2D Poisson solver...")

    os.makedirs("./data_2d_poisson", exist_ok=True)

    N = 51
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    # Zero BC, constant f=1
    bc = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}
    f = np.ones((N, N))

    u = solve_poisson_2d(x, y, f, bc)

    print(f"Solution min: {u.min():.4f}, max: {u.max():.4f}")
    print("Expected: dome shape with peak at center")

    plot_3d_solution(x, y, u, title="Test: Zero BC, f=1",
                     save_path="./data_2d_poisson/test_dome.png")

    return u


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--grid_size", type=int, default=51)
    parser.add_argument("--bc_type", type=str, default='two_sides',
                        choices=['constant', 'two_sides', 'four_sides', 'varying'])
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()

    if args.test:
        test_solver()
    else:
        dataset = gen_2d_datasets(
            n_samples=args.n_samples,
            N=args.grid_size,
            bc_type=args.bc_type
        )
        print(f"\nDataset contains {len(dataset)} samples")
        print(f"Grid shape: {dataset[0]['u_solution'].shape}")