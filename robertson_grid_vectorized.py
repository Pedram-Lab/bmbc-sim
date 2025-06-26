#!/usr/bin/env python3
"""
Vectorized Robertson Problem: Multiple ODEs on a 2D Grid

This script demonstrates how to solve multiple Robertson ODEs concurrently
using vectorized operations. Each point on an n×n grid has its own Robertson
system with concentrations y1, y2, y3.

Robertson Problem:
y1' = -0.04*y1 + 1e4*y2*y3
y2' = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
y3' = 3e7*y2^2

Initial conditions: y1(0)=1, y2(0)=0, y3(0)=0 (uniform across grid)

VECTORIZATION STRATEGY:
- Solution state: 3 × n² matrix (3 species × n² grid points)
- Jacobian: 9 × n² matrix (3×3 blocks × n² grid points)
- All operations (function evaluation, Jacobian, linear solve) vectorized
- Methods: Implicit Euler (Newton) vs Explicit Euler comparison

AUTHOR: Generated for vectorized ODE solving demonstration
DATE: 2025
"""

import time
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

def robertson_rhs_vectorized(t: float, Y: np.ndarray) -> np.ndarray:
    """
    Vectorized right-hand side of the Robertson problem for multiple systems.

    :param t: Time (not used in this autonomous system)
    :param Y: State matrix of shape (3, n²) where:
              Y[0, :] = y1 values for all grid points
              Y[1, :] = y2 values for all grid points
              Y[2, :] = y3 values for all grid points
    :returns: Derivative matrix of same shape as Y
    """
    y1, y2, y3 = Y[0, :], Y[1, :], Y[2, :]

    dy1dt = -0.04 * y1 + 1e4 * y2 * y3
    dy2dt = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
    dy3dt = 3e7 * y2**2

    return np.array([dy1dt, dy2dt, dy3dt])

def robertson_jacobian_vectorized(t: float, Y: np.ndarray) -> np.ndarray:
    """
    Vectorized Jacobian computation for multiple Robertson systems.

    :param t: Time (not used)
    :param Y: State matrix of shape (3, n²)
    :returns: Jacobian matrix of shape (9, n²) where each column contains
              the 9 elements of the 3×3 Jacobian for one grid point,
              stored in row-major order: [J[0,0], J[0,1], J[0,2], J[1,0], ...]
    """
    n_points = Y.shape[1]
    y1, y2, y3 = Y[0, :], Y[1, :], Y[2, :]

    # Compute Jacobian elements for all grid points simultaneously
    J = np.zeros((9, n_points))

    # Row 0: ∂f₁/∂y₁, ∂f₁/∂y₂, ∂f₁/∂y₃
    J[0, :] = -0.04                    # ∂f₁/∂y₁
    J[1, :] = 1e4 * y3                # ∂f₁/∂y₂
    J[2, :] = 1e4 * y2                # ∂f₁/∂y₃

    # Row 1: ∂f₂/∂y₁, ∂f₂/∂y₂, ∂f₂/∂y₃
    J[3, :] = 0.04                    # ∂f₂/∂y₁
    J[4, :] = -1e4 * y3 - 6e7 * y2   # ∂f₂/∂y₂
    J[5, :] = -1e4 * y2               # ∂f₂/∂y₃

    # Row 2: ∂f₃/∂y₁, ∂f₃/∂y₂, ∂f₃/∂y₃
    J[6, :] = 0.0                     # ∂f₃/∂y₁
    J[7, :] = 6e7 * y2                # ∂f₃/∂y₂
    J[8, :] = 0.0                     # ∂f₃/∂y₃

    return J

def solve_vectorized_linear_systems(A_vec: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    """
    Solve multiple 3×3 linear systems Ax = b simultaneously using vectorized operations.

    :param A_vec: Matrix coefficients of shape (9, n²) where each column
                  contains a 3×3 matrix in row-major order
    :param b_vec: Right-hand sides of shape (3, n²)
    :returns: Solutions x of shape (3, n²)
    """
    n_points = A_vec.shape[1]

    # Reshape A_vec from (9, n²) to (n², 3, 3) for vectorized solve
    A_batch = A_vec.T.reshape(n_points, 3, 3)  # (n², 3, 3)
    b_batch = b_vec.T                          # (n², 3)

    try:
        # Use NumPy's vectorized solve - this is much faster than loop-based solving!
        # np.linalg.solve can handle batch operations: (n², 3, 3) @ (n², 3, 1) -> (n², 3, 1)
        # So we need to add a dimension to b_batch
        b_batch_expanded = b_batch[..., np.newaxis]  # (n², 3, 1)
        x_batch_expanded = np.linalg.solve(A_batch, b_batch_expanded)  # (n², 3, 1)
        x_batch = x_batch_expanded.squeeze(-1)  # (n², 3)
        x_vec = x_batch.T  # (3, n²)
        return x_vec

    except np.linalg.LinAlgError:
        # Fallback: solve each system individually if batch solve fails
        x_vec = np.zeros((3, n_points))

        for i in range(n_points):
            A_i = A_batch[i]  # (3, 3)
            b_i = b_batch[i]  # (3,)

            try:
                # Check condition number to detect potential singularity
                cond_num = np.linalg.cond(A_i)
                if cond_num > 1e12:
                    # Use pseudo-inverse for ill-conditioned systems
                    x_vec[:, i] = np.linalg.pinv(A_i) @ b_i
                else:
                    x_vec[:, i] = np.linalg.solve(A_i, b_i)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                try:
                    x_vec[:, i] = np.linalg.pinv(A_i) @ b_i
                except:
                    # Last resort: zero update
                    x_vec[:, i] = 0.0

        return x_vec

def explicit_euler_vectorized(Y0: np.ndarray, t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized explicit Euler method for multiple Robertson systems.

    :param Y0: Initial conditions of shape (3, n²)
    :param t_span: Time span (t_start, t_end)
    :param dt: Time step size
    :returns: (time_array, solution_array) where solution has shape (n_steps, 3, n²)
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    n_species, n_points = Y0.shape

    # Initialize arrays
    t = np.linspace(t_start, t_end, n_steps)
    Y_history = np.zeros((n_steps, n_species, n_points))
    Y_history[0] = Y0

    Y_current = Y0.copy()

    for i in range(n_steps - 1):
        # Vectorized function evaluation for all grid points
        dY = robertson_rhs_vectorized(t[i], Y_current)

        # Euler step for all points simultaneously
        Y_current = Y_current + dt * dY

        # Ensure non-negativity
        Y_current = np.maximum(Y_current, 0.0)

        # Check for numerical instability
        if np.any(np.isnan(Y_current)) or np.any(np.isinf(Y_current)):
            print(f"Explicit Euler: Numerical instability at step {i}")
            break

        Y_history[i + 1] = Y_current

    return t[:i+2], Y_history[:i+2]

def implicit_euler_newton_vectorized(Y0: np.ndarray, t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized implicit Euler method with Newton's method for multiple Robertson systems.

    :param Y0: Initial conditions of shape (3, n²)
    :param t_span: Time span (t_start, t_end)
    :param dt: Time step size
    :returns: (time_array, solution_array) where solution has shape (n_steps, 3, n²)
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    n_species, n_points = Y0.shape

    # Initialize arrays
    t = np.linspace(t_start, t_end, n_steps)
    Y_history = np.zeros((n_steps, n_species, n_points))
    Y_history[0] = Y0

    Y_current = Y0.copy()
    failed_steps = 0

    for i in range(n_steps - 1):
        Y_old = Y_current.copy()

        # Initial guess: explicit Euler step
        f_old = robertson_rhs_vectorized(t[i], Y_old)
        Y_new = Y_old + dt * f_old
        Y_new = np.maximum(Y_new, 0.0)  # Ensure non-negativity

        # Newton iteration (vectorized across all grid points)
        newton_converged = False
        for newton_iter in range(20):  # Max 15 Newton iterations
            # Evaluate residual for all points: R = Y_new - Y_old - dt * f(Y_new)
            f_new = robertson_rhs_vectorized(t[i + 1], Y_new)

            # Check for numerical issues in function evaluation
            if np.any(np.isnan(f_new)) or np.any(np.isinf(f_new)):
                print(f"Implicit Euler: Function evaluation failed at step {i}, Newton iter {newton_iter}")
                failed_steps += 1
                break

            residual = Y_new - Y_old - dt * f_new

            # Check convergence
            residual_norm = np.linalg.norm(residual)
            if residual_norm < 1e-8:
                newton_converged = True
                break

            # Compute Jacobian for all points
            J_vec = robertson_jacobian_vectorized(t[i + 1], Y_new)

            # Form system matrix: I - dt*J for each point
            I_vec = np.tile(np.eye(3).flatten(), (n_points, 1)).T  # (9, n²)
            A_vec = I_vec - dt * J_vec

            # Solve linear systems: (I - dt*J) * delta_Y = -residual
            delta_Y = solve_vectorized_linear_systems(A_vec, -residual)

            # Check if linear solve produced reasonable results
            if np.any(np.isnan(delta_Y)) or np.any(np.isinf(delta_Y)):
                print(f"Implicit Euler: Linear solve produced invalid results at step {i}, Newton iter {newton_iter}")
                failed_steps += 1
                break

            # Ensure non-negativity
            Y_new += delta_Y
            Y_new = np.maximum(Y_new, 0.0)

        if not newton_converged and failed_steps > 5:
            print(f"Implicit Euler: Too many failed steps, stopping at step {i}")
            break

        Y_current = Y_new

        # Final check for numerical issues
        if np.any(np.isnan(Y_current)) or np.any(np.isinf(Y_current)):
            print(f"Implicit Euler: Numerical instability at step {i}")
            break

        Y_history[i + 1] = Y_current

    return t[:i+2], Y_history[:i+2]

def run_vectorized_comparison():
    """
    Run comparison between explicit and implicit Euler on a grid of Robertson systems.
    """
    print("Vectorized Robertson Problem: Grid Comparison")
    print("=" * 80)

    # Grid setup
    n_grid = 8  # 8×8 = 64 systems to better show vectorization benefits
    n_points = n_grid * n_grid

    print(f"Grid size: {n_grid}×{n_grid} = {n_points} Robertson systems")
    print(f"Each system: 3 species (y1, y2, y3)")
    print(f"Total variables: {3 * n_points}")
    print()

    # Create initial conditions: uniform [1, 0, 0] across all grid points
    Y0 = np.zeros((3, n_points))
    Y0[0, :] = 1.0  # y1 = 1
    Y0[1, :] = 0.0  # y2 = 0
    Y0[2, :] = 0.0  # y3 = 0

    # Time span and step sizes
    t_span = (0.0, 5.0)   # Shorter time span for demonstration
    dt_explicit = 0.0001  # Small step for stability
    dt_implicit = 0.01    # Larger step possible with implicit method

    print(f"Time span: {t_span}")
    print(f"Explicit Euler dt: {dt_explicit}")
    print(f"Implicit Euler dt: {dt_implicit}")
    print()

    # Run explicit Euler
    print("Running Explicit Euler (vectorized)...")
    start_time = time.time()
    t_exp, Y_exp = explicit_euler_vectorized(Y0, t_span, dt_explicit)
    time_explicit = time.time() - start_time

    print(f"  Runtime: {time_explicit:.4f} seconds")
    print(f"  Time steps: {len(t_exp)}")
    print(f"  Final time reached: {t_exp[-1]:.4f}")

    # Check mass conservation for explicit method
    if len(Y_exp) > 0:
        mass_exp = np.sum(Y_exp[-1], axis=0)  # Sum over species for each grid point
        mass_error_exp = np.abs(mass_exp - 1.0)
        print(f"  Mass conservation error (max): {np.max(mass_error_exp):.2e}")
        print(f"  Mass conservation error (mean): {np.mean(mass_error_exp):.2e}")
    print()

    # Run implicit Euler
    print("Running Implicit Euler + Newton (vectorized)...")
    start_time = time.time()
    t_imp, Y_imp = implicit_euler_newton_vectorized(Y0, t_span, dt_implicit)
    time_implicit = time.time() - start_time

    print(f"  Runtime: {time_implicit:.4f} seconds")
    print(f"  Time steps: {len(t_imp)}")
    print(f"  Final time reached: {t_imp[-1]:.4f}")

    # Check mass conservation for implicit method
    if len(Y_imp) > 0:
        mass_imp = np.sum(Y_imp[-1], axis=0)  # Sum over species for each grid point
        mass_error_imp = np.abs(mass_imp - 1.0)
        print(f"  Mass conservation error (max): {np.max(mass_error_imp):.2e}")
        print(f"  Mass conservation error (mean): {np.mean(mass_error_imp):.2e}")
    print()

    # Performance comparison
    steps_per_sec_exp = len(t_exp) * n_points / time_explicit if time_explicit > 0 else 0
    steps_per_sec_imp = len(t_imp) * n_points / time_implicit if time_implicit > 0 else 0

    print("Performance Summary:")
    print("-" * 60)
    print(f"{'Method':<25} {'Runtime (s)':<12} {'Steps/Point':<12} {'Steps/s/Point':<15}")
    print("-" * 60)
    print(f"{'Explicit Euler':<25} {time_explicit:<12.4f} {len(t_exp):<12d} {steps_per_sec_exp:<15.0f}")
    print(f"{'Implicit Euler':<25} {time_implicit:<12.4f} {len(t_imp):<12d} {steps_per_sec_imp:<15.0f}")
    print("-" * 60)

    if time_explicit > 0 and time_implicit > 0:
        speedup = time_explicit / time_implicit
        print(f"Speedup (implicit vs explicit): {speedup:.2f}x")

        efficiency_ratio = (steps_per_sec_imp * dt_implicit) / (steps_per_sec_exp * dt_explicit)
        print(f"Efficiency ratio (time advancement per second): {efficiency_ratio:.2f}x")
    print()

    # Create visualization
    create_grid_visualization(n_grid, t_exp, Y_exp, t_imp, Y_imp)

def create_grid_visualization(n_grid: int, t_exp: np.ndarray, Y_exp: np.ndarray,
                            t_imp: np.ndarray, Y_imp: np.ndarray):
    """
    Create visualization comparing the two methods on the grid.
    """
    # Select a few representative grid points for time series plots
    n_points = n_grid * n_grid

    # Choose grid points: corners and center
    indices = [
        0,                           # Top-left corner
        n_grid - 1,                 # Top-right corner
        n_grid * (n_grid - 1),      # Bottom-left corner
        n_points - 1,               # Bottom-right corner
        n_points // 2               # Center (approximately)
    ]

    labels = ['Top-left', 'Top-right', 'Bottom-left', 'Bottom-right', 'Center']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Vectorized Robertson Problem: {n_grid}×{n_grid} Grid Comparison',
                fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    species_names = ['y₁ (reactant)', 'y₂ (intermediate)', 'y₃ (product)']

    # Plot time series for selected points
    for i, (idx, label) in enumerate(zip(indices, labels)):
        if i >= 5:  # Only plot first 5 points
            break

        row = i // 3
        col = i % 3
        ax = axes[row, col]

        # Plot explicit Euler results
        if len(Y_exp) > 0:
            for j in range(3):
                ax.semilogy(t_exp, Y_exp[:, j, idx], color=colors[j],
                          linestyle='-', linewidth=2, alpha=0.8,
                          label=f'{species_names[j]} (Explicit)' if i == 0 else "")

        # Plot implicit Euler results
        if len(Y_imp) > 0:
            for j in range(3):
                ax.semilogy(t_imp, Y_imp[:, j, idx], color=colors[j],
                          linestyle='--', linewidth=2, alpha=0.8,
                          label=f'{species_names[j]} (Implicit)' if i == 0 else "")

        ax.set_title(f'Grid Point: {label} (index {idx})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration (log scale)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1e-12, 2)

        if i == 0:  # Add legend to first subplot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Remove empty subplot
    if len(indices) < 6:
        axes[1, 2].remove()

    # Final state comparison (grid visualization)
    ax_final = axes[1, 2] if len(indices) >= 6 else fig.add_subplot(2, 3, 6)

    # Show final y1 concentration across the grid
    if len(Y_imp) > 0:
        final_y1 = Y_imp[-1, 0, :].reshape(n_grid, n_grid)
        im = ax_final.imshow(final_y1, cmap='viridis', origin='lower')
        ax_final.set_title('Final y₁ Concentration\n(Implicit Euler)')
        ax_final.set_xlabel('Grid X')
        ax_final.set_ylabel('Grid Y')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_final)
        cbar.set_label('y₁ concentration')

    plt.tight_layout()
    plt.show()

    # Print some statistics
    if len(Y_exp) > 0 and len(Y_imp) > 0:
        print("Final State Statistics:")
        print("-" * 40)

        # Compare final states at same time point (if possible)
        if abs(t_exp[-1] - t_imp[-1]) < 0.1:  # If both reached similar final time
            diff_y1 = np.abs(Y_exp[-1, 0, :] - Y_imp[-1, 0, :])
            diff_y2 = np.abs(Y_exp[-1, 1, :] - Y_imp[-1, 1, :])
            diff_y3 = np.abs(Y_exp[-1, 2, :] - Y_imp[-1, 2, :])

            print(f"Max difference in y₁: {np.max(diff_y1):.2e}")
            print(f"Max difference in y₂: {np.max(diff_y2):.2e}")
            print(f"Max difference in y₃: {np.max(diff_y3):.2e}")

            print(f"Mean difference in y₁: {np.mean(diff_y1):.2e}")
            print(f"Mean difference in y₂: {np.mean(diff_y2):.2e}")
            print(f"Mean difference in y₃: {np.mean(diff_y3):.2e}")

if __name__ == "__main__":
    run_vectorized_comparison()
