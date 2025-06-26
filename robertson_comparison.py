#!/usr/bin/env python3
"""
Robertson Problem: Comprehensive Numerical Method Comparison

This script implements and compares five different numerical methods for solving
the Robertson problem, a classic benchmark for stiff ODEs:

y1' = -0.04*y1 + 1e4*y2*y3
y2' = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2  
y3' = 3e7*y2^2

Initial conditions: y1(0)=1, y2(0)=0, y3(0)=0

The system is stiff due to widely separated time scales:
- Fast scale: τ_fast ~ 1/(3e7*y2) ~ 1e-7 seconds (very fast)
- Slow scale: τ_slow ~ 1/0.04 = 25 seconds

METHODS IMPLEMENTED:
1. Explicit Euler: Simple but stability-limited for stiff problems
2. Implicit Euler + Newton: A-stable, robust for stiff systems
3. Implicit Euler + Fixed-point: Jacobian-free but slower convergence
4. Midpoint Rule + Newton: Second-order accurate with good stability
5. Midpoint Rule + Fixed-point: Higher accuracy but challenging for stiff problems
6. Rosenbrock 2nd Order: L-stable, efficient for stiff problems with 2nd order accuracy

OUTPUT:
- 3x2 subplot showing concentration vs time for each method
- Console output with runtime and function evaluation counts
- Mass conservation analysis (y1 + y2 + y3 = 1)
- Performance comparison and method characteristics

AUTHOR: Generated for numerical analysis comparison
DATE: 2025
"""

import time
from typing import Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt

# Global counter for function evaluations
function_evaluations = 0

def robertson_rhs(t: float, y: np.ndarray) -> np.ndarray:
    """
    Right-hand side of the Robertson problem.

    :param t: Time (not used in this autonomous system)
    :param y: State vector [y1, y2, y3]
    :returns: Derivative vector [y1', y2', y3']
    """
    global function_evaluations
    function_evaluations += 1

    y1, y2, y3 = y

    dy1dt = -0.04 * y1 + 1e4 * y2 * y3
    dy2dt = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
    dy3dt = 3e7 * y2**2

    return np.array([dy1dt, dy2dt, dy3dt])

def robertson_jacobian(t: float, y: np.ndarray) -> np.ndarray:
    """
    Jacobian matrix of the Robertson problem.

    :param t: Time (not used)
    :param y: State vector [y1, y2, y3]
    :returns: 3x3 Jacobian matrix
    """
    y1, y2, y3 = y
    global function_evaluations
    function_evaluations += 3

    J = np.array([
        [-0.04, 1e4 * y3, 1e4 * y2],
        [0.04, -1e4 * y3 - 6e7 * y2, -1e4 * y2],
        [0.0, 6e7 * y2, 0.0]
    ])

    return J

def explicit_euler(
    rhs: Callable, y0: np.ndarray, t_span: Tuple[float, float], dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Explicit Euler method.
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1

    t = np.linspace(t_start, t_end, n_steps)
    y = np.zeros((n_steps, len(y0)))

    y[0] = y0

    for i in range(n_steps - 1):
        dy = rhs(t[i], y[i])
        y[i + 1] = y[i] + dt * dy

        # Check for numerical instability
        if np.any(np.isnan(y[i + 1])) or np.any(np.isinf(y[i + 1])) or np.any(y[i + 1] < -1e-6):
            # Return partial solution
            return t[:i+2], y[:i+2]

    return t, y

def implicit_euler_newton(rhs: Callable, jacobian: Callable, y0: np.ndarray, 
                         t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implicit Euler method with Newton's method for nonlinear system.
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1

    t = np.linspace(t_start, t_end, n_steps)
    y = np.zeros((n_steps, len(y0)))

    y[0] = y0

    for i in range(n_steps - 1):
        # Solve: y_{n+1} - y_n - dt * f(t_{n+1}, y_{n+1}) = 0
        y_old = y[i]
        y_new = y_old + dt * rhs(t[i], y_old)  # Better initial guess

        # Newton iteration
        for _ in range(20):  # Max 20 Newton iterations
            residual = y_new - y_old - dt * rhs(t[i + 1], y_new)

            if np.linalg.norm(residual) < 1e-8:
                break

            J = np.eye(len(y0)) - dt * jacobian(t[i + 1], y_new)
            try:
                delta_y = np.linalg.solve(J, -residual)
                y_new += delta_y
            except np.linalg.LinAlgError:
                break

            # Ensure non-negativity for concentrations
            y_new = np.maximum(y_new, 0.0)

        y[i + 1] = y_new

    return t, y

def implicit_euler_fixedpoint(rhs: Callable, y0: np.ndarray, 
                             t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implicit Euler method with fixed-point iteration.
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1

    t = np.linspace(t_start, t_end, n_steps)
    y = np.zeros((n_steps, len(y0)))

    y[0] = y0

    for i in range(n_steps - 1):
        y_old = y[i]
        y_new = y_old + dt * rhs(t[i], y_old)  # Better initial guess

        # Fixed-point iteration with adaptive relaxation
        relaxation = 0.5
        for fp_iter in range(100):
            y_next = y_old + dt * rhs(t[i + 1], y_new)
            y_next = np.maximum(y_next, 0.0)  # Ensure non-negativity

            if np.linalg.norm(y_next - y_new) < 1e-8:
                y_new = y_next
                break

            # Adaptive relaxation
            if fp_iter > 10 and np.linalg.norm(y_next - y_new) > np.linalg.norm(y_new - y_old):
                relaxation *= 0.8

            y_new = (1 - relaxation) * y_new + relaxation * y_next

        y[i + 1] = y_new

    return t, y

def midpoint_newton(rhs: Callable, jacobian: Callable, y0: np.ndarray, 
                   t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Midpoint rule with Newton's method for nonlinear system.
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1

    t = np.linspace(t_start, t_end, n_steps)
    y = np.zeros((n_steps, len(y0)))

    y[0] = y0

    for i in range(n_steps - 1):
        y_old = y[i]
        y_new = y_old + dt * rhs(t[i], y_old)  # Better initial guess
        t_mid = t[i] + dt/2

        # Newton iteration
        for _ in range(20):
            y_mid = 0.5 * (y_old + y_new)
            residual = y_new - y_old - dt * rhs(t_mid, y_mid)

            if np.linalg.norm(residual) < 1e-8:
                break

            J = np.eye(len(y0)) - 0.5 * dt * jacobian(t_mid, y_mid)
            try:
                delta_y = np.linalg.solve(J, -residual)
                y_new += delta_y
            except np.linalg.LinAlgError:
                break

            y_new = np.maximum(y_new, 0.0)

        y[i + 1] = y_new

    return t, y

def midpoint_fixedpoint(rhs: Callable, y0: np.ndarray, 
                       t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Midpoint rule with fixed-point iteration.
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1

    t = np.linspace(t_start, t_end, n_steps)
    y = np.zeros((n_steps, len(y0)))

    y[0] = y0

    for i in range(n_steps - 1):
        y_old = y[i]
        y_new = y_old + dt * rhs(t[i], y_old)  # Better initial guess
        t_mid = t[i] + dt/2

        # Fixed-point iteration with adaptive relaxation
        relaxation = 0.5
        for fp_iter in range(100):
            y_mid = 0.5 * (y_old + y_new)
            y_next = y_old + dt * rhs(t_mid, y_mid)
            y_next = np.maximum(y_next, 0.0)

            if np.linalg.norm(y_next - y_new) < 1e-8:
                y_new = y_next
                break

            # Adaptive relaxation
            if fp_iter > 10 and np.linalg.norm(y_next - y_new) > np.linalg.norm(y_new - y_old):
                relaxation *= 0.8

            y_new = (1 - relaxation) * y_new + relaxation * y_next

        y[i + 1] = y_new

    return t, y

def rosenbrock_2nd_order(rhs: Callable, jacobian: Callable, y0: np.ndarray, 
                        t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    2nd order Rosenbrock method (Ros2).

    This implements the classical Ros2 method, which is L-stable and second-order accurate.
    The scheme is:

    k1 = (I - γ*dt*J)^(-1) * f(t_n, y_n)
    k2 = (I - γ*dt*J)^(-1) * [f(t_n + dt, y_n + dt*k1) - 2*dt*J*k1]
    y_{n+1} = y_n + dt*k2

    where γ = 1 + 1/√2 ≈ 1.7071
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1

    t = np.linspace(t_start, t_end, n_steps)
    y = np.zeros((n_steps, len(y0)))

    y[0] = y0

    # Rosenbrock parameter for ROS2
    gamma = 1.0 + 1.0/np.sqrt(2.0)

    for i in range(n_steps - 1):
        y_n = y[i]
        t_n = t[i]

        # Evaluate function and Jacobian at current point
        f_n = rhs(t_n, y_n)
        J_n = jacobian(t_n, y_n)

        # Form the iteration matrix (I - γ*dt*J)
        W = np.eye(len(y0)) - gamma * dt * J_n

        try:
            # First stage: solve (I - γ*dt*J) * k1 = f(t_n, y_n)
            k1 = np.linalg.solve(W, f_n)

            # Intermediate point for second stage
            y_temp = y_n + dt * k1
            f_temp = rhs(t_n + dt, y_temp)

            # Second stage: solve (I - γ*dt*J) * k2 = f(t_n + dt, y_n + dt*k1) - 2*dt*J*k1
            rhs_k2 = f_temp - 2.0 * dt * np.dot(J_n, k1)
            k2 = np.linalg.solve(W, rhs_k2)

            # Update: y_{n+1} = y_n + dt*k2
            y[i + 1] = y_n + dt * k2

            # Ensure non-negativity for physical concentrations
            y[i + 1] = np.maximum(y[i + 1], 0.0)

        except np.linalg.LinAlgError:
            # Fallback to implicit Euler if matrix is singular
            try:
                A_euler = np.eye(len(y0)) - dt * J_n
                rhs_euler = f_n
                k_euler = np.linalg.solve(A_euler, rhs_euler)
                y[i + 1] = y_n + dt * k_euler
                y[i + 1] = np.maximum(y[i + 1], 0.0)
            except np.linalg.LinAlgError:
                # Last resort: explicit Euler
                y[i + 1] = y_n + dt * f_n
                y[i + 1] = np.maximum(y[i + 1], 0.0)

    return t, y

def run_method_comparison():
    """
    Run comparison with multiple time step sizes to show method behavior.
    """
    global function_evaluations

    # Problem setup
    y0 = np.array([1.0, 0.0, 0.0])
    t_span = (0.0, 40.0)

    methods = [
        ("Explicit Euler", explicit_euler, False, 0.0005),
        ("Implicit Euler (Newton)", implicit_euler_newton, True, 0.1),
        ("Implicit Euler (Fixed-point)", implicit_euler_fixedpoint, False, 0.001),
        ("Midpoint (Newton)", midpoint_newton, True, 0.02),
        ("Midpoint (Fixed-point)", midpoint_fixedpoint, False, 0.002),
        ("Rosenbrock 2nd Order", rosenbrock_2nd_order, True, 0.005),
    ]

    print("Robertson Problem: Method Comparison")
    print("=" * 80)
    print(f"Time span: {t_span}")
    print(f"Initial conditions: y1={y0[0]}, y2={y0[1]}, y3={y0[2]}")
    print()

    results = []

    for method_name, method_func, needs_jacobian, dt in methods:
        print(f"Running {method_name}...")

        function_evaluations = 0
        start_time = time.time()

        try:
            if needs_jacobian:
                t, y = method_func(robertson_rhs, robertson_jacobian, y0, t_span, dt)
            else:
                t, y = method_func(robertson_rhs, y0, t_span, dt)

            elapsed_time = time.time() - start_time

            results.append((method_name, t, y, elapsed_time, function_evaluations))

            print(f"  Runtime: {elapsed_time:.4f} seconds")
            print(f"  Function evaluations: {function_evaluations}")
            print(f"  Time steps completed: {len(t)}/{int((t_span[1] - t_span[0])/dt + 1)}")

            if len(y) > 0 and not np.any(np.isnan(y[-1, :])):
                print(f"  Final values: y1={y[-1,0]:.6e}, y2={y[-1,1]:.6e}, y3={y[-1,2]:.6e}")
                total_mass = np.sum(y[-1, :])
                print(f"  Mass conservation: {total_mass:.10f} (error: {abs(total_mass-1.0):.2e})")
            else:
                print("  NUMERICAL INSTABILITY")
            print()

        except Exception as e:
            print(f"  FAILED: {e}")
            # Create dummy result for plotting
            t_dummy = np.linspace(t_span[0], t_span[1], 10)
            y_dummy = np.ones((10, 3)) * np.array([1.0, 0.0, 0.0])
            results.append((method_name + " (FAILED)", t_dummy, y_dummy, 0.0, 0))
            print()

    # Create the 3x2 plot with shared axes for synchronized zooming/panning
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle("Robertson Problem Comparison", fontsize=16, fontweight="bold")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    labels = ['y₁ (reactant)', 'y₂ (intermediate)', 'y₃ (product)']

    # Create all secondary axes first and link them
    secondary_axes = []
    for i in range(len(results)):
        row = i // 2
        col = i % 2
        ax2 = axes[row, col].twinx()
        secondary_axes.append(ax2)
    
    # Share the secondary y-axes manually
    for ax2 in secondary_axes[1:]:
        ax2.sharey(secondary_axes[0])

    for i, (method_name, t, y, runtime, func_evals) in enumerate(results):
        # Convert linear index to 2D indices
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        ax2 = secondary_axes[i]

        # Plot all three species on left y-axis
        for j in range(3):
            if len(y) > 0:
                ax.semilogy(t, y[:, j], color=colors[j], label=labels[j], linewidth=2)

        # Calculate and plot mass conservation
        if len(y) > 0:
            total_mass = np.sum(y, axis=1)  # Sum over species at each time point
            mass_error = np.abs(total_mass - 1.0)  # Error from ideal conservation (should be 1.0)
            
            # Plot mass conservation error on right axis
            line_mass = ax2.semilogy(t, mass_error, 'r--', alpha=0.7, linewidth=1.5, 
                                   label='Mass conservation error')
            ax2.set_ylabel("Mass conservation error", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Set consistent y-limits for mass conservation (only on the first axis, others will follow)
            if i == 0:
                ax2.set_ylim(1e-16, 1e-1)

        # Set labels and formatting for left axis
        ax.set_title(f"{method_name}\nRuntime: {runtime:.4f}s, Function evaluations: {func_evals}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration (log scale)")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1e-12, 2)
        ax.set_xlim(t_span[0], t_span[1])
        
        # Add mass conservation legend to right axis
        if len(y) > 0:
            ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # Print summary table
    print("\nPerformance Summary:")
    print("-" * 100)
    print(f"{'Method':<30} {'Runtime (s)':<12} {'Func Evals':<12} {'Efficiency':<15} {'Status':<15}")
    print("-" * 100)

    for method_name, t, y, runtime, func_evals in results:
        if runtime > 0 and func_evals > 0:
            efficiency = func_evals / runtime
            status = "SUCCESS" if len(t) > 100 and not np.any(np.isnan(y[-1, :])) else "UNSTABLE"
        else:
            efficiency = 0
            status = "FAILED"

        print(f"{method_name:<30} {runtime:<12.4f} {func_evals:<12d} {efficiency:<15.0f} {status:<15}")

    print("-" * 100)

    # Show method characteristics
    print("\nMethod Characteristics:")
    print("-" * 60)
    print("Explicit Euler:")
    print("  + Simple implementation, low cost per step")
    print("  - Stability limited by stiffness (very small dt required)")
    print()
    print("Implicit Euler + Newton:")
    print("  + A-stable, good for stiff problems")
    print("  + Quadratic convergence")
    print("  - Requires Jacobian computation and matrix solve")
    print()
    print("Implicit Euler + Fixed-point:")
    print("  + No Jacobian required")
    print("  - Linear convergence, may not converge for stiff problems")
    print()
    print("Midpoint + Newton:")
    print("  + Second-order accuracy")
    print("  + Good stability properties")
    print("  - More expensive than Implicit Euler")
    print()
    print("Midpoint + Fixed-point:")
    print("  + Second-order accuracy, no Jacobian")
    print("  - Convergence issues for stiff problems")
    print()
    print("Rosenbrock 2nd Order:")
    print("  + L-stable, excellent for stiff problems")
    print("  + Second-order accuracy")
    print("  + Efficient (only one matrix factorization per step)")
    print("  - Requires Jacobian computation")

if __name__ == "__main__":
    run_method_comparison()
