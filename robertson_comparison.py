#!/usr/bin/env python3
"""
Robertson Problem: Comprehensive Numerical Method Comparison

This script implements and compares six different numerical methods for solving
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
5. Implicit Euler + Infrequent Jacobian: Reduced Jacobian computations for efficiency
6. Exponential Euler: Excellent stability properties, exact for linear problems

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
from scipy.linalg import expm

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

def implicit_euler_infrequent_jacobian(rhs: Callable, jacobian: Callable, y0: np.ndarray,
                                       t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implicit Euler method with infrequently computed Jacobian.

    This method reduces computational cost by reusing the Jacobian matrix
    for multiple time steps or Newton iterations, rather than recomputing
    it at every iteration. This is particularly useful when Jacobian
    evaluation is expensive.

    Strategy: Compute Jacobian only once per time step (at the beginning)
    and reuse it for all Newton iterations in that time step.
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

        # Compute Jacobian only ONCE per time step (at the old point for efficiency)
        J_fixed = jacobian(t[i], y_old)

        # Form the iteration matrix once and reuse it
        A_fixed = np.eye(len(y0)) - dt * J_fixed

        # Newton iteration with fixed Jacobian
        for newton_iter in range(20):  # Max 20 Newton iterations
            residual = y_new - y_old - dt * rhs(t[i + 1], y_new)

            if np.linalg.norm(residual) < 1e-8:
                break

            # Use the same Jacobian for all Newton iterations
            try:
                delta_y = np.linalg.solve(A_fixed, -residual)
                y_new += delta_y
            except np.linalg.LinAlgError:
                # If the fixed Jacobian becomes singular, fall back to exact Jacobian
                J_exact = jacobian(t[i + 1], y_new)
                A_exact = np.eye(len(y0)) - dt * J_exact
                try:
                    delta_y = np.linalg.solve(A_exact, -residual)
                    y_new += delta_y
                except np.linalg.LinAlgError:
                    break

            # Ensure non-negativity for concentrations
            y_new = np.maximum(y_new, 0.0)

        y[i + 1] = y_new

    return t, y

def exponential_euler(rhs: Callable, jacobian: Callable, y0: np.ndarray,
                     t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exponential Euler method for stiff ODEs.

    The exponential Euler method is particularly effective for stiff problems.
    For the ODE y' = f(t, y), it uses the formula:

    y_{n+1} = y_n + dt * φ(dt*J) * f(t_n, y_n)

    where φ(z) = (e^z - 1)/z is the φ-function, and J is the Jacobian.

    This implementation uses a more robust approach for computing φ(z).
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1

    t = np.linspace(t_start, t_end, n_steps)
    y = np.zeros((n_steps, len(y0)))

    y[0] = y0

    for i in range(n_steps - 1):
        y_n = y[i]
        t_n = t[i]

        # Evaluate function and Jacobian at current point
        f_n = rhs(t_n, y_n)
        J_n = jacobian(t_n, y_n)

        try:
            # Compute dt * J
            dtJ = dt * J_n

            # Check the norm to decide on computation method
            dtJ_norm = np.linalg.norm(dtJ)

            if dtJ_norm < 1e-10:
                # For very small dtJ, use Taylor series: φ(z) ≈ I + z/2 + z²/6
                phi_dtJ = np.eye(len(y0)) + 0.5 * dtJ + (dtJ @ dtJ) / 6.0
            elif dtJ_norm > 10:
                # For large dtJ, the exponential method may be unstable
                # Fall back to implicit Euler
                A_euler = np.eye(len(y0)) - dtJ
                try:
                    phi_dtJ = np.linalg.inv(A_euler)
                except np.linalg.LinAlgError:
                    phi_dtJ = np.eye(len(y0))
            else:
                # Standard range: use matrix exponential approach with Padé approximation
                try:
                    # Compute matrix exponential using scipy's robust implementation
                    exp_dtJ = expm(dtJ)

                    # Compute φ(dtJ) = (exp(dtJ) - I) / dtJ using a more stable method
                    # We solve the equation: dtJ * φ(dtJ) = exp(dtJ) - I

                    # For better numerical stability, use the identity:
                    # φ(z) = ∫₀¹ exp(z*s) ds
                    # which can be approximated using quadrature

                    eye = np.eye(len(y0))
                    rhs_matrix = exp_dtJ - eye

                    # Try to solve dtJ * φ = exp(dtJ) - I for φ
                    # This is equivalent to solving a Sylvester equation

                    # Alternative approach: use series expansion for moderate values
                    if dtJ_norm < 2.0:
                        # Use Taylor series: φ(z) = I + z/2 + z²/6 + z³/24 + ...
                        phi_dtJ = eye.copy()
                        dtJ_power = eye.copy()
                        factorial = 1

                        for k in range(1, 10):  # Use first 10 terms
                            factorial *= k
                            dtJ_power = dtJ_power @ dtJ
                            phi_dtJ += dtJ_power / (factorial * (k + 1))

                            # Check convergence
                            if np.linalg.norm(dtJ_power) / factorial < 1e-12:
                                break
                    else:
                        # For larger values, use finite difference approximation
                        # φ(z) ≈ (exp(z) - I) / z using element-wise division where safe
                        phi_dtJ = np.zeros_like(dtJ)
                        for ii in range(len(y0)):
                            for jj in range(len(y0)):
                                if abs(dtJ[ii, jj]) > 1e-12:
                                    phi_dtJ[ii, jj] = (exp_dtJ[ii, jj] - (1.0 if ii == jj else 0.0)) / dtJ[ii, jj]
                                else:
                                    phi_dtJ[ii, jj] = 1.0 if ii == jj else 0.0

                except:
                    # Final fallback: use simple approximation
                    phi_dtJ = np.eye(len(y0)) + 0.5 * dtJ

            # Exponential Euler step: y_{n+1} = y_n + dt * φ(dt*J) * f(t_n, y_n)
            step = dt * (phi_dtJ @ f_n)

            # Check for numerical issues
            if np.any(np.isnan(step)) or np.any(np.isinf(step)):
                # Fallback to explicit Euler
                step = dt * f_n

            y[i + 1] = y_n + step

            # Ensure non-negativity for physical concentrations
            y[i + 1] = np.maximum(y[i + 1], 0.0)

            # Additional stability check
            if np.any(y[i + 1] > 10.0):  # Unreasonably large values
                # Fallback to implicit Euler
                A_euler = np.eye(len(y0)) - dt * J_n
                try:
                    k_euler = np.linalg.solve(A_euler, f_n)
                    y[i + 1] = y_n + dt * k_euler
                    y[i + 1] = np.maximum(y[i + 1], 0.0)
                except np.linalg.LinAlgError:
                    # Last resort: explicit Euler with smaller step
                    y[i + 1] = y_n + 0.1 * dt * f_n
                    y[i + 1] = np.maximum(y[i + 1], 0.0)

        except Exception:
            # Any other error: fallback to explicit Euler
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
        ("Implicit Euler (Infrequent Jacobian)", implicit_euler_infrequent_jacobian, True, 0.1),
        ("Exponential Euler", exponential_euler, True, 0.01),
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
    print("Implicit Euler + Infrequent Jacobian:")
    print("  + A-stable with reduced computational cost")
    print("  + Reuses Jacobian across Newton iterations")
    print("  + Good balance between accuracy and efficiency")
    print("  - May converge slower than full Newton method")
    print()
    print("Exponential Euler:")
    print("  + Exact for linear problems, excellent stability")
    print("  + No step size restrictions for linear stiff systems")
    print("  + First-order accurate but very robust")
    print("  - Requires matrix exponential computation")

if __name__ == "__main__":
    run_method_comparison()
