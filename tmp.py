import warnings
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

class ChemicalReactionODE:
    """
    Solver for stiff ODE system modeling 5 interacting chemical species (a, b, c, ab, ac)
    with complex formation reactions: a + b <-> ab and a + c <-> ac
    
    Uses implicit midpoint rule for time stepping and Newton's method for 
    nonlinear system solution.
    """

    def __init__(self, k_ab_on: float, k_ab_off: float, k_ac_on: float, k_ac_off: float):
        """
        Initialize the chemical reaction system.
        
        :param k_ab_on: Rate constant for a + b -> ab reaction
        :param k_ab_off: Rate constant for ab -> a + b reaction
        :param k_ac_on: Rate constant for a + c -> ac reaction
        :param k_ac_off: Rate constant for ac -> a + c reaction
        """
        self.k_ab_on = k_ab_on
        self.k_ab_off = k_ab_off
        self.k_ac_on = k_ac_on
        self.k_ac_off = k_ac_off

    def reaction_rates(self, y: np.ndarray) -> np.ndarray:
        """
        Compute reaction rates dy/dt for the system.
        
        System: a + b <-> ab, a + c <-> ac
        da/dt = -k_ab_on * a * b + k_ab_off * ab - k_ac_on * a * c + k_ac_off * ac
        db/dt = -k_ab_on * a * b + k_ab_off * ab
        dc/dt = -k_ac_on * a * c + k_ac_off * ac
        d(ab)/dt = k_ab_on * a * b - k_ab_off * ab
        d(ac)/dt = k_ac_on * a * c - k_ac_off * ac
        
        :param y: Current concentrations [a, b, c, ab, ac]
        :returns: Reaction rates [da/dt, db/dt, dc/dt, d(ab)/dt, d(ac)/dt]
        """
        a, b, c, ab, ac = y

        dadt = -self.k_ab_on * a * b + self.k_ab_off * ab - self.k_ac_on * a * c + self.k_ac_off * ac
        dbdt = -self.k_ab_on * a * b + self.k_ab_off * ab
        dcdt = -self.k_ac_on * a * c + self.k_ac_off * ac
        dabdt = self.k_ab_on * a * b - self.k_ab_off * ab
        dacdt = self.k_ac_on * a * c - self.k_ac_off * ac

        return np.array([dadt, dbdt, dcdt, dabdt, dacdt])

    def reaction_rates_batch(self, y_batch: np.ndarray) -> np.ndarray:
        """
        Compute reaction rates for multiple systems simultaneously.
        Optimized for vectorized operations across many spatial points.
        
        :param y_batch: Concentrations shape (n_points, 5) where each row is [a, b, c, ab, ac]
        :returns: Reaction rates shape (n_points, 5)
        """
        # Extract all species for all points
        a, b, c, ab, ac = y_batch[:, 0], y_batch[:, 1], y_batch[:, 2], y_batch[:, 3], y_batch[:, 4]
        
        # Vectorized reaction rate computation
        dadt = -self.k_ab_on * a * b + self.k_ab_off * ab - self.k_ac_on * a * c + self.k_ac_off * ac
        dbdt = -self.k_ab_on * a * b + self.k_ab_off * ab
        dcdt = -self.k_ac_on * a * c + self.k_ac_off * ac
        dabdt = self.k_ab_on * a * b - self.k_ab_off * ab
        dacdt = self.k_ac_on * a * c - self.k_ac_off * ac
        
        return np.column_stack([dadt, dbdt, dcdt, dabdt, dacdt])

    def jacobian(self, y: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix of the reaction system.
        
        J[i,j] = ∂(dy_i/dt)/∂y_j
        
        :param y: Current concentrations [a, b, c, ab, ac]
        :returns: 5x5 Jacobian matrix
        """
        a, b, c, ab, ac = y
        
        # Partial derivatives of reaction rates
        # Row 0: da/dt derivatives
        # Row 1: db/dt derivatives  
        # Row 2: dc/dt derivatives
        # Row 3: d(ab)/dt derivatives
        # Row 4: d(ac)/dt derivatives
        J = np.array([
            # ∂/∂a, ∂/∂b, ∂/∂c, ∂/∂ab, ∂/∂ac
            [-self.k_ab_on * b - self.k_ac_on * c, -self.k_ab_on * a, -self.k_ac_on * a, self.k_ab_off, self.k_ac_off],  # da/dt
            [-self.k_ab_on * b, -self.k_ab_on * a, 0.0, self.k_ab_off, 0.0],  # db/dt
            [-self.k_ac_on * c, 0.0, -self.k_ac_on * a, 0.0, self.k_ac_off],  # dc/dt
            [self.k_ab_on * b, self.k_ab_on * a, 0.0, -self.k_ab_off, 0.0],   # d(ab)/dt
            [self.k_ac_on * c, 0.0, self.k_ac_on * a, 0.0, -self.k_ac_off]    # d(ac)/dt
        ])

        return J

    def implicit_midpoint_residual(self, y_new: np.ndarray, y_old: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute residual for implicit midpoint rule.
        
        Implicit midpoint: y_{n+1} = y_n + dt * f((y_n + y_{n+1})/2)
        Residual: R = y_{n+1} - y_n - dt * f((y_n + y_{n+1})/2)
        
        :param y_new: New solution estimate [a, b, c, ab, ac]
        :param y_old: Previous solution [a, b, c, ab, ac]
        :param dt: Time step size
        :returns: Residual vector
        """
        y_mid = 0.5 * (y_old + y_new)
        f_mid = self.reaction_rates(y_mid)
        residual = y_new - y_old - dt * f_mid
        return residual

    def implicit_midpoint_jacobian(self, y_new: np.ndarray, y_old: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute Jacobian of the implicit midpoint residual with respect to y_new.
        
        ∂R/∂y_{n+1} = I - dt/2 * J((y_n + y_{n+1})/2)
        
        :param y_new: New solution estimate [a, b, c, ab, ac]
        :param y_old: Previous solution [a, b, c, ab, ac]
        :param dt: Time step size
        :returns: 5x5 Jacobian matrix of residual
        """
        y_mid = 0.5 * (y_old + y_new)
        J_mid = self.jacobian(y_mid)
        return np.eye(5) - 0.5 * dt * J_mid

    def newton_solve(self, y_old: np.ndarray, dt: float, y_guess: Optional[np.ndarray] = None,
                    max_iter: int = 20, tol: float = 1e-12) -> Tuple[np.ndarray, bool]:
        """
        Solve the implicit midpoint equation using Newton's method (legacy method).
        
        :param y_old: Previous solution [a, b, c, ab, ac]
        :param dt: Time step size
        :param y_guess: Initial guess for Newton iteration
        :param max_iter: Maximum number of Newton iterations
        :param tol: Convergence tolerance
        :returns: (solution, converged_flag)
        """
        if y_guess is None:
            # Use explicit Euler as initial guess
            y_guess = y_old + dt * self.reaction_rates(y_old)

        y_new = y_guess.copy()

        for _ in range(max_iter):
            residual = self.implicit_midpoint_residual(y_new, y_old, dt)

            # Check convergence
            if np.linalg.norm(residual) < tol:
                return y_new, True

            # Compute Jacobian and solve linear system
            J_res = self.implicit_midpoint_jacobian(y_new, y_old, dt)

            try:
                delta_y = np.linalg.solve(J_res, -residual)
                y_new += delta_y
            except np.linalg.LinAlgError:
                warnings.warn("Singular Jacobian in Newton iteration")
                return y_new, False

        warnings.warn(f"Newton method did not converge after {max_iter} iterations")
        return y_new, False

    def fixed_point_solve(self, y_old: np.ndarray, dt: float, y_guess: Optional[np.ndarray] = None,
                         max_iter: int = 50, tol: float = 1e-12, relaxation: float = 0.8) -> Tuple[np.ndarray, bool]:
        """
        Solve the implicit midpoint equation using fixed-point iteration with relaxation.
        
        Rearranges: y_new = y_old + dt * f((y_old + y_new)/2)
        Into fixed-point form and applies relaxation for better convergence.
        
        :param y_old: Previous solution [a, b, c, ab, ac]
        :param dt: Time step size
        :param y_guess: Initial guess for iteration
        :param max_iter: Maximum number of iterations
        :param tol: Convergence tolerance
        :param relaxation: Relaxation parameter (0 < relaxation <= 1)
        :returns: (solution, converged_flag)
        """
        if y_guess is None:
            # Use explicit Euler as initial guess
            y_guess = y_old + dt * self.reaction_rates(y_old)

        y_new = y_guess.copy()
        
        for iteration in range(max_iter):
            # Fixed point function: y = y_old + dt * f((y_old + y)/2)
            y_mid = 0.5 * (y_old + y_new)
            f_mid = self.reaction_rates(y_mid)
            y_next = y_old + dt * f_mid
            
            # Check convergence
            if np.linalg.norm(y_next - y_new) < tol:
                return y_next, True
            
            # Apply relaxation for better convergence
            y_new = (1 - relaxation) * y_new + relaxation * y_next

        warnings.warn(f"Fixed-point iteration did not converge after {max_iter} iterations")
        return y_new, False

    def fixed_point_solve_batch(self, y0_batch: np.ndarray, dt: float, 
                               max_iter: int = 50, tol: float = 1e-12,
                               relaxation: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve multiple chemical systems simultaneously using vectorized fixed-point iteration.
        
        This is optimized for finite element applications where the same chemical system
        needs to be solved at thousands of spatial points.
        
        :param y0_batch: Initial conditions shape (n_points, 5) where each row is [a, b, c, ab, ac]
        :param dt: Time step size (same for all points)
        :param max_iter: Maximum number of iterations
        :param tol: Convergence tolerance
        :param relaxation: Relaxation parameter (0.5-0.9 typically works well)
        :returns: (solutions, converged_flags) both shape (n_points, 5) and (n_points,)
        """
        n_points = y0_batch.shape[0]
        y_batch = y0_batch.copy()
        
        # Use explicit Euler as initial guess for all points
        y_batch = y0_batch + dt * self.reaction_rates_batch(y0_batch)
        
        # Track convergence for each point
        converged = np.zeros(n_points, dtype=bool)
        
        for iteration in range(max_iter):
            # Compute reaction rates for all points simultaneously
            y_mid_batch = 0.5 * (y0_batch + y_batch)
            f_mid_batch = self.reaction_rates_batch(y_mid_batch)
            
            # Fixed-point update: y_new = y_old + dt * f((y_old + y_new)/2)
            y_next_batch = y0_batch + dt * f_mid_batch
            
            # Check convergence for each point
            residuals = np.linalg.norm(y_next_batch - y_batch, axis=1)
            newly_converged = (residuals < tol) & (~converged)
            converged |= newly_converged
            
            # Early exit if all points converged
            if np.all(converged):
                return y_next_batch, converged
            
            # Apply relaxation only to non-converged points
            mask = ~converged
            y_batch[mask] = ((1 - relaxation) * y_batch[mask] + 
                            relaxation * y_next_batch[mask])
            
            # For converged points, keep the converged solution
            y_batch[converged] = y_next_batch[converged]
        
        return y_batch, converged

    def solve(self, y0: np.ndarray, t_span: Tuple[float, float], dt: float = 0.01,
              adaptive: bool = True, dt_min: float = 1e-8, dt_max: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the ODE system using implicit midpoint rule with Newton's method.
        
        :param y0: Initial conditions [a0, b0, c0, ab0, ac0]
        :param t_span: Time span (t_start, t_end)
        :param dt: Initial time step size
        :param adaptive: Whether to use adaptive time stepping
        :param dt_min: Minimum allowed time step
        :param dt_max: Maximum allowed time step
        :returns: (time_points, solution_matrix)
        """
        t_start, t_end = t_span

        # Initialize arrays
        t = t_start
        y = y0.copy()

        t_points = [t]
        y_points = [y.copy()]

        current_dt = dt

        while t < t_end:
            # Adjust time step if needed
            if t + current_dt > t_end:
                current_dt = t_end - t

            # Solve implicit midpoint equation
            y_new, converged = self.newton_solve(y, current_dt)

            if not converged and adaptive:
                # Reduce time step and retry
                current_dt *= 0.5
                if current_dt < dt_min:
                    raise RuntimeError(f"Time step became too small: {current_dt}")
                continue

            # Accept the step
            t += current_dt
            y = y_new.copy()

            t_points.append(t)
            y_points.append(y.copy())

            # Adaptive time stepping
            if adaptive and converged:
                # Simple adaptation: increase dt if Newton converged quickly
                current_dt = min(current_dt * 1.1, dt_max)

        return np.array(t_points), np.array(y_points)

    def plot_solution(self, t: np.ndarray, y: np.ndarray, title: str = "Chemical Species Evolution"):
        """
        Plot the solution showing evolution of all five species.
        
        :param t: Time points
        :param y: Solution matrix (n_points x 5)
        :param title: Plot title
        """
        plt.figure(figsize=(12, 8))

        plt.plot(t, y[:, 0], 'b-', label='Species A', linewidth=2)
        plt.plot(t, y[:, 1], 'r-', label='Species B', linewidth=2)
        plt.plot(t, y[:, 2], 'g-', label='Species C', linewidth=2)
        plt.plot(t, y[:, 3], 'm-', label='Complex AB', linewidth=2)
        plt.plot(t, y[:, 4], 'c-', label='Complex AC', linewidth=2)

        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def check_conservation(self, y: np.ndarray) -> np.ndarray:
        """
        Check mass conservation for each atomic species.
        
        Total A atoms: A + AB + AC (should be constant)
        Total B atoms: B + AB (should be constant)  
        Total C atoms: C + AC (should be constant)
        
        :param y: Solution matrix (n_points x 5)
        :returns: Conservation quantities [total_A, total_B, total_C] at each time point
        """
        total_A = y[:, 0] + y[:, 3] + y[:, 4]  # A + AB + AC
        total_B = y[:, 1] + y[:, 3]            # B + AB
        total_C = y[:, 2] + y[:, 4]            # C + AC
        return np.column_stack([total_A, total_B, total_C])


def example_usage():
    """
    Example demonstrating how to use the ChemicalReactionODE solver.
    """
    print("Example: 5-Species Chemical Reaction System")
    print("Reactions: A + B ⇌ AB, A + C ⇌ AC")
    print("="*50)

    # Define rate constants
    k_ab_on = 2.0    # A + B -> AB
    k_ab_off = 0.5   # AB -> A + B
    k_ac_on = 1.0    # A + C -> AC
    k_ac_off = 0.1   # AC -> A + C

    # Create solver instance
    solver = ChemicalReactionODE(k_ab_on, k_ab_off, k_ac_on, k_ac_off)

    # Initial conditions [A, B, C, AB, AC]
    y0 = np.array([1.0, 1.0, 1.0, 0.0, 0.0])  # Start with A, B, C, no complexes

    # Time span
    t_span = (0.0, 5.0)

    print(f"Initial conditions: A={y0[0]}, B={y0[1]}, C={y0[2]}, AB={y0[3]}, AC={y0[4]}")
    print(f"Rate constants: k_AB_on={k_ab_on}, k_AB_off={k_ab_off}")
    print(f"                k_AC_on={k_ac_on}, k_AC_off={k_ac_off}")
    print(f"Time span: {t_span}")

    # Solve the system
    print("\nSolving ODE system...")
    t, y = solver.solve(y0, t_span, dt=0.01, adaptive=True)

    # Check conservation
    conservation = solver.check_conservation(y)
    print("Mass conservation check:")
    print(f"  Total A variation: {np.max(conservation[:, 0]) - np.min(conservation[:, 0]):.2e}")
    print(f"  Total B variation: {np.max(conservation[:, 1]) - np.min(conservation[:, 1]):.2e}")
    print(f"  Total C variation: {np.max(conservation[:, 2]) - np.min(conservation[:, 2]):.2e}")

    # Print final concentrations
    print("\nFinal concentrations:")
    print(f"A  = {y[-1, 0]:.6f}")
    print(f"B  = {y[-1, 1]:.6f}")
    print(f"C  = {y[-1, 2]:.6f}")
    print(f"AB = {y[-1, 3]:.6f}")
    print(f"AC = {y[-1, 4]:.6f}")

    # Plot results
    solver.plot_solution(t, y, "5-Species Chemical Reaction (A+B ⇌ AB, A+C ⇌ AC)")

    return solver, t, y


def equilibrium_analysis(solver: ChemicalReactionODE, y0: np.ndarray):
    """
    Analyze equilibrium concentrations analytically.
    
    For the system A + B <-> AB, A + C <-> AC, at equilibrium:
    k_ab_on * A_eq * B_eq = k_ab_off * AB_eq
    k_ac_on * A_eq * C_eq = k_ac_off * AC_eq
    Conservation laws:
    A_eq + AB_eq + AC_eq = A_total
    B_eq + AB_eq = B_total
    C_eq + AC_eq = C_total
    
    :param solver: ChemicalReactionODE solver instance
    :param y0: Initial concentrations [a0, b0, c0, ab0, ac0]
    :returns: Equilibrium concentrations [A_eq, B_eq, C_eq, AB_eq, AC_eq]
    """
    # Total amounts (conserved quantities)
    A_total = y0[0] + y0[3] + y0[4]  # A + AB + AC
    B_total = y0[1] + y0[3]          # B + AB
    C_total = y0[2] + y0[4]          # C + AC
    
    print("\nEquilibrium Analysis:")
    print(f"Conservation laws:")
    print(f"  Total A atoms: {A_total:.6f}")
    print(f"  Total B atoms: {B_total:.6f}")
    print(f"  Total C atoms: {C_total:.6f}")
    
    # For complex formation, analytical solution is more complex
    # We'll use a simple approximation for demonstration
    # In practice, you might solve this numerically
    print("Note: Analytical equilibrium for complex formation requires solving")
    print("      a system of nonlinear equations. Running numerical solution...")
    
    # Run to equilibrium numerically
    t_eq, y_eq = solver.solve(y0, (0, 100), dt=0.01, adaptive=True)
    equilibrium = y_eq[-1, :]
    
    print(f"Numerical equilibrium concentrations:")
    print(f"  A_eq  = {equilibrium[0]:.6f}")
    print(f"  B_eq  = {equilibrium[1]:.6f}")
    print(f"  C_eq  = {equilibrium[2]:.6f}")
    print(f"  AB_eq = {equilibrium[3]:.6f}")
    print(f"  AC_eq = {equilibrium[4]:.6f}")
    
    # Verify equilibrium constants
    if equilibrium[0] > 1e-10 and equilibrium[1] > 1e-10 and equilibrium[2] > 1e-10:
        K_ab_calc = equilibrium[3] / (equilibrium[0] * equilibrium[1])
        K_ac_calc = equilibrium[4] / (equilibrium[0] * equilibrium[2])
        K_ab_expected = solver.k_ab_on / solver.k_ab_off
        K_ac_expected = solver.k_ac_on / solver.k_ac_off
        
        print(f"Equilibrium constant verification:")
        print(f"  K_AB calculated: {K_ab_calc:.6f}, expected: {K_ab_expected:.6f}")
        print(f"  K_AC calculated: {K_ac_calc:.6f}, expected: {K_ac_expected:.6f}")
    
    return equilibrium


if __name__ == "__main__":
    # Run example
    solver, t, y = example_usage()

    # Equilibrium analysis
    y0 = np.array([1.0, 1.0, 1.0, 0.0, 0.0])  # Initial conditions for analysis
    y_eq_theory = equilibrium_analysis(solver, y0)

    print("\nComparison with numerical equilibrium:")
    print(f"Final solution: [A={y[-1, 0]:.6f}, B={y[-1, 1]:.6f}, C={y[-1, 2]:.6f}, AB={y[-1, 3]:.6f}, AC={y[-1, 4]:.6f}]")
    print(f"Long-time eq.:  [A={y_eq_theory[0]:.6f}, B={y_eq_theory[1]:.6f}, C={y_eq_theory[2]:.6f}, AB={y_eq_theory[3]:.6f}, AC={y_eq_theory[4]:.6f}]")
