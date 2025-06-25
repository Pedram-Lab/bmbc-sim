import warnings
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

class ChemicalReactionODE:
    """
    Solver for stiff ODE system modeling 5 interacting chemical species (a, b, c, ab, ac)
    with complex formation reactions: a + b <-> ab and a + c <-> ac
    
    Uses implicit midpoint rule for time stepping and adaptive fixed-point iteration for 
    nonlinear system solution. The fixed-point method includes:
    - Adaptive relaxation based on convergence behavior
    - Early stagnation detection and recovery
    - Optimized time step adaptation
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
                    max_iter: int = 50, tol: float = 1e-12) -> Tuple[np.ndarray, bool]:
        """
        Solve the implicit midpoint equation using adaptive fixed-point iteration.
        
        This method replaces Newton's method with fixed-point iteration using:
        - Adaptive relaxation based on convergence behavior
        - Early convergence detection
        - Optimized initial guess strategies
        
        :param y_old: Previous solution [a, b, c, ab, ac]
        :param dt: Time step size
        :param y_guess: Initial guess for iteration
        :param max_iter: Maximum number of iterations
        :param tol: Convergence tolerance
        :returns: (solution, converged_flag)
        """
        if y_guess is None:
            # Use explicit Euler as initial guess
            y_guess = y_old + dt * self.reaction_rates(y_old)

        y_new = y_guess.copy()
        
        # Adaptive relaxation parameters
        relaxation = 0.8  # Start with moderate relaxation
        convergence_history = []
        
        for iteration in range(max_iter):
            # Fixed point function: y = y_old + dt * f((y_old + y)/2)
            y_mid = 0.5 * (y_old + y_new)
            f_mid = self.reaction_rates(y_mid)
            y_next = y_old + dt * f_mid
            
            # Compute residual for convergence check
            residual_norm = np.linalg.norm(y_next - y_new)
            convergence_history.append(residual_norm)
            
            # Check convergence
            if residual_norm < tol:
                return y_next, True
            
            # Adaptive relaxation based on convergence behavior
            relaxation = self._adapt_relaxation(convergence_history, relaxation, iteration)
            
            # Apply relaxation for better convergence
            y_new = (1 - relaxation) * y_new + relaxation * y_next
            
            # Early termination if convergence stagnates
            if iteration > 10 and self._check_stagnation(convergence_history):
                # Try different relaxation strategy
                if relaxation > 0.3:
                    relaxation *= 0.5
                    continue
                else:
                    warnings.warn("Fixed-point iteration stagnated")
                    return y_new, False

        warnings.warn(f"Fixed-point iteration did not converge after {max_iter} iterations")
        return y_new, False
    
    def _adapt_relaxation(self, convergence_history: list, current_relaxation: float, iteration: int) -> float:
        """
        Adapt relaxation parameter based on convergence behavior.
        
        :param convergence_history: List of residual norms
        :param current_relaxation: Current relaxation parameter
        :param iteration: Current iteration number
        :returns: Adapted relaxation parameter
        """
        if iteration < 3:
            return current_relaxation
        
        # Analyze recent convergence trend
        recent_residuals = convergence_history[-3:]
        
        # Check if residuals are decreasing (good convergence)
        if len(recent_residuals) >= 2:
            trend = recent_residuals[-1] / recent_residuals[-2]
            
            if trend < 0.1:  # Very fast convergence
                return min(current_relaxation * 1.2, 0.95)
            elif trend < 0.7:  # Good convergence
                return min(current_relaxation * 1.05, 0.9)
            elif trend > 1.2:  # Diverging
                return max(current_relaxation * 0.7, 0.3)
            elif trend > 0.95:  # Slow convergence
                return max(current_relaxation * 0.9, 0.5)
        
        return current_relaxation
    
    def _check_stagnation(self, convergence_history: list, window: int = 5) -> bool:
        """
        Check if convergence has stagnated.
        
        :param convergence_history: List of residual norms
        :param window: Window size for stagnation detection
        :returns: True if stagnation detected
        """
        if len(convergence_history) < window:
            return False
        
        recent = convergence_history[-window:]
        # Check if relative improvement is very small
        if recent[0] > 0:
            relative_improvement = (recent[0] - recent[-1]) / recent[0]
            return relative_improvement < 0.01  # Less than 1% improvement
        
        return False

    def solve(self, y0: np.ndarray, t_span: Tuple[float, float], dt: float = 0.01,
              adaptive: bool = True, dt_min: float = 1e-8, dt_max: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the ODE system using implicit midpoint rule with adaptive fixed-point iteration.
        
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
        
        # Track performance for adaptive strategies
        convergence_history = []
        failed_steps = 0
        
        while t < t_end:
            # Adjust time step if needed
            if t + current_dt > t_end:
                current_dt = t_end - t

            # Solve implicit midpoint equation using fixed-point iteration
            y_new, converged = self.newton_solve(y, current_dt)
            
            if converged:
                # Accept the step
                t += current_dt
                y = y_new.copy()

                t_points.append(t)
                y_points.append(y.copy())
                
                convergence_history.append(True)
                failed_steps = 0
                
                # Adaptive time stepping based on convergence performance
                if adaptive:
                    current_dt = self._adapt_time_step(current_dt, convergence_history, 
                                                     dt_min, dt_max, failed_steps)
            else:
                # Step failed - reduce time step and retry
                if adaptive:
                    convergence_history.append(False)
                    failed_steps += 1
                    
                    current_dt *= 0.5
                    if current_dt < dt_min:
                        raise RuntimeError(f"Time step became too small: {current_dt}")
                else:
                    # Without adaptivity, accept the solution anyway with a warning
                    warnings.warn("Fixed-point iteration did not converge, accepting solution")
                    t += current_dt
                    y = y_new.copy()
                    t_points.append(t)
                    y_points.append(y.copy())

        return np.array(t_points), np.array(y_points)
    
    def _adapt_time_step(self, current_dt: float, convergence_history: list, 
                        dt_min: float, dt_max: float, failed_steps: int) -> float:
        """
        Adapt time step based on convergence performance.
        
        :param current_dt: Current time step
        :param convergence_history: Recent convergence history
        :param dt_min: Minimum allowed time step
        :param dt_max: Maximum allowed time step
        :param failed_steps: Number of recent failed steps
        :returns: Adapted time step
        """
        if len(convergence_history) < 5:
            return current_dt
        
        # Analyze recent performance
        recent_performance = convergence_history[-5:]
        success_rate = sum(recent_performance) / len(recent_performance)
        
        # Adjust time step based on success rate
        if success_rate >= 0.9 and failed_steps == 0:
            # High success rate - can increase time step
            return min(current_dt * 1.1, dt_max)
        elif success_rate >= 0.7:
            # Moderate success rate - keep current time step
            return current_dt
        elif success_rate >= 0.5:
            # Lower success rate - slightly reduce time step
            return max(current_dt * 0.9, dt_min)
        else:
            # Poor success rate - significantly reduce time step
            return max(current_dt * 0.7, dt_min)

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

    def newton_solve_legacy(self, y_old: np.ndarray, dt: float, y_guess: Optional[np.ndarray] = None,
                           max_iter: int = 20, tol: float = 1e-12) -> Tuple[np.ndarray, bool]:
        """
        Legacy Newton's method implementation for comparison.
        
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

    def solve_with_method(self, y0: np.ndarray, t_span: Tuple[float, float], dt: float = 0.01,
                         adaptive: bool = True, dt_min: float = 1e-8, dt_max: float = 0.1,
                         method: str = 'fixed_point') -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the ODE system with specified method.
        
        :param y0: Initial conditions [a0, b0, c0, ab0, ac0]
        :param t_span: Time span (t_start, t_end)
        :param dt: Initial time step size
        :param adaptive: Whether to use adaptive time stepping
        :param dt_min: Minimum allowed time step
        :param dt_max: Maximum allowed time step
        :param method: Solver method ('fixed_point' or 'newton')
        :returns: (time_points, solution_matrix)
        """
        # Temporarily switch solver method
        original_solver = self.newton_solve
        
        if method == 'newton':
            self.newton_solve = self.newton_solve_legacy
        
        try:
            result = self.solve(y0, t_span, dt, adaptive, dt_min, dt_max)
        finally:
            # Restore original solver
            self.newton_solve = original_solver
        
        return result


def example_usage():
    """
    Example demonstrating how to use the ChemicalReactionODE solver with fixed-point iteration.
    """
    print("Example: 5-Species Chemical Reaction System with Fixed-Point Iteration")
    print("Reactions: A + B ⇌ AB, A + C ⇌ AC")
    print("="*70)

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

    # Solve the system using fixed-point iteration
    print("\nSolving ODE system with adaptive fixed-point iteration...")
    import time
    start_time = time.time()
    t, y = solver.solve(y0, t_span, dt=0.01, adaptive=True)
    fp_time = time.time() - start_time
    
    print(f"Fixed-point solution time: {fp_time:.4f} seconds")
    print(f"Number of time steps: {len(t)}")

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

    # Compare with Newton's method
    print("\nComparing with legacy Newton's method...")
    start_time = time.time()
    t_newton, y_newton = solver.solve_with_method(y0, t_span, dt=0.01, adaptive=True, method='newton')
    newton_time = time.time() - start_time
    
    print(f"Newton solution time: {newton_time:.4f} seconds")
    print(f"Number of time steps: {len(t_newton)}")
    print(f"Performance ratio (Newton/Fixed-point): {newton_time/fp_time:.2f}")
    
    # Compare final solutions
    solution_diff = np.linalg.norm(y[-1, :] - y_newton[-1, :])
    print(f"Solution difference (L2 norm): {solution_diff:.2e}")

    # Plot results
    solver.plot_solution(t, y, "5-Species Chemical Reaction (Fixed-Point Method)")

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
