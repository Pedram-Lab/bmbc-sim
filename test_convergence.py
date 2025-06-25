import numpy as np
from tmp import ChemicalReactionODE, equilibrium_analysis

def test_long_time_convergence():
    """Test that the solution converges to the numerical equilibrium."""
    
    # Same parameters as the example
    k_ab_on, k_ab_off = 2.0, 0.5
    k_ac_on, k_ac_off = 1.0, 0.1
    
    solver = ChemicalReactionODE(k_ab_on, k_ab_off, k_ac_on, k_ac_off)
    y0 = np.array([1.0, 1.0, 1.0, 0.0, 0.0])  # [A, B, C, AB, AC]
    
    # Run for moderate time  
    t_span = (0.0, 20.0)
    t, y = solver.solve(y0, t_span, dt=0.01, adaptive=True)
    
    # Get numerical equilibrium (run longer)
    y_eq_numerical = equilibrium_analysis(solver, y0)
    
    # Check convergence
    final_solution = y[-1, :]
    error = np.abs(final_solution - y_eq_numerical)
    # Avoid division by zero for small concentrations
    relative_error = np.where(y_eq_numerical > 1e-8, error / y_eq_numerical, error)
    
    print("Long-time convergence test:")
    print(f"Time: {t[-1]:.1f}")
    print(f"Final solution:  [A={final_solution[0]:.6f}, B={final_solution[1]:.6f}, C={final_solution[2]:.6f}, AB={final_solution[3]:.6f}, AC={final_solution[4]:.6f}]")
    print(f"Equilibrium:     [A={y_eq_numerical[0]:.6f}, B={y_eq_numerical[1]:.6f}, C={y_eq_numerical[2]:.6f}, AB={y_eq_numerical[3]:.6f}, AC={y_eq_numerical[4]:.6f}]")
    print(f"Absolute error:  [{error[0]:.2e}, {error[1]:.2e}, {error[2]:.2e}, {error[3]:.2e}, {error[4]:.2e}]")
    print(f"Relative error (%): [{relative_error[0]*100:.3f}, {relative_error[1]*100:.3f}, {relative_error[2]*100:.3f}, {relative_error[3]*100:.3f}, {relative_error[4]*100:.3f}]")
    
    # Test passes if relative error is < 5% (more lenient for complex system)
    assert np.all(relative_error < 0.05), "Solution did not converge to equilibrium within 5%"
    print("✓ Test passed: Solution converged to numerical equilibrium!")

def test_conservation():
    """Test that conservation laws are maintained."""
    
    solver = ChemicalReactionODE(2.0, 0.5, 1.0, 0.1)
    y0 = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    
    t, y = solver.solve(y0, (0, 10), dt=0.01, adaptive=True)
    
    conservation = solver.check_conservation(y)
    
    # Check that conservation quantities don't change much
    A_variation = np.max(conservation[:, 0]) - np.min(conservation[:, 0])
    B_variation = np.max(conservation[:, 1]) - np.min(conservation[:, 1])
    C_variation = np.max(conservation[:, 2]) - np.min(conservation[:, 2])
    
    print(f"\nConservation test:")
    print(f"Total A variation: {A_variation:.2e}")
    print(f"Total B variation: {B_variation:.2e}")
    print(f"Total C variation: {C_variation:.2e}")
    
    assert A_variation < 1e-10, f"A conservation violated: {A_variation}"
    assert B_variation < 1e-10, f"B conservation violated: {B_variation}"
    assert C_variation < 1e-10, f"C conservation violated: {C_variation}"
    print("✓ Conservation test passed!")

if __name__ == "__main__":
    test_long_time_convergence()
    test_conservation()
