#!/usr/bin/env python3
"""
Performance comparison between fixed-point and Newton methods
for the 5-species chemical reaction system.
"""

import numpy as np
import time
from tmp import ChemicalReactionODE

def performance_comparison():
    """Compare performance of fixed-point vs Newton methods."""
    
    print("Performance Comparison: Fixed-Point vs Newton Methods")
    print("="*60)
    
    # Test parameters
    k_ab_on, k_ab_off = 2.0, 0.5
    k_ac_on, k_ac_off = 1.0, 0.1
    solver = ChemicalReactionODE(k_ab_on, k_ab_off, k_ac_on, k_ac_off)
    
    # Test cases with different initial conditions
    test_cases = [
        np.array([1.0, 1.0, 1.0, 0.0, 0.0]),      # Balanced initial conditions
        np.array([2.0, 0.5, 0.1, 0.0, 0.0]),      # Excess A
        np.array([0.1, 2.0, 2.0, 0.0, 0.0]),      # Excess B and C
        np.array([0.5, 0.5, 0.5, 0.2, 0.3]),      # Mixed initial conditions
    ]
    
    case_names = ["Balanced", "Excess A", "Excess B&C", "Mixed"]
    t_span = (0.0, 5.0)
    
    print(f"{'Case':<12} {'FP Time':<10} {'Newton Time':<12} {'Ratio':<8} {'Solution Diff':<15}")
    print("-" * 60)
    
    total_fp_time = 0
    total_newton_time = 0
    
    for i, (y0, name) in enumerate(zip(test_cases, case_names)):
        # Fixed-point method (default)
        start_time = time.time()
        t_fp, y_fp = solver.solve(y0, t_span, dt=0.01, adaptive=True)
        fp_time = time.time() - start_time
        
        # Newton method
        start_time = time.time()
        t_newton, y_newton = solver.solve_with_method(y0, t_span, dt=0.01, 
                                                     adaptive=True, method='newton')
        newton_time = time.time() - start_time
        
        # Compare solutions
        solution_diff = np.linalg.norm(y_fp[-1, :] - y_newton[-1, :])
        ratio = newton_time / fp_time if fp_time > 0 else float('inf')
        
        print(f"{name:<12} {fp_time:<10.4f} {newton_time:<12.4f} {ratio:<8.2f} {solution_diff:<15.2e}")
        
        total_fp_time += fp_time
        total_newton_time += newton_time
    
    print("-" * 60)
    total_ratio = total_newton_time / total_fp_time if total_fp_time > 0 else float('inf')
    print(f"{'TOTAL':<12} {total_fp_time:<10.4f} {total_newton_time:<12.4f} {total_ratio:<8.2f}")
    
    print(f"\nSummary:")
    print(f"- Fixed-point method uses adaptive relaxation and stagnation detection")
    print(f"- No matrix inversions required (more cache-friendly)")
    print(f"- Better suited for batch processing and FEM applications")
    if total_ratio < 1.0:
        print(f"- Fixed-point is {1/total_ratio:.1f}x faster than Newton")
    else:
        print(f"- Newton is {total_ratio:.1f}x faster than fixed-point")
    
    # Test stiff system behavior
    print(f"\nStiff System Test:")
    print("Testing with very different rate constants...")
    
    # Create a stiffer system
    stiff_solver = ChemicalReactionODE(k_ab_on=100.0, k_ab_off=0.001, 
                                      k_ac_on=50.0, k_ac_off=0.01)
    y0_stiff = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    
    try:
        start_time = time.time()
        t_fp_stiff, y_fp_stiff = stiff_solver.solve(y0_stiff, (0, 1.0), dt=0.001, adaptive=True)
        fp_stiff_time = time.time() - start_time
        print(f"Fixed-point stiff system: {fp_stiff_time:.4f} seconds, {len(t_fp_stiff)} steps")
    except Exception as e:
        print(f"Fixed-point stiff system failed: {e}")
    
    try:
        start_time = time.time()
        t_newton_stiff, y_newton_stiff = stiff_solver.solve_with_method(
            y0_stiff, (0, 1.0), dt=0.001, adaptive=True, method='newton')
        newton_stiff_time = time.time() - start_time
        print(f"Newton stiff system: {newton_stiff_time:.4f} seconds, {len(t_newton_stiff)} steps")
    except Exception as e:
        print(f"Newton stiff system failed: {e}")

def test_adaptive_behavior():
    """Test the adaptive behavior of fixed-point iteration."""
    
    print("\n" + "="*60)
    print("Testing Adaptive Fixed-Point Behavior")
    print("="*60)
    
    solver = ChemicalReactionODE(2.0, 0.5, 1.0, 0.1)
    y0 = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    
    print("Testing convergence with different tolerances:")
    tolerances = [1e-8, 1e-10, 1e-12, 1e-14]
    
    for tol in tolerances:
        start_time = time.time()
        # Modify the solver to use specific tolerance (temporary hack)
        original_solve = solver.newton_solve
        
        def custom_solve(y_old, dt, y_guess=None, max_iter=50, tol_param=tol):
            return original_solve(y_old, dt, y_guess, max_iter, tol_param)
        
        solver.newton_solve = custom_solve
        
        try:
            t, y = solver.solve(y0, (0, 2.0), dt=0.01, adaptive=True)
            solve_time = time.time() - start_time
            conservation = solver.check_conservation(y)
            max_conservation_error = np.max([
                np.max(conservation[:, 0]) - np.min(conservation[:, 0]),
                np.max(conservation[:, 1]) - np.min(conservation[:, 1]),
                np.max(conservation[:, 2]) - np.min(conservation[:, 2])
            ])
            print(f"Tolerance {tol:.0e}: {solve_time:.4f}s, {len(t)} steps, conservation error: {max_conservation_error:.2e}")
        except Exception as e:
            print(f"Tolerance {tol:.0e}: Failed - {e}")
        finally:
            solver.newton_solve = original_solve

if __name__ == "__main__":
    performance_comparison()
    test_adaptive_behavior()
