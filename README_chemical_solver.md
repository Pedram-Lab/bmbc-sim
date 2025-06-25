# Stiff ODE Solver for 5-Species Chemical Complex Formation System

## Overview

This implementation solves a stiff ODE system modeling 5 interacting chemical species (A, B, C, AB, AC) with the following complex formation reactions:

- A + B ⇌ AB (with rate constants k_ab_on, k_ab_off)
- A + C ⇌ AC (with rate constants k_ac_on, k_ac_off)

## Mathematical Model

The system of ODEs is:
```
da/dt = -k_ab_on * a * b + k_ab_off * ab - k_ac_on * a * c + k_ac_off * ac
db/dt = -k_ab_on * a * b + k_ab_off * ab
dc/dt = -k_ac_on * a * c + k_ac_off * ac
d(ab)/dt = k_ab_on * a * b - k_ab_off * ab
d(ac)/dt = k_ac_on * a * c - k_ac_off * ac
```

## Conservation Laws

The system conserves the total number of each atomic species:
- Total A atoms: A + AB + AC = constant
- Total B atoms: B + AB = constant  
- Total C atoms: C + AC = constant

## Numerical Method

### Implicit Midpoint Rule
The implicit midpoint rule is used for time stepping:
```
y_{n+1} = y_n + dt * f((y_n + y_{n+1})/2)
```

This method is:
- **A-stable**: Suitable for stiff systems
- **Second-order accurate**: Better than implicit Euler
- **Symplectic**: Preserves conservation laws well

### Newton's Method
The nonlinear system arising from the implicit midpoint rule is solved using Newton's method:
```
J(y^k) * Δy = -R(y^k)
y^{k+1} = y^k + Δy
```

Where:
- R(y) is the residual: `y_{n+1} - y_n - dt * f((y_n + y_{n+1})/2)`
- J(y) is the Jacobian: `∂R/∂y_{n+1} = I - dt/2 * ∂f/∂y`

## Key Features

1. **Customizable Parameters**: Easy to set initial conditions and rate constants
2. **Adaptive Time Stepping**: Automatically adjusts step size for efficiency
3. **Mass Conservation**: Monitors and preserves total mass
4. **Analytical Verification**: Compares with analytical equilibrium solution
5. **Robust Error Handling**: Handles convergence failures gracefully

## Usage Example

```python
# Create solver with rate constants
solver = ChemicalReactionODE(k_ab_on=2.0, k_ab_off=0.5, 
                           k_ac_on=1.0, k_ac_off=0.1)

# Set initial conditions [A, B, C, AB, AC]
y0 = np.array([1.0, 1.0, 1.0, 0.0, 0.0])  # Start with A, B, C, no complexes

# Solve the system
t, y = solver.solve(y0, t_span=(0, 5), dt=0.01, adaptive=True)

# Plot results
solver.plot_solution(t, y)
```

## Equilibrium Analysis

At equilibrium, the system satisfies:
```
k_ab_on * A_eq * B_eq = k_ab_off * AB_eq  =>  K_AB = AB_eq / (A_eq * B_eq) = k_ab_on / k_ab_off
k_ac_on * A_eq * C_eq = k_ac_off * AC_eq  =>  K_AC = AC_eq / (A_eq * C_eq) = k_ac_on / k_ac_off
```

Combined with the conservation laws, this forms a system of nonlinear equations that can be solved numerically.

## Validation

The implementation has been validated by:
1. **Conservation laws**: Total atomic species are preserved to machine precision
2. **Equilibrium convergence**: Numerical solution converges to equilibrium satisfying mass action laws
3. **Equilibrium constants**: Calculated equilibrium constants match expected values (K_AB, K_AC)
4. **Stiffness handling**: Stable solution even for large rate constant ratios

## Performance Characteristics

- **Stability**: Unconditionally stable for any step size
- **Accuracy**: Second-order accurate in time
- **Efficiency**: Adaptive time stepping minimizes computational cost
- **Robustness**: Handles stiff problems that would cause explicit methods to fail

The solver is particularly well-suited for chemical complex formation systems where:
- Formation and dissociation rates span multiple orders of magnitude (stiff systems)
- High accuracy is required
- Conservation of atomic species is critical
- Long-time behavior and equilibrium properties need to be captured accurately
- Nonlinear reaction kinetics (bimolecular reactions) are involved
