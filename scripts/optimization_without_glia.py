# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import astropy.units as u
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from ecsim.geometry import create_rusakov_geometry, create_mesh
from ecsim.simulation import SimulationClock
from ngsolve import H1, BilinearForm, LinearForm, grad, GridFunction, Mesh, Parameter, dx, ds

def run_simulation(params):
    """
    Runs the FEM simulation in NGSolve with the given parameters.
    Returns the calcium concentrations at the evaluation points.
    """
    # Extract parameters and assign units
    TOTAL_SIZE = params[0] * u.um
    SYNAPSE_RADIUS = params[1] * u.um
    CLEFT_SIZE = params[2] * u.um

    # Fixed at 0
    GLIA_DISTANCE = 0 * u.um
    GLIA_WIDTH = 0 * u.um
    GLIA_COVERAGE = 0  # Remains 0

    angle = float(np.arccos(1 - 2 * GLIA_COVERAGE)) * u.rad  # Always 0 rad

    # Create geometry and mesh
    geo = create_rusakov_geometry(
        total_size=TOTAL_SIZE,
        synapse_radius=SYNAPSE_RADIUS,
        cleft_size=CLEFT_SIZE,
        glia_distance=GLIA_DISTANCE,
        glia_width=GLIA_WIDTH,
        glial_coverage_angle=angle
    )
    mesh_size = 0.1 * u.um  # Convert mesh size to micrometers
    mesh = Mesh(create_mesh(geo, mesh_size=mesh_size))

    # Define FEM
    fes = H1(mesh, order=1, definedon="ecs", dirichlet="ecs_boundary")
    v_test, v_trial = fes.TnT()

    D = 0.4  # Diffusion coefficient
    a = BilinearForm(fes)
    a += D * grad(v_test) * grad(v_trial) * dx
    a.Assemble()

    m = BilinearForm(fes)
    m += v_test * v_trial * dx
    m.Assemble()

    # Initial conditions
    ca = GridFunction(fes)
    ca.Set(1.3)  # Initial concentration

    # Time simulation
    tau = 1e-3  # Time step
    clock = SimulationClock(time_step=tau, end_time=1.5, events={"sampling": 10})
    evaluations = []

    while clock.is_running():
        clock.advance()
        evaluations.append(ca.vec.FV().NumPy())

    return np.array(evaluations)

def match_dimensions(simulated_data, experimental_data):
    """
    Interpolate simulated_data so that it has the same length as experimental_data.
    """
    sim_time = np.linspace(0, 1, simulated_data.shape[0])  # Normalized simulation time
    exp_time = np.linspace(0, 1, len(experimental_data))   # Normalized experimental time

    interpolator = interp1d(sim_time, simulated_data.mean(axis=1), kind='linear', fill_value="extrapolate")
    return interpolator(exp_time)

def objective_function(params, experimental_data):
    """
    Cost function: measures the difference between the simulated and experimental data.
    """
    simulated_data = run_simulation(params)
    interpolated_data = match_dimensions(simulated_data, experimental_data)
    error = np.linalg.norm(interpolated_data - experimental_data)
    return error    

# Define the CSV file paths
file_paths = {
    "Point 1": "/Users/perezrosasn/Documents/GitHub/ecm-simulations/scripts/rusakov_data/fig4a_point1_rusakov_2001.csv",
    "Point 2": "/Users/perezrosasn/Documents/GitHub/ecm-simulations/scripts/rusakov_data/fig4a_point2_rusakov_2001.csv",
    "Point 3": "/Users/perezrosasn/Documents/GitHub/ecm-simulations/scripts/rusakov_data/fig4a_point3_rusakov_2001.csv",
    "Point 4-5": "/Users/perezrosasn/Documents/GitHub/ecm-simulations/scripts/rusakov_data/fig4a_point4_5_rusakov_2001.csv",
}

# Read the CSV files into a dictionary of DataFrames
data = {point: pd.read_csv(path) for point, path in file_paths.items()}

# Find the minimum length among all files
min_length = min(len(df) for df in data.values())

# Crop all series to the same size
experimental_data = np.mean(
    [df.iloc[:min_length, 1].values for df in data.values()],
    axis=0
)

# Initial parameter range without GLIA_DISTANCE, GLIA_WIDTH, and GLIA_COVERAGE
initial_params = [2.0, 0.1, 30e-3]
bounds = [(1.5, 2.5), (0.05, 0.2), (20e-3, 40e-3)]

# Numerical optimization
result = minimize(objective_function, initial_params, args=(experimental_data,), bounds=bounds, method='L-BFGS-B')
optimized_params = result.x

print("Optimal parameters:")
print(f"  1. TOTAL_SIZE       = {optimized_params[0]} um")
print(f"  2. SYNAPSE_RADIUS   = {optimized_params[1]} um")
print(f"  3. CLEFT_SIZE       = {optimized_params[2]} um")
