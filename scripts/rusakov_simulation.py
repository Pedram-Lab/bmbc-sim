# %% [markdown]
# # Rusakov Simulation
# This script recreates the simulation from [Rusakov 2001]. Specifically, we
# will recreate the simulation of Ca-depletion for presynaptic, AP-driven
# calcium influx (Figure 4, top row).

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
import astropy.constants as const
from ngsolve.webgui import Draw
from ngsolve import (H1, BilinearForm, LinearForm, grad, GridFunction, Mesh, Parameter,
                     Integrate, BND, exp, dx, ds, Compress)

from ecsim.geometry import create_rusakov_geometry, create_mesh, PointEvaluator
from ecsim.units import DIFFUSIVITY, CONCENTRATION, SUBSTANCE, TIME, LENGTH, convert
from ecsim.simulation import SimulationClock

# %%
clipping_settings = {"function": True,  "pnt": (0, 0, 0), "vec": (0, 1, 0)}
visualization_settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}}

# %%
# Geometry parameters
TOTAL_SIZE = 2 * u.um        # Guessed/probably 2 - 20 um 
SYNAPSE_RADIUS = 0.1 * u.um  # Fig. 4 / 0.11 um
CLEFT_SIZE = 30 * u.nm       # Sec. "Ca2 diffusion in a calyx-type synapse" (30 nm -200 nm)
GLIA_DISTANCE = 30 * u.nm    # Guessed
GLIA_WIDTH = 100 * u.nm      # Sec. "Glial sheath and glutamate transporter density"
GLIA_COVERAGE = 0.5          # Varied
TORTUOSITY = 1.4             # Sec. "Synaptic geometry"
POROSITY = 0.12              # Sec. "Synaptic geometry"

# Ca parameters
CA_RESTING = 1.3 * u.mmol / u.L       # Sec. "Presynaptic calcium influx"
CHANNEL_CURRENT = 0.5 * u.pA          # Sec. "Presynaptic calcium influx"
D_COEFFICIENT = 0.4 * u.um**2 / u.ms  # Fig. 4
TIME_CONSTANT = 10 / u.ms             # Sec. "Presynaptic calcium influx"
N_CHANNELS = 39                       # Fig. 4

# Simulation parameters
MESH_SIZE = 0.1 * u.um
TIME_STEP = 1.0 * u.us
END_TIME = 1.5 * u.ms

# %%

# %%
# Create the geometry
angle = float(np.arccos(1 - 2 * GLIA_COVERAGE)) * u.rad
geo = create_rusakov_geometry(
    total_size=TOTAL_SIZE,
    synapse_radius=SYNAPSE_RADIUS,
    cleft_size=CLEFT_SIZE,
    glia_distance=GLIA_DISTANCE,
    glia_width=GLIA_WIDTH,
    glial_coverage_angle=angle,
)
Draw(geo, clipping=clipping_settings, settings=visualization_settings)

# %%
# Create the mesh
mesh = Mesh(create_mesh(geo, mesh_size=MESH_SIZE))

# %%
# Set up FEM objects
# see https://docu.ngsolve.org/latest/i-tutorials/unit-2.13-interfaces/interfaceresistivity.html
fes_synapse = Compress(H1(mesh, order=1, definedon="synapse_ecs"))
fes_neuropil = Compress(H1(mesh, order=1, definedon="neuropil", dirichlet="neuropil_boundary"))
fes = fes_synapse * fes_neuropil
(test_s, test_n), (trial_s, trial_n) = fes.TnT()

D_synapse = convert(D_COEFFICIENT, DIFFUSIVITY)
D_neuropil = convert(D_COEFFICIENT, DIFFUSIVITY) / TORTUOSITY**2
a = BilinearForm(fes)
a += D_synapse * grad(test_s) * grad(trial_s) * dx("synapse_ecs")
a += D_neuropil * grad(test_n) * grad(trial_n) * dx("neuropil")
a += POROSITY * (test_s - test_n) * (trial_s - trial_n) * ds("synapse_boundary")
a.Assemble()

m = BilinearForm(fes)
m += test_s * trial_s * dx("synapse_ecs")
m += test_n * trial_n * dx("neuropil")
m.Assemble()

phi = convert(TIME_CONSTANT, 1 / TIME)
t_param = Parameter(0)
const_F = const.e.si * const.N_A
terminal_area = Integrate(1, mesh.Boundaries("presynaptic_membrane"), BND)
Q_0 = convert(CHANNEL_CURRENT / (2 * const_F), SUBSTANCE / TIME)
Q = N_CHANNELS * Q_0 * phi * t_param * exp(-phi * t_param) / terminal_area
b = LinearForm(fes)
b += D_synapse * Q * trial_s * ds("presynaptic_membrane")

# %%
# Initialize the simulation
t_end = convert(END_TIME, TIME)
tau = convert(TIME_STEP, TIME)
events = {"sampling": 10}
clock = SimulationClock(time_step=tau, end_time=t_end, events=events, verbose=True)

dist = convert(SYNAPSE_RADIUS + GLIA_DISTANCE / 2, LENGTH)
eval_points = np.array([
    [0, 0, 0],          # 1: center
    [dist, 0, 0],       # 2: inside glia, near cleft
    [0, 0, dist],       # 3: inside glia, far from cleft
])
eval_synapse = PointEvaluator(mesh, eval_points)
dist = convert(SYNAPSE_RADIUS + GLIA_WIDTH + 2 * GLIA_DISTANCE, LENGTH)
eval_points = np.array([
    [0, 0, - dist],  # 4: outside glia (below)
    [0, 0, dist],   # 5: outside glia (above)
])
eval_neuropil = PointEvaluator(mesh, eval_points)

mstar = m.mat.CreateMatrix()
mstar.AsVector().data = m.mat.AsVector() + tau * a.mat.AsVector()
mstar_inv = mstar.Inverse(freedofs=fes.FreeDofs())

ca_0 = convert(CA_RESTING, CONCENTRATION)
ca = GridFunction(fes)
ca.components[0].Set(ca_0)
ca.components[1].Set(ca_0)

# %%
# Run the simulation
evaluations = [np.concat((eval_synapse.evaluate(ca.components[0]),
                          eval_neuropil.evaluate(ca.components[1])))]
time_points = [clock.current_time]
delta = ca.vec.CreateVector()
while clock.is_running():
    t_param.Set(clock.current_time)
    b.Assemble()
    delta.data = mstar_inv * (a.mat * ca.vec + b.vec)
    ca.vec.data += -tau * delta
    clock.advance()
    if clock.event_occurs("sampling"):
        values = np.concat((eval_synapse.evaluate(ca.components[0]),
                            eval_neuropil.evaluate(ca.components[1])))
        evaluations.append(values)
        time_points.append(clock.current_time)
Draw(ca.components[0], clipping=clipping_settings, settings=visualization_settings)

# %%
# Plot the point values over time
evaluations = np.array(evaluations)
time_points = np.array(time_points)

plt.figure(figsize=(10, 6))
for i, values in enumerate(evaluations.T):
    plt.plot(time_points, values, label=f'Point {i+1}')

plt.xlabel('Time (ms)')
plt.ylabel('Calcium Concentration (mM)')
plt.xlim(0, convert(END_TIME, TIME))
plt.ylim(0.4, 1.4)
plt.title('Calcium Concentration Over Time at Different Points')
plt.legend()
plt.show()


# Extract the minima from each time series
min_values = np.min(evaluations, axis=0)
min_time_indices = np.argmin(evaluations, axis=0)
min_time_points = time_points[min_time_indices]

# Plot the minimum values
plt.figure(figsize=(10, 6))
for i, (t, v) in enumerate(zip(min_time_points, min_values)):
    plt.plot(t, v, 'o', label=f'Point {i+1}')
    plt.text(t, v, f'Point {i+1}', fontsize=10, verticalalignment='bottom', horizontalalignment='right')

plt.xlabel('Time (ms)')
plt.ylabel('Minimum Calcium Concentration (mM)')
plt.title('Minimum Calcium Concentration Over Time')
plt.legend()
plt.show()

# %%
# Create a list of minimum points
min_points_list = [(f'Point {i+1}', t, v) for i, (t, v) in enumerate(zip(min_time_points, min_values))]
print("List of minimum points:")
print(min_points_list)

# Extract the minima from each time series
min_values = np.min(evaluations, axis=0)
min_time_indices = np.argmin(evaluations, axis=0)
min_time_points = time_points[min_time_indices]

# Plot the minimum values
plt.figure(figsize=(10, 6))
plt.plot(min_time_points, min_values, 'o-', label='Minimum Values')

plt.xlabel('Time (ms)')
plt.ylabel('Minimum Calcium Concentration (mM)')
plt.title('Minimum Calcium Concentration Over Time')
plt.legend()
plt.show()

# %%
# Convert data to a DataFrame and save to CSV
evaluations = np.array(evaluations)
time_points = np.array(time_points)
data = {'Time (ms)': time_points}
for i in range(evaluations.shape[1]):
    data[f'Point {i+1} (mM)'] = evaluations[:, i]
df = pd.DataFrame(data)
df.to_csv("calcium_total_size_2_um_synapse_radius_0p1_cleft_size_0p03.csv", index=False)

# Plot the point values over time
plt.figure(figsize=(10, 6))
for i, point in enumerate(eval_points):
    plt.plot(time_points, evaluations[:, i], label=f'Point {i+1}')

plt.xlabel('Time (ms)')
plt.ylabel('Calcium Concentration (mM)')
plt.title('Calcium Concentration Over Time at Different Points')
plt.legend()
plt.show()

print("Data saved in 'calcium_cleft_size_10nm.csv'")