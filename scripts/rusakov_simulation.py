# %% [markdown]
# # Rusakov Simulation
# This script recreates the simulation from [Rusakov 2001]. Specifically, we
# will recreate the simulation of Ca-depletion for presynaptic, AP-driven
# calcium influx (Figure 4, top row).

# %%
import numpy as np
import matplotlib.pyplot as plt
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
TOTAL_SIZE = 2 * u.um        # Guessed
SYNAPSE_RADIUS = 0.1 * u.um  # Fig. 4
CLEFT_SIZE = 30 * u.nm       # Sec. "Ca2 diffusion in a calyx-type synapse"
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
fes_synapse = Compress(H1(mesh, order=1, definedon="synapse_ecs"))
fes_neuropil = Compress(H1(mesh, order=1, definedon="ecs", dirichlet="ecs_boundary"))
test_synapse, trial_synapse = fes_synapse.TnT()
test_neuropil, trial_neuropil = fes_neuropil.TnT()

D_synapse = convert(D_COEFFICIENT, DIFFUSIVITY)
a_synapse = BilinearForm(fes_synapse)
a_synapse += D_synapse * grad(test_synapse) * grad(trial_synapse) * dx
a_synapse.Assemble()

m_synapse = BilinearForm(fes_synapse)
m_synapse += test_synapse * trial_synapse * dx
m_synapse.Assemble()

D_neuropil = D_synapse / TORTUOSITY**2
a_neuropil = BilinearForm(fes_neuropil)
a_neuropil += D_neuropil * grad(test_neuropil) * grad(trial_neuropil) * dx
a_neuropil.Assemble()

m_neuropil = BilinearForm(fes_neuropil)
m_neuropil += test_neuropil * trial_neuropil * dx
m_neuropil.Assemble()

phi = convert(TIME_CONSTANT, 1 / TIME)
t_param = Parameter(0)
const_F = const.e.si * const.N_A
terminal_area = Integrate(1, mesh.Boundaries("presynaptic_membrane"), BND)
Q_0 = convert(CHANNEL_CURRENT / (2 * const_F), SUBSTANCE / TIME)
Q = N_CHANNELS * Q_0 * phi * t_param * exp(-phi * t_param) / terminal_area
b = LinearForm(fes_synapse)
b += D_synapse * Q * trial_synapse * ds("presynaptic_membrane")

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

mstar_synapse = m_synapse.mat.CreateMatrix()
mstar_synapse.AsVector().data = m_synapse.mat.AsVector() + tau * a_synapse.mat.AsVector()
inv_synapse = mstar_synapse.Inverse(freedofs=fes_synapse.FreeDofs())

mstar_neuropil = m_neuropil.mat.CreateMatrix()
mstar_neuropil.AsVector().data = m_neuropil.mat.AsVector() + tau * a_neuropil.mat.AsVector()
inv_neuropil = mstar_neuropil.Inverse(freedofs=fes_neuropil.FreeDofs())

ca_0 = convert(CA_RESTING, CONCENTRATION)
ca_synapse = GridFunction(fes_synapse)
ca_synapse.Set(ca_0)
ca_neuropil = GridFunction(fes_neuropil)
ca_neuropil.Set(ca_0)

# %%
# Run the simulation
evaluations = [np.concat((eval_synapse.evaluate(ca_synapse), eval_neuropil.evaluate(ca_neuropil)))]
time_points = [clock.current_time]
delta_synapse = ca_synapse.vec.CreateVector()
delta_neuropil = ca_neuropil.vec.CreateVector()
while clock.is_running():
    t_param.Set(clock.current_time)
    b.Assemble()
    delta_synapse.data = inv_synapse * (a_synapse.mat * ca_synapse.vec + b.vec)
    delta_neuropil.data = inv_neuropil * (a_neuropil.mat * ca_neuropil.vec)
    ca_synapse.vec.data += -tau * delta_synapse
    ca_neuropil.vec.data += -tau * delta_neuropil
    clock.advance()
    if clock.event_occurs("sampling"):
        values = np.concat((eval_synapse.evaluate(ca_synapse), eval_neuropil.evaluate(ca_neuropil)))
        evaluations.append(values)
        time_points.append(clock.current_time)
Draw(ca_synapse, clipping=clipping_settings, settings=visualization_settings)

# %%
# Plot the point values over time
evaluations = np.array(evaluations)
time_points = np.array(time_points)

plt.figure(figsize=(10, 6))
for i, values in enumerate(evaluations.T):
    plt.plot(time_points, values, label=f'Point {i+1}')

plt.xlabel('Time (ms)')
plt.ylabel('Calcium Concentration (mM)')
plt.title('Calcium Concentration Over Time at Different Points')
plt.legend()
plt.show()

# %%
