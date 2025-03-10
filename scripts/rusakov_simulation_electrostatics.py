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
import ngsolve as ngs

from ecsim.geometry import create_rusakov_geometry, create_mesh, PointEvaluator
from ecsim.units import to_simulation_units
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
GLIA_WIDTH = 50 * u.nm       # Sec. "Glial sheath and glutamate transporter density"
GLIA_COVERAGE = 0.5          # Varied
TORTUOSITY = 1.4             # Sec. "Synaptic geometry"
POROSITY = 0.12              # Sec. "Synaptic geometry"

# Ca parameters
CA_RESTING = 1.3 * u.mmol / u.L       # Sec. "Presynaptic calcium influx"
CHANNEL_CURRENT = 0.5 * u.pA          # Sec. "Presynaptic calcium influx"
D_COEFFICIENT = 0.4 * u.um**2 / u.ms  # Fig. 4
TIME_CONSTANT = 10 / u.ms             # Sec. "Presynaptic calcium influx"
N_CHANNELS = 39                       # Fig. 4

# Electrostatics parameters
RELATIVE_PERMITTIVITY = 80.0
permittivity = to_simulation_units(RELATIVE_PERMITTIVITY * const.eps0, 'permittivity')
F = to_simulation_units(96485.3365 * u.C / u.mol)
VALENCE = 2
beta = to_simulation_units(VALENCE * const.e.si / (const.k_B * 310 * u.K))

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
mesh = ngs.Mesh(create_mesh(geo, mesh_size=MESH_SIZE))

# %%
# Set up FEM objects for diffusion
# see https://docu.ngsolve.org/latest/i-tutorials/unit-2.13-interfaces/interfaceresistivity.html
fes_synapse = ngs.Compress(ngs.H1(mesh, order=1, definedon="synapse_ecs"))
fes_neuropil = ngs.Compress(
    ngs.H1(mesh, order=1, definedon="neuropil", dirichlet="neuropil_boundary"))
fes = fes_synapse * fes_neuropil
(test_s, test_n), (trial_s, trial_n) = fes.TnT()
ca = ngs.GridFunction(fes)

# Set up FEM objects for electrostatics
constraint_fes = ngs.FESpace("number", mesh)
potential_fes = ngs.FESpace([fes_synapse, fes_neuropil, constraint_fes])
pot = ngs.GridFunction(potential_fes)

# Define equations
D_synapse = to_simulation_units(D_COEFFICIENT, 'diffusivity')
D_neuropil = to_simulation_units(D_COEFFICIENT, 'diffusivity') / TORTUOSITY**2
a = ngs.BilinearForm(fes)
a += D_synapse * ngs.grad(test_s) * ngs.grad(trial_s) * ngs.dx("synapse_ecs")
a += D_neuropil * ngs.grad(test_n) * ngs.grad(trial_n) * ngs.dx("neuropil")
a += (test_s - test_n) * (trial_s - trial_n / POROSITY) * ngs.ds("synapse_boundary")
a.Assemble()

m = ngs.BilinearForm(fes)
m += test_s * trial_s * ngs.dx("synapse_ecs")
m += test_n * trial_n * ngs.dx("neuropil")
m.Assemble()

(pot_test_s, pot_test_n, p), (pot_trial_s, pot_trial_n, q) = potential_fes.TnT()
a_pot = ngs.BilinearForm(potential_fes)
a_pot += permittivity * ngs.grad(pot_test_s) * ngs.grad(pot_trial_s) * ngs.dx
a_pot += permittivity * ngs.grad(pot_test_n) * ngs.grad(pot_trial_n) * ngs.dx
a_pot += p * (pot_trial_s + pot_trial_n) * ngs.dx
a_pot += q * (pot_test_s + pot_test_n) * ngs.dx

phi = to_simulation_units(TIME_CONSTANT, 'frequency')
t_param = ngs.Parameter(0)
const_F = const.e.si * const.N_A
terminal_area = ngs.Integrate(1, mesh.Boundaries("presynaptic_membrane"), ngs.BND)
Q_0 = to_simulation_units(CHANNEL_CURRENT / (2 * const_F), 'catalytic activity')
Q = N_CHANNELS * Q_0 * phi * t_param * ngs.exp(-phi * t_param) / terminal_area
b = ngs.LinearForm(fes)
b += D_synapse * Q * trial_s * ngs.ds("presynaptic_membrane")
b += -D_synapse * beta * ca.components[0] \
    * ngs.InnerProduct(ngs.grad(pot.components[0]), ngs.grad(trial_s)) * ngs.dx("synapse_ecs")
b += -D_neuropil * beta * ca.components[1] \
    * ngs.InnerProduct(ngs.grad(pot.components[1]), ngs.grad(trial_n)) * ngs.dx("neuropil")

b_pot = ngs.LinearForm(potential_fes)
b_pot += F * VALENCE * ca.components[0] * trial_s * ngs.dx
b_pot += F * VALENCE * ca.components[1] * trial_n * ngs.dx

# %%
# Initialize the simulation
t_end = to_simulation_units(END_TIME, 'time')
tau = to_simulation_units(TIME_STEP, 'time')
events = {"sampling": 10}
clock = SimulationClock(time_step=tau, end_time=t_end, events=events, verbose=True)

dist = to_simulation_units(SYNAPSE_RADIUS + GLIA_DISTANCE / 2, 'length')
eval_points = np.array([
    [0, 0, 0],          # 1: center
    [dist, 0, 0],       # 2: inside glia, near cleft
    [0, 0, dist],       # 3: inside glia, far from cleft
])
eval_synapse = PointEvaluator(mesh, eval_points)
eval_points = np.array([
    [0, 0, -2 * dist],  # 4: outside glia (below)
    [0, 0, 2 * dist],   # 5: outside glia (above)
])
eval_neuropil = PointEvaluator(mesh, eval_points)

mstar = m.mat.CreateMatrix()
mstar.AsVector().data = m.mat.AsVector() + tau * a.mat.AsVector()
mstar_inv = mstar.Inverse(freedofs=fes.FreeDofs())

a_pot.Assemble()
a_pot_inv = a_pot.mat.Inverse(potential_fes.FreeDofs())

ca_0 = to_simulation_units(CA_RESTING, 'molar concentration')
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
    b_pot.Assemble()
    pot.vec.data = a_pot_inv * b_pot.vec
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
plt.xlim(0, to_simulation_units(END_TIME, 'time'))
plt.ylim(0.4, 1.4)
plt.title('Calcium Concentration Over Time at Different Points')
plt.legend()
plt.show()

# %%
