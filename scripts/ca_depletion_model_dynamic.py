# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# This script takes the geometry from `scripts/channel_geometry_with_occ.py` and adds reaction-diffusion of chemical species on top (units are handled by astropy):
# * Ca can diffuse from the ECS to the cytosol through the channel.
#
# The dynamics of the system are resolved in time.

# %%
from ngsolve import *
from ngsolve.webgui import Draw
from tqdm.notebook import trange
from astropy import units as u

from ecsim.geometry import create_ca_depletion_mesh
from ecsim.simulation import Simulation

# %%
diameter_ch = 10 * u.nm
density_channel = 10000 / u.um**2
i_max = 0.1 * u.picoampere

# %%
# Create meshed geometry
mesh = create_ca_depletion_mesh(
    side_length=3 * u.um,
    cytosol_height=3 * u.um,
    ecs_height=0.1 * u.um,
    mesh_size=0.25 * u.um,
    channel_radius=0.5 * u.um
)

# %%
# Set up a simulation on the mesh with BAPTA as a buffer
simulation = Simulation(mesh, time_step=1 * u.ms)
calcium = simulation.add_species(
    "calcium",
    diffusivity={"ecs": 600 * u.um**2 / u.s, "cytosol": 220 * u.um**2 / u.s},
    clamp={"ecs_top": 15 * u.mmol / u.L},
    valence=2
)
free_buffer = simulation.add_species(
    "free_buffer",
    diffusivity={"cytosol": 95 * u.um**2 / u.s},
    valence=-2
)
bound_buffer = simulation.add_species(
    "bound_buffer",
    diffusivity={"cytosol": 113 * u.um**2 / u.s},
    valence=0
)
simulation.add_reaction(
    reactants=(calcium, free_buffer),
    products=bound_buffer,
    kf={"cytosol": 450 * u.umol / (u.L * u.s)},
    kr={"cytosol": 80 / u.s}
)
simulation.add_channel_flux(
    left="ecs",
    right="cytosol",
    boundary="channel",
    rate=1 * u.mmol / (u.L * u.s)
)
    
# Alternative: EGTA as buffer
# free_buffer = simulation.add_species("free_buffer", diffusivity={"cytosol": 113 * u.um**2 / u.s}, valence=-2)
# bound_buffer = simulation.add_species("bound_buffer", diffusivity={"cytosol": 113 * u.um**2 / u.s}, valence=0)
# simulation.add_reaction(reactants=(calcium, free_buffer), products=bound_buffer, kf={"cytosol": 2.7 * u.mmol / (u.L * u.s)}, kr={"cytosol": 0.5 / u.s})

# %%
# Internally set up all finite element infrastructure
simulation.setup_problem()


# %%
# Time stepping - define a function that pre-computes all timesteps
def time_stepping(simulation, t_end, n_samples):
    n_steps = int(ceil(t_end.value / simulation._time_step_size.to(u.s).value))
    sample_int = int(ceil(n_steps / n_samples))
    u_ca_t = GridFunction(simulation._fes, multidim=0)
    u_buf_t = GridFunction(simulation._fes, multidim=0)
    u_com_t = GridFunction(simulation._fes, multidim=0)
    u_ca_t.AddMultiDimComponent(simulation.concentrations["calcium"].vec)
    u_buf_t.AddMultiDimComponent(simulation.concentrations["free_buffer"].vec)
    u_com_t.AddMultiDimComponent(simulation.concentrations["bound_buffer"].vec)
    
    for i in trange(n_steps):
        simulation.time_step()
        if i % sample_int == 0:
            u_ca_t.AddMultiDimComponent(simulation.concentrations["calcium"].vec)
            u_buf_t.AddMultiDimComponent(simulation.concentrations["free_buffer"].vec)
            u_com_t.AddMultiDimComponent(simulation.concentrations["bound_buffer"].vec)
    return u_ca_t, u_buf_t, u_com_t


# %%
# Time stepping - set initial conditions and do time stepping
with TaskManager():
    simulation.init_concentrations(
        calcium={"ecs": 15 * u.mmol / u.L, "cytosol": 0.1 * u.umol / u.L},
        free_buffer={"cytosol": 1 * u.mmol / u.L}, # bapta
        # free_buffer={"cytosol": 4.5 * u.mmol / u.L}, # low egta
        # free_buffer={"cytosol": 40 * u.mmol / u.L}, # high egta
        bound_buffer={"cytosol": 0 * u.mmol / u.L}
    )
    ca_t, buffer_t, complex_t = time_stepping(simulation, t_end=0.1 * u.s, n_samples=100)

# %%
# Visualize (because of the product structure of the FESpace, the usual
# visualization of time-dependent functions via multidim is not possible)
visualization = mesh.MaterialCF({"ecs": ca_t.components[0], "cytosol": ca_t.components[1]})
clipping = {"function": True,  "pnt": (0, 0, 1.5), "vec": (0, 1, 0)}
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}, "Colormap": {"ncolors": 32, "autoscale": False, "max": 15}}
Draw(ca_t.components[1], clipping=clipping, settings=settings, interpolate_multidim=True, animate=True)

# %%
