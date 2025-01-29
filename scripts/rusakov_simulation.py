# %% [markdown]
# # Rusakov Simulation
# This script recreates the simulation from [Rusakov 2001]. Specifically, we
# will recreate the simulation of Ca-depletion for presynaptic, AP-driven
# calcium influx (Figure 4, top row).

# %%
import astropy.units as u
from netgen.webgui import Draw

from ecsim.geometry import create_rusakov_geometry, create_mesh

# %%
clipping_settings = {"function": False,  "pnt": (0, 0, 0), "vec": (0, 1, 0)}
visualization_settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}}

# %%
# Geometry parameters
TOTAL_SIZE = 2 * u.um        # guessed
SYNAPSE_RADIUS = 0.1 * u.um  # Fig. 4
CLEFT_SIZE = 30 * u.nm       # Sec. "Ca2 diffusion in a calyx-type synapse"
GLIA_DISTANCE = 30 * u.nm    # guessed
GLIA_WIDTH = 100 * u.nm      # Sec. "Glial sheath and glutamate transporter density"

# Ca parameters
CA_RESTING = 1.3 * u.mmol / u.L  # Sec. "Presynaptic calcium influx"
TIME_CONSTANT = 10 / u.ms        # Sec. "Presynaptic calcium influx"
N_CHANNELS = 39                  # Fig. 4

# Simulation parameters
MESH_SIZE = 0.1 * u.um
TIME_STEP = 1.0 * u.us
END_TIME = 1.5 * u.ms

# %%
# No glial coverage
geo = create_rusakov_geometry(
    total_size=TOTAL_SIZE,
    synapse_radius=SYNAPSE_RADIUS,
    cleft_size=CLEFT_SIZE,
    glia_distance=GLIA_DISTANCE,
    glia_width=GLIA_WIDTH,
    glial_coverage_angle=90 * u.deg,
)
Draw(geo, clipping=clipping_settings, settings=visualization_settings)

# %%
mesh = create_mesh(geo, mesh_size=MESH_SIZE)
Draw(mesh, clipping=clipping_settings, settings=visualization_settings)

# %%
