# %% [markdown]
# # Geometry of the Rusakov model
# Here, we showcase the geometry of the Rusakov model. The geometry consists of
# two semi-spheric synaptic terminals separated by a synaptic cleft. The
# terminals are surrounded in some distance by a glial cell with an adjustable
# coverage angle (measured from the top). The whole geometry is contained in an
# enclosing box.
#
# In all plots, the presynaptic terminal is colored red, the postsynaptic terminal
# is colored blue, and the glial cell is colored green.

# %%
from netgen.webgui import Draw
from astropy.units import um, nm, deg

from bmbcsim.geometry import create_rusakov_geometry

# %%
clipping_settings = {"function": False,  "pnt": (0, 0, 0), "vec": (0, 1, 0)}
visualization_settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}}

# %%
# No glial coverage
geo = create_rusakov_geometry(
    total_size=3 * um,
    synapse_radius=0.4 * um,
    cleft_size=30 * nm,
    glia_distance=30 * nm,
    glia_width=0.2 * um,
    glial_coverage_angle=0 * deg,
)
Draw(geo, clipping=clipping_settings, settings=visualization_settings)

# %%
# "The sailor"
geo = create_rusakov_geometry(
    total_size=3 * um,
    synapse_radius=0.4 * um,
    cleft_size=30 * nm,
    glia_distance=30 * nm,
    glia_width=0.2 * um,
    glial_coverage_angle=45 * deg,
)
Draw(geo, clipping=clipping_settings, settings=visualization_settings)

# %%
# Half of the terminal is covered by glia
geo = create_rusakov_geometry(
    total_size=3 * um,
    synapse_radius=0.4 * um,
    cleft_size=30 * nm,
    glia_distance=30 * nm,
    glia_width=0.2 * um,
    glial_coverage_angle=90 * deg,
)
Draw(geo, clipping=clipping_settings, settings=visualization_settings)

# %%
# Most of the terminal is covered by glia
geo = create_rusakov_geometry(
    total_size=3 * um,
    synapse_radius=0.4 * um,
    cleft_size=30 * nm,
    glia_distance=30 * nm,
    glia_width=0.2 * um,
    glial_coverage_angle=150 * deg,
)
Draw(geo, clipping=clipping_settings, settings=visualization_settings)

# %%
# All of the terminal is covered by glia
geo = create_rusakov_geometry(
    total_size=3 * um,
    synapse_radius=0.4 * um,
    cleft_size=30 * nm,
    glia_distance=30 * nm,
    glia_width=0.2 * um,
    glial_coverage_angle=180 * deg,
)
Draw(geo, clipping=clipping_settings, settings=visualization_settings)

# %%
