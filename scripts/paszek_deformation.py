# %% [markdown]
# # Verify mechanical parameters based on the Paszek paper
# This script recreates the setup of Paszek et al. *Integrin Clustering Is
# Driven by Mechanical Resistance from the Glycocalyx and the Substrate* (2009).
#
# According to this paper, Young's modulus is determined by the following expression:
#
# $Y = \frac{5\sigma}{2 \Delta x}$
#
# where $\sigma$ is the Hookean spring constant, and $\Delta x$ is the LSM lattice node spacing.
#
# | Parameter   | Definition                      | Value            |
# |-------------|---------------------------------| -----------------|
# | $\sigma_g$  | Glycocalyx spring constant      | 0.01 pN/nm       |
# | $\sigma_m$  | Membrane spring constant        | 0.4 pN/nm        |
# | $\sigma_b$  | Bond spring constant            | 2 pN/nm          |
# | $\sigma_s$  | Substrate spring constant       | 1000 pN/nm       |
# | $\Delta x$  | Lattice node spacing            | 20 nm            |
# | $ν$         | Poisson ratio                   | 0.25             |
# | Y$_g$       | Young's modulus for glycocalyx  | 0.0025 pN/nm$^2$ |
# | Y$_{m}$     | Young's modulus for PM          | 0.05 pN/nm$^2$   |
# | Y$_{b}$     | Young's modulus for bond        | 0.25 pN/nm$^2$   |
# | Y$_{s}$     | Young's modulus for substrate   | 125 pN/nm$^2$    |
# | l$_{g}$     | Glycocalyx thickness            | 45 nm            |
# | l$_{b}$     | Equilibrium bond length         | 18 nm            |
# | F           | Bond force                      | 4 pN             |
# | l$_{d}$     | Equilibrium separation distance | 13.5 nm          |
#
#
# | Compartment                  | Size                         |
# |------------------------------|------------------------------|
# | Cell membrane                | 1.4 μm x 1.4 μm x 40 nm      |
# | Glycocalyx                   | 1.4 μm x 1.4 μm x 45 nm      |

# %%
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import ngsolve as ngs
from netgen import occ
from ngsolve.webgui import Draw
import pyvista as pv

from ecsim.units import to_simulation_units

# %%
# Define units and parameters for the simulation
S = to_simulation_units(1.4 * u.um) / 2
ECS_HEIGHT = to_simulation_units(45 * u.nm)
MEMBRANE_HEIGHT = to_simulation_units(40 * u.nm)
MESH_SIZE = to_simulation_units(10 * u.nm)
CUTOUT_SIZE = to_simulation_units(300 * u.nm)

PULLING_FORCE = to_simulation_units(10 * u.pN)
YOUNG_MODULUS_ECS = to_simulation_units(2.5 * u.fN / u.nm ** 2)
YOUNG_MODULUS_MEMBRANE = to_simulation_units(50 * u.fN / u.nm ** 2)
POISSON_RATIO = 0.25

# %%
# Define geometry (we consider the substrate as fixed and don't model it explicitly)
ecs = occ.Box((-S, -S, 0), (S, S, ECS_HEIGHT)).mat("ecs").bc("side")
membrane = occ.Box(
    (-S, -S, ECS_HEIGHT),
    (S, S, ECS_HEIGHT + MEMBRANE_HEIGHT)
).mat("membrane").bc("side")
geo = occ.Glue([ecs, membrane])
geo.faces[4].bc("substrate")
geo.faces[10].bc("interface")
geo.faces[11].bc("top")

geo = occ.OCCGeometry(geo)
mesh = ngs.Mesh(geo.GenerateMesh(maxh=MESH_SIZE))
Draw(mesh)

# %%
# Set Lamé constants based on Young's modulus and Poisson ratio
E = mesh.MaterialCF({"ecs": YOUNG_MODULUS_ECS, "membrane": YOUNG_MODULUS_MEMBRANE})
NU = POISSON_RATIO

mu = E / (2 * (1 + NU))
lam = E * NU / ((1 + NU) * (1 - 2 * NU))

clipping = {"function": False,  "pnt": (0, 0, 0), "vec": (0, 1, 0)}
visualization_settings = {
    "camera": {"transformations": [{"type": "rotateX", "angle": -80}]},
    "deformation": 3.0,
}


# %%
def stress(strain):
    """Stress-strain relation for linear isotropic materials."""
    return 2 * mu * strain + lam * ngs.Trace(strain) * ngs.Id(3)

# %%
# Define mechanic problem
fes = ngs.VectorH1(mesh, order=1, dirichlet="substrate")
u, v = fes.TnT()

blf = ngs.BilinearForm(fes)
blf += ngs.InnerProduct(stress(ngs.Sym(ngs.Grad(u))), ngs.Sym(ngs.Grad(v))).Compile() * ngs.dx
with ngs.TaskManager():
    blf.Assemble()
    a_inv = blf.mat.Inverse(fes.FreeDofs())


# %%
def compute_solution(points, force):
    """
    Compute elastic deformation of membrane and ECS for given pulling forces
    at specified points.
    """
    deformation = ngs.GridFunction(fes)
    lf = ngs.LinearForm(fes)
    for p in points:
        lf += (-force * v[2])(*p)

    with ngs.TaskManager():
        lf.Assemble()
        deformation.vec.data = a_inv * lf.vec
    return deformation


# %%
def sample_cutout(fe_mesh, deformation, n_samples=100):
    """Sample deformation on a square cutout centered at (0, 0, ECS_HEIGHT)."""
    x = np.linspace(-CUTOUT_SIZE / 2, CUTOUT_SIZE / 2, n_samples)
    y = np.linspace(-CUTOUT_SIZE / 2, CUTOUT_SIZE / 2, n_samples)
    x, y = np.meshgrid(x, y)
    z = ECS_HEIGHT

    points = [(x, y, z) for x, y in zip(x.flatten(), y.flatten())]
    mips = [fe_mesh(*p) for p in points]
    deformation_z = [deformation.components[2](p) for p in mips]

    return x, y, z + np.array(deformation_z).reshape((n_samples, n_samples))

# %%
# Compute elastic deformation of membrane and ECS for one ...
pulling_points = [(0, 0, ECS_HEIGHT)]
deformation_1 = compute_solution(pulling_points, PULLING_FORCE)
# Draw(deformation_1, settings=visualization_settings, clipping=clipping)

# %%
# ... two ...
pulling_points = [(-0.04, 0, ECS_HEIGHT), (0.04, 0, ECS_HEIGHT)]
deformation_2 = compute_solution(pulling_points, 0.9 * PULLING_FORCE)
# Draw(deformation_2, settings=visualization_settings, clipping=clipping)

# %%
# ... and three points
pulling_points = [(-0.03, -0.02, ECS_HEIGHT), (0.03, -0.02, ECS_HEIGHT), (0, 0.03, ECS_HEIGHT)]
deformation_3 = compute_solution(pulling_points, 0.7 * PULLING_FORCE)
# Draw(deformation_3, settings=visualization_settings, clipping=clipping)

# %%
# Recreate the colormap used in Paszek et al.
cmap_x = [0.0, 0.2667, 0.5529, 0.8157, 1.0]
cmap_rgb = [
    [0.00025, 0.00392, 0.02264],   # black
    [0.01895, 0.00596, 0.96124],   # blue
    [0.98346, 0.03252, 0.01562],   # red
    [0.99593, 0.96221, 0.01912],   # yellow
    [0.99216, 1.00000, 0.57027],   # end
]
custom_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "paszek", list(zip(cmap_x, cmap_rgb))
)

# %%
def visualize_deformation_2d(fe_mesh, deformation):
    """Visualize deformation as 2D heatmap."""
    x, y, z = sample_cutout(fe_mesh, deformation)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(
        x,
        y,
        (ECS_HEIGHT - z) * 1000,
        cmap=custom_cmap,
        shading="gouraud",
        vmin=0,
        vmax=18,
    )
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(c, ax=ax, label='deformation [nm]')
    plt.show()

# %%
def visualize_deformation_3d(fe_mesh, deformation):
    """Visualize deformation as 3D surface plot."""
    x, y, z = sample_cutout(fe_mesh, deformation)
    deformation_nm = (ECS_HEIGHT - z) * 10

    # Create a structured grid that mirrors the rectangular sampling domain.
    grid = pv.StructuredGrid(
        np.ascontiguousarray(x),
        np.ascontiguousarray(y),
        np.ascontiguousarray(deformation_nm),
    )

    grid["deformation"] = deformation_nm.flatten(order="F")
    grid.set_active_scalars("deformation")

    plotter = pv.Plotter()
    plotter.set_background("black")
    plotter.add_mesh(
        grid,
        cmap=custom_cmap,
        clim=(0, 0.18),
        show_edges=True,
        show_scalar_bar=False,
    )

    plotter.set_scale(zscale=0.5)

    direction = np.array([-0.5, -1.0, 0.5]) / 3
    plotter.camera_position = [direction.tolist(), [0, 0, 0.03], [0, 0, -1]]
    plotter.camera.Roll(0)

    plotter.show()

# %%
# Visualize elastic deformation of membrane and ECS after some post-processing
visualize_deformation_2d(mesh, deformation_1)
visualize_deformation_2d(mesh, deformation_2)
visualize_deformation_2d(mesh, deformation_3)

# %%
# Visualize elastic deformation of membrane and ECS after some post-processing
visualize_deformation_3d(mesh, deformation_1)
visualize_deformation_3d(mesh, deformation_2)
visualize_deformation_3d(mesh, deformation_3)

# %%
# Show colorbar for 3D plots
plotter = pv.Plotter()
plotter.set_background("white")
dummy_sphere = pv.Sphere(radius=0.001)
dummy_sphere["deformation"] = np.linspace(0, 18, dummy_sphere.n_points)
mesh_plot = plotter.add_mesh(
    dummy_sphere,
    cmap=custom_cmap,
    clim=(0, 18),
    show_scalar_bar=False,
)
plotter.add_scalar_bar(
    title='Deformation distance [nm]',
    n_labels=10,
    title_font_size=20,
    label_font_size=16,
    width=0.5,
    height=0.1,
    fmt="%.0f",
)
mesh_plot.SetVisibility(False)
plotter.show()

# %%
