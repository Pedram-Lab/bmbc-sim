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
# | Compartment                          | Size                         |
# |------------------------------------|--------------------------------|
# | Cell membrane                | 1.4 μm x 1.4 μm x 40 nm          |
# | Glycocalyx                   | 1.4 μm x 1.4 μm x 45 nm                               |

# %%
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import ngsolve as ngs
import netgen.occ as occ
from ngsolve.webgui import Draw

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
geo.faces[4].bc("top")
geo.faces[10].bc("interface")
geo.faces[11].bc("substrate")

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
def sample_cutout(fe_mesh, deformation, n_samples=300):
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
def visualize_deformation_2d(fe_mesh, deformation):
    """Visualize deformation as 2D heatmap."""
    x, y, z = sample_cutout(fe_mesh, deformation)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(
        x,
        y,
        (ECS_HEIGHT - z) * 1000,
        cmap="gnuplot",
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
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    c = ax.plot_surface(
        x,
        y,
        (ECS_HEIGHT - z) * 1000,
        cmap="gnuplot",
        vmin=0,
        vmax=18,
        antialiased=False,
    )
    ax.set_xlim3d(-0.05, 0.05)
    ax.set_ylim3d(-0.05, 0.05)
    ax.view_init(160, 70, 0)
    ax.set_facecolor('black')
    ax.set_box_aspect((10, 10, 5))
    ax.set_axis_off()
    fig.colorbar(c, ax=ax, label='deformation [nm]')
    plt.show()

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
