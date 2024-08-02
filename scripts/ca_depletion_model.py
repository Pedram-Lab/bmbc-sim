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
# This script takes the `geometry from scripts/channel_geometry_with_occ.py` and adds reaction-diffusion of chemical species on top and adds the library Astropy to handle units.
#
# # Astropy
#
# Astropy is a Python library specifically designed for astronomy and astrophysics. It provides a wide range of tools and features to facilitate the processing and analysis of astronomical data. Astropy has a robust system for handling **physical units and quatities**, allowing for easy unit conversions and ensuring that calculation involving physical dimension are consistent and correct. Given its capabilities, we have incorporated Astropy into our project focused on the modeling of the extracellular matrix. Consequently, Astropy has been included in the environment.yaml file.
#
#
# ## Constants 
#
# The module`astropy.constants` includes a wide range of physical constants. For example,   
#
# ```python
# from astropy import constants as const
# print(const.N_A)
# ```
#
# will print the Avogadro's constant, denoted here as N_A. To view all available constants, use the following code:
#
# ```python
# for name in dir(const):
#     obj = getattr(const, name)
#     if isinstance(obj, const.Constant):
#         print(f"{name}: {obj.name} ({obj.value} {obj.unit})")
# ```
#
# ## Units 
#
# `astropy.units` is a module that facilitates handling, converting, and manipulating physical units. It provides support for defining quantities with units, performing arithmetic with these quantities while automatically handling unit conversions, and ensuring that unit-consistent calculations are maintained throughout scientific computing tasks. 
#
# Astropy defines basic units like mole, meter, second, etc. These units can be combined with SI prefixes to form new units such as femtomole, kilometer, or millisecond. Prefixes such as "femto-" for femtomole are handled through the `units` module, which supports a comprehensive set of SI prefixes ranging from "yocto" ($10^{-24}$) to "yotta" ($10^{24}$). To view all available units, use the following code:  
#
# ```python
# import astropy.units as u
# available_units = [attr for attr in dir(u) if isinstance(getattr(u, attr), u.UnitBase)]
# ```
#
# To define a parameter with its respective units, use the module `units` followed by the name of the unit. For example, the Faraday constant $F = 9.65 \times 10^{-11} \frac{\text{C}}{\text{fmol}}$, can be defined as:
#
# ```python
# F_femtomol = 9.65e-11 * u.C / u.femtomole
# ```
#
# Conversion of the units can be handle using the module `.to`. For example, converting $F = 9.65 \times 10^{-11} \frac{\text{C}}{\text{fmol}}$ to $F = 96.5 \frac{\text{C}}{\text{mmol}}$, can be code as: 
#
# ```python
# F_femtomol = 9.65e-11 * u.C / u.femtomole
# F_millimol = F_femtomol.to(u.C / u.millimole)
# ```
#
# This will automatically handle the ($10^{-15}$) conversion factor inherent in the "femto-" prefix.
#
#
# ## Calculations 
#
# When conducting calculations, Astropy verifies the compatibility of units and raises errors if incompatible units are combined. It handles arithmetic operations, checking and adjusting units as needed to avoid unit-related errors in scientific computations. For instance, to calculate the individual current through a calcium channel using the formula  $I_{\text{ch}} = \nu_{\text{Ch}} \left([Ca^{2+}]_{\text{ext}} - [Ca^{2+}]_{\text{cyt}}\right)$, you can use the following code:
#
# ```python
# nu = 4 * u.picoSiemens
# ca_ext = 15 * u.millimole
# ca_cyt = 0.0001 * u.millimole
# delta_ca = ca_ext - ca_cyt
# I_ch = nu * delta_ca
# print(f"I_ch = {I_ch.value}{I_ch.unit}")
# ```
#
# ```python
# I_ch = 59.9996mmol pS
# ```
#
# Astropy generates an error message when incompatible units are used together. The code below is designed to intentionally trigger such an error due to unit incompatibility:
#
# ```python
# nu = 4 * u.picoSiemens
# ca_ext = 15 * u.millimole
# ca_cyt = 0.0001 * u.millimole
# delta_ca = ca_ext - ca_cyt
# I_ch = nu + delta_ca
# print(f"I_ch = {I_ch.value}{I_ch.unit}")
# ```
# ```python
# UnitConversionError: 'mmol' (amount of substance) and 'pS' (electrical conductance) are not convertible
# UnitConversionError: Can only apply 'add' function to quantities with compatible dimensions
#
# ```
#
# The following code calculates the flux for a single channel. This computation will be scrutinized thoroughly due to the unexpected result concerning the units.
#
# ```python
# nu = 4 * u.picoSiemens
# ca_ext = 15 * u.millimole
# ca_cyt = 0.0001 * u.millimole
# delta_ca = ca_ext - ca_cyt
# I_ch = nu * delta_ca
# F_femtomol = 9.65e-11 * u.C / u.femtomole
# F_millimol = F_femtomol.to(u.C / u.millimole)
# dv = 1.25e-19 * u.liter
# J_ch = I_ch / (2*F_millimol*dv)
# print(f"J_ch = {J_ch.value}{J_ch.unit}")
# ```
#
# ```python
# J_ch = 2.4870300518134717e+18mmol2 pS / (C l)
# ```
#
# ## Using Numpy with Astropy Units
#
# Astropy allows the integrations of its units and quantities with Numpy arrays. This is particularly useful for vectorized operations and analyzing datasets:
#
#
# ```python
# import numpy as np
# from astropy import units as u
#
# # Create a NumPy array with Astropy units
# ca_ext, ca_cyt = np.array([15, 0.0001]) * u.millimole
#
# print("[Ca2+]_ext:", ca_ext)
# ```
#
# ```python
# [Ca2+]_ext: 15.0 mmol
# ```
#

# %%
from ngsolve import *
from ngsolve.webgui import Draw

from ecsim.geometry import create_ca_depletion_mesh
from astropy import units as u
import matplotlib.pyplot as plt

# %%
ca_ext = 15 * u.millimole
ca_cyt = 0.0001 * u.millimole
egta_1 = 4.5 * u.millimole
egta_2 = 40 * u.millimole
bapta = 1 * u.millimole
diff_ca_ext = 600 * u.um**2 / u.s
diff_ca_cyt = 220 * u.um**2 / u.s
diff_free_egta = 113 * u.um**2 / u.s
diff_bound_egta = 113 * u.um**2 / u.s
diff_free_bapta = 95 * u.um**2 / u.s
diff_bound_bapta = 113 * u.um**2 / u.s
k_f_egta = 2.7 * u.micromole / u.s
k_r_egta = 0.5 / u.s
k_f_bapta = 450 * u.micromole / u.s
k_r_bapta = 80 / u.s
diameter_ch = 10 * u.nm
density_channel = 10000 / u.um**2
i_max = 0.1 * u.picoampere

# %%
# Create meshed geometry
mesh = create_ca_depletion_mesh(side_length=3, cytosol_height=3, ecs_height=0.1, mesh_size=0.25, channel_radius=0.5)
print(mesh.GetBoundaries())

# %%
clipping = {"function": True,  "pnt": (0, 0, 1.5), "vec": (0, 1, 0)}
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -60}]}}
Draw(mesh, clipping=clipping, settings=settings)

# %%
# Define and assemble the FE-problem
# We set the cytosol boundary to zero for visualization purposes
ecs_fes = H1(mesh, order=2, definedon=mesh.Materials("ecs"), dirichlet="ecs_top")
cytosol_fes = H1(mesh, order=2, definedon=mesh.Materials("cytosol"), dirichlet="boundary")
fes = FESpace([ecs_fes, cytosol_fes])
u_ecs, u_cyt = fes.TrialFunction()
v_ecs, v_cyt = fes.TestFunction()

f = LinearForm(fes)

a = BilinearForm(fes)
a += grad(u_ecs) * grad(v_ecs) * dx("ecs")              # diffusion in ecs
a += grad(u_cyt) * grad(v_cyt) * dx("cytosol")          # diffusion in cytosol
a += (u_ecs - u_cyt) * (v_ecs - v_cyt) * ds("channel")  # interface flux

a.Assemble()
f.Assemble()

# %%
# Set concentration at top to 15 and solve the system
concentration = GridFunction(fes)
concentration.components[0].Set(15, definedon=mesh.Boundaries("ecs_top"))
res = f.vec.CreateVector()
res.data = f.vec - a.mat * concentration.vec
concentration.vec.data += a.mat.Inverse(fes.FreeDofs()) * res

# %%
# Visualize (the colormap is quite extreme for dramatic effect)
visualization = mesh.MaterialCF({"ecs": concentration.components[0], "cytosol": concentration.components[1]})
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}, "Colormap": {"ncolors": 256, "autoscale": False, "max": 3}}
Draw(visualization, mesh, clipping=clipping, settings=settings)

# %%
# Para evaluar la función en un conjunto de puntos dentro del citosol:
X = np.linspace(-1.5, 1.5, num=50)  # Ajusta los límites según la geometría real del citosol
Y = np.zeros_like(X)
Z = np.linspace(0, 3, num=50)  # Altura del citosol

values = []
for x, z in zip(X, Z):
    if mesh.Inside((x, 0, z)):  # Comprueba si el punto está dentro de la malla del citosol
        mip = mesh(x, 0, z)  # Mapped Integration Point
        values.append(cytosol_function(mip))
    else:
        values.append(None)  # Para los puntos fuera de la malla

# Trazar los valores de la función en los puntos del citosol
plt.figure()
plt.plot(Z, values)  # Asume que queremos ver cómo cambia la función con la altura z
plt.xlabel('z')
plt.ylabel('Function value')
plt.title('Function values along a vertical line in the cytosol')
plt.grid(True)
plt.show()

