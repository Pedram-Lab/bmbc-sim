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
# # Astropy
#
# Astropy is a Python library specifically designed for astronomy and astrophysics. It provides a wide range of tools and features to facilitate the processing and analysis of astronomical data. Astropy has a robust system for handling **physical units and quatities**, allowing for easy unit conversions and ensuring that calculation involving physical dimension are consistent and correct. Thus, the `astropy.units` module brings great value to our project focused on the modeling of the extracellular matrix.

# %% [markdown]
# ## Constants
#
# The module`astropy.constants` includes a wide range of physical constants. For example,   

# %%
import astropy
from astropy import constants as const
print(const.N_A)

# %% [markdown]
# will print the Avogadro's constant, denoted here as N_A. To view all available constants, use the following code:

# %%
for name in dir(const):
    obj = getattr(const, name)
    if isinstance(obj, const.Constant):
        print(f"{name}: {obj.name} ({obj.value} {obj.unit})")

# %% [markdown]
# ## Units 
#
# `astropy.units` is a module that facilitates handling, converting, and manipulating physical units. It provides support for defining quantities with units, performing arithmetic with these quantities while automatically handling unit conversions, and ensuring that unit-consistent calculations are maintained throughout scientific computing tasks. 
#
# Astropy defines basic units like mole, meter, second, etc. These units can be combined with SI prefixes to form new units such as femtomole, kilometer, or millisecond. Prefixes such as "femto-" for femtomole are handled through the `units` module, which supports a comprehensive set of SI prefixes ranging from "yocto" ($10^{-24}$) to "yotta" ($10^{24}$). To view all available units, use the following code:  

# %%
import astropy.units as u
available_units = [attr for attr in dir(u) if isinstance(getattr(u, attr), u.UnitBase)]

# %% [markdown]
# To define a parameter with its respective units, use the module `units` followed by the name of the unit. For example, the Faraday constant $F = 9.65 \times 10^{-11} \frac{\text{C}}{\text{fmol}}$, can be defined as:

# %%
F_femtomol = 9.65e-11 * u.C / u.femtomole

# %% [markdown]
# Conversion of the units can be handle using the module `.to`. For example, converting $F = 9.65 \times 10^{-11} \frac{\text{C}}{\text{fmol}}$ to $F = 96.5 \frac{\text{C}}{\text{mmol}}$, can be code as: 

# %%
F_femtomol = 9.65e-11 * u.C / u.femtomole
F_millimol = F_femtomol.to(u.C / u.millimole)

# %% [markdown]
# This will automatically handle the ($10^{-15}$) conversion factor inherent in the "femto-" prefix.

# %% [markdown]
# ## Calculations 
#
# When conducting calculations, Astropy verifies the compatibility of units and raises errors if incompatible units are combined. It handles arithmetic operations, checking and adjusting units as needed to avoid unit-related errors in scientific computations. For instance, to calculate the individual current through a calcium channel using the formula  $I_{\text{ch}} = \nu_{\text{Ch}} \left([Ca^{2+}]_{\text{ext}} - [Ca^{2+}]_{\text{cyt}}\right)$, you can use the following code:

# %%
nu = 4 * u.picoSiemens
ca_ext = 15 * u.millimole
ca_cyt = 0.0001 * u.millimole
delta_ca = ca_ext - ca_cyt
I_ch = nu * delta_ca
print(f"I_ch = {I_ch.value}{I_ch.unit}")

# %% [markdown]
# Astropy generates an error message when incompatible units are used together. The code below is designed to intentionally trigger such an error due to unit incompatibility:

# %%
nu = 4 * u.picoSiemens
ca_ext = 15 * u.millimole
ca_cyt = 0.0001 * u.millimole
delta_ca = ca_ext - ca_cyt
I_ch = nu + delta_ca
print(f"I_ch = {I_ch.value}{I_ch.unit}")

# %% [markdown]
# ## Using Numpy with Astropy Units
#
# Astropy allows the integrations of its units and quantities with Numpy arrays. This is particularly useful for vectorized operations and analyzing datasets:

# %%
import numpy as np

# Create a NumPy array with Astropy units
ca_ext, ca_cyt = np.array([15, 0.0001]) * u.millimole

print("[Ca2+]_ext:", ca_ext)
print("[Ca2+]_cyt:", ca_cyt)

# %% [markdown]
# The following code calculates the flux for a single channel. This computation will be scrutinized thoroughly due to the unexpected result concerning the units.

# %%
nu = 4 * u.picoSiemens
ca_ext = 15 * u.millimole
ca_cyt = 0.0001 * u.millimole
delta_ca = ca_ext - ca_cyt
I_ch = nu * delta_ca
F_femtomol = 9.65e-11 * u.C / u.femtomole
F_millimol = F_femtomol.to(u.C / u.millimole)
dv = 1.25e-19 * u.liter
J_ch = I_ch / (2*F_millimol*dv)
print(f"J_ch = {J_ch.value}{J_ch.unit}")

# %% [markdown]
# The following lines contain the parameter that will be used in the calcium depletion model.

# %%
ca_ext = 15 * u.millimole
print(ca_ext)

# %%
ca_cyt = 0.0001 * u.millimole
print(ca_cyt)

# %%
egta_1 = 4.5 * u.millimole
print(egta_1)

# %%
egta_2 = 40 * u.millimole
print(egta_2)

# %%
bapta = 1 * u.millimole
print(bapta)

# %%
diff_ca_ext = 600 * u.um**2 / u.s
print(diff_ca_ext)

# %%
diff_ca_cyt = 220 * u.um**2 / u.s
print(diff_ca_cyt)

# %%
diff_free_egta = 113 * u.um**2 / u.s
print(diff_free_egta)

# %%
diff_bound_egta = 113 * u.um**2 / u.s
print(diff_free_egta)

# %%
diff_free_bapta = 95 * u.um**2 / u.s
print(diff_free_bapta)

# %%
diff_bound_bapta = 113 * u.um**2 / u.s
print(diff_free_bapta)

# %%
k_f_egta = 2.7 * u.micromole / u.s
print(k_f_egta)

# %%
k_r_egta = 0.5 / u.s
print(k_r_egta)

# %%
k_f_bapta = 450 * u.micromole / u.s
print(k_f_bapta)


# %%
k_r_bapta = 80 / u.s
print(k_r_bapta)

# %%
diameter_ch = 10 * u.nm
print(diameter_ch)

# %%
density_channel = 10000 / u.um**2
print(density_channel)

# %%
i_max = 0.1 * u.picoampere
print(i_max)

# %%
