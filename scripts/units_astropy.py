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
# Uncomment to print all units - warning: there are a lot of units!
# print(available_units)

# %% [markdown]
# To define a parameter with its respective units, use the module `units` followed by the name of the unit. For example, the Faraday constant $F = 9.65 \times 10^{-11} \frac{\text{C}}{\text{fmol}}$, can be defined as:

# %%
F_femtomol = 9.65e-11 * u.C / u.femtomole
print(f"Pretty output by default: {F_femtomol}")
print(f"F has a value '{F_femtomol.value}' and a unit '{F_femtomol.unit}'")

# %% [markdown]
# Conversion of the units can be handle using the module `.to`. For example, converting $F = 9.65 \times 10^{-11} \frac{\text{C}}{\text{fmol}}$ to $F = 96.5 \frac{\text{C}}{\text{mmol}}$, can be code as: 

# %%
F_millimol = F_femtomol.to(u.C / u.millimole)
print(f"Before conversion: {F_femtomol}")
print(f"After conversion: {F_millimol}")

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
print(f"I_ch = {I_ch}")

# %% [markdown]
# Astropy generates an error message when incompatible units are used together. The code below is designed to intentionally trigger such an error due to unit incompatibility:

# %%
nu = 4 * u.picoSiemens
ca_ext = 15 * u.millimole
ca_cyt = 0.0001 * u.millimole
delta_ca = ca_ext - ca_cyt
try:
    I_ch = nu + delta_ca
except Exception as e:
    print(f"The following {type(e).__name__} occurred:")
    print(f"    \"{e}\"")

# %% [markdown]
# ## Using Numpy with Astropy Units
#
# Astropy allows the integrations of its units and quantities with Numpy arrays. This is particularly useful for vectorized operations and analyzing datasets:

# %%
import numpy as np

# Create a NumPy array with Astropy units
ca = np.array([15, 0.0001]) * u.millimole
ca_ext, ca_cyt = ca

print(f"[Ca2+]_ext: {ca_ext}")
print(f"[Ca2+]_cyt: {ca_cyt}")
print(f"As an array: {ca}")
print(f"Units are also respected in Numpy operations, e.g., the dot product: {ca.dot(ca)}")

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
print(f"J_ch = {J_ch}")

# %% [markdown]
# The following lines contain the parameter that will be used in the calcium depletion model.

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
