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
# # Unit conversions
# This little example should showcase how changing the default prefixes of base units change the prefixes in dependent units. E.g., by giving all substances in fmol and all lengths in um, the default unit for concentrations is mM (where M = mol / l).

# %%
import astropy.units as u

# %%
# Define a unit for molarity
molar = u.mol / u.L

# %%
# Define all quantities that show up in the system
phi = u.V
epsilon = u.F / u.m
c = molar
D = u.m**2 / u.s
kf = (u.s * molar)**(-1)
kr = u.s**(-1)
beta = u.C / u.J
F = u.C / u.mol
rho = u.F * u.V / u.m**3
quantities = dict(phi=phi, epsilon=epsilon, c=c, D=D, kf=kf, kr=kr, beta=beta, F=F, rho=rho)

# %%
for name, q in quantities.items():
    print(f"{name} = {q.decompose()}")

# %%
# Determine prefixes of SI base units
# Keep in mind that changing time and length units also change temporal and spatial derivatives!
mass = u.kg
time = u.ms
length = u.um
current = u.uA
substance = u.pmol
temp = u.K
luminous_intensity = u.cd

# %%
# Write units in the new base units (as above)
phi_unit = length**2 * mass / (current * time**3)
epsilon_unit = time**4 * current**2 / (mass * length**3)
D_unit = length**2 / time
c_unit = substance / length**3
kf_unit = (c_unit * time)**(-1)
kr_unit = time**(-1)
beta_unit = time**3 * current / (mass * length**2)
F_unit = current * time / substance
rho_unit = current * time / length**3

# %%
# This determines the prefixes for dependent units
for name, q in quantities.items():
    unit = locals()[name + '_unit']
    print(f"{name} = {q.to(unit):.0e} {unit} [{q}]")

# %%
