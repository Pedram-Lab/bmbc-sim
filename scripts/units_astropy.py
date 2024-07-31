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

# %%
import astropy

# %%
from astropy import units as u

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
