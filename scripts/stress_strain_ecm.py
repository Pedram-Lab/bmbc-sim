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
# Accoding to Paszek, Young's modulus is determined by the following expression:
#
# $Y = \frac{5\sigma}{2 \Delta x}$
#
# where $σ$ is the Hookean spring constant, and $\Delta x$ is the LSM lattice node spacing. If $\sigma_g = 0.02 pN/nm$, $\sigma_m = 0.02 pN/nm$, and $\Delta x = 20 nm$. We get the following parameter values for the stress-strain ECM. 
#
# | Parameter                          | Value                          |
# |------------------------------------|--------------------------------|
# | Poisson ratio ($ν$)                | 0.25                           |       
# | Young's modulus for glycocalyx     | 0.0025 pN/nm$^2$               |
# | Young's modulus for PM             | 0.05 pN/nm$^2$                 |
