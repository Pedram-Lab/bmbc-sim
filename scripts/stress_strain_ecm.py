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
# where $σ$ is the Hookean spring constant, and $\Delta x$ is the LSM lattice node spacing. 
#
# | Parameter                          | Definition                              | Value                      |
# |------------------------------------|-----------------------------------------| ---------------------------|
# | $\sigma_g$                         | Glycocalyx spring constant              | 0.02 pN/nm                 |
# | $\sigma_m$                         | Membrane spring constant                | 0.4 pN/nm                  |
# | $\sigma_b$                         | Bond spring constant                    | 2 pN/nm                    |
# | $\Delta x$                         | Lattice node spacing                    | 20 nm                      |
# | $ν$                                | Poisson ratio                           | 0.25                       |       
# | Y$_g$                              | Young's modulus for glycocalyx          | 0.0025 pN/nm$^2$           |
# | Y$_{m}$                            | Young's modulus for PM                  | 0.05 pN/nm$^2$             |
# | Y$_{b}$                            | Young's modulus for bond                | 0.25 pN/nm$^2$             |
# | l$_{g}$                            | Glycocalyx thickness                    | 43 nm                      |
# | l$_{b}$                            | Equilibrium bond lenght                 | 27 nm                      |
# | F                                  | Bond force                              | 0 - 10 pN                  |
# | l$_{d}$                            | Equilibrium separation distance         | 0-15 nm                    |
#
#
# | Compartment                          | Size                         |
# |------------------------------------|--------------------------------|
# | Cell membrane                | 1.4 μm x 1.4 μm x 40 nm   (Nodes = 70x70x3)           |       
# | ECM substrate                | 1.4 μm x 1.4 μm x 400 nm    (Nodes = 70x70x21)        |
# | Glycocalyx                   | 1.4 μm x 1.4 μm x 43 nm                               |
# | Bond formation geometry      | 240 nm x 240 nm x height of the compartment                               |

# %%
