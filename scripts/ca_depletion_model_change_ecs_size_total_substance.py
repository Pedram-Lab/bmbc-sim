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
# This script takes the geometry from `scripts/channel_geometry_with_occ.py` and adds reaction-diffusion of chemical species on top (units are handled by astropy):
#
#
# \begin{equation}
#     \mathrm{Ca}^{2+}_{\mathrm{cyt}} + \mathrm{EGTA} \rightleftharpoons  \mathrm{EGTA}_{\mathrm{bound}}
# \end{equation}
#
#
# \begin{aligned}
#     \frac{\partial [\text{Ca}^{2+}]_{\text{cyt}}}{\partial t} &= D_{\text{Ca}_{\text{cyt}}} \Delta [\text{Ca}^{2+}]_{\text{cyt}} - k_{\text{EGTA}_+} [\text{Ca}^{2+}]_{\text{cyt}} [\text{EGTA}] + k_{\text{EGTA}_-} [\text{EGTA-Ca}], \\
#     \frac{\partial [\text{EGTA}]}{\partial t} &= D_{\text{EGTA}_{\text{free}}} \Delta [\text{EGTA}] - k_{\text{EGTA}_+} [\text{Ca}^{2+}]_{\text{cyt}} [\text{EGTA}] + k_{\text{EGTA}_-} [\text{EGTA-Ca}], \\
#     \frac{\partial [\text{EGTA-Ca}]}{\partial t} &= D_{\text{EGTA}_{\text{bound}}} \Delta [\text{Ca}^{2+}]_{\text{cyt}} + k_{\text{EGTA}_+} [\text{Ca}^{2+}]_{\text{cyt}} [\text{EGTA}] - k_{\text{EGTA}_-} [\text{EGTA-Ca}].
# \end{aligned}
#
#
#
# *Table: Parameter values*
#
# | Parameter               | Meaning                                   | Value                               | 
# |-------------------------|-------------------------------------------|-------------------------------------|
# | D$^{ext}_{Ca}$      | Diffusion of extracellular Ca$^{2+}$          | 600 μm²/s                           |
# | D$^{int}_{Ca}$      | Diffusion of intracellular Ca$^{2+}$          | 220 μm²/s                           |
# | D$^{free}_{EGTA}$   | Diffusion of free EGTA                        | 113 μm²/s                           | 
# | D$^{bound}_{EGTA}$  | Diffusion of EGTA bound to Ca$^{2+}$          | 113 μm²/s                           |
# | D$^{free}_{BAPTA}$  | Diffusion of free BAPTA                       | 95 μm²/s                            |
# | D$^{bound}_{BAPTA}$ | Diffusion of BAPTA bound to Ca$^{2+}$         | 95 μm²/s                            | 
# | k$^{+}_{EGTA}$      | Forward rate constant for EGTA                | 2.7 μM⁻¹·s⁻¹                        | 
# | k$^{-}_{EGTA}$      | Reverse rate constant for EGTA                | 0.5 s⁻¹                             | 
# | k$^{+}_{BAPTA}$     | Forward rate constant for BAPTA               | 450 μM⁻¹·s⁻¹                        |
# | k$^{-}_{BAPTA}$     | Reverse rate constant for BAPTA               | 80 s⁻¹                              |
#
# *Table: Initial conditions*
#
# | Specie        | Value                          |
# |---------------|--------------------------------|
# | $\text{Ca}^{2+}_{\text{ext}}$ | 15 mM          |
# | $\text{Ca}^{2+}_{\text{cyt}}$ | 0.0001 mM      |
# | EGTA          | 4.5 mM / 40 mM                 |
# | BAPTA         | 1 mM                           |
#
#
# Dimensions:
#
# ECS= 3 μm x 3 μm x 0.1 μm
#
# cytosol = 3 μm x 3 μm x 3 μm
#
# channel$_{radius}$ = 5 nm
#
#
# * Ca can diffuse from the ECS to the cytosol through the channel.
# * The dynamics of the system are resolved in time.

# %%
from math import ceil
import csv

import matplotlib.pyplot as plt
from ngsolve import GridFunction, TaskManager, SetNumThreads
from ngsolve.webgui import Draw
from tqdm.notebook import trange
from astropy import units as u

from ecsim.geometry import create_ca_depletion_mesh, LineEvaluator
from ecsim.simulation import Simulation

# %%
# Create meshed geometry
mesh = create_ca_depletion_mesh(
    side_length_x = 2 * u.um,
    side_length_y = 1 * u.um,
    #side_length=3 * u.um,
    ecs_height=50 * u.nm,
    mesh_size=0.25 * u.um,
    channel_radius=25 * u.nm
)

# %%
# Set up a simulation on the mesh with BAPTA as a buffer
simulation = Simulation(mesh, time_step=1 * u.us, t_end=20 * u.ms)
calcium = simulation.add_species(
    "calcium",
    diffusivity={"ecs": 600 * u.um**2 / u.s, "cytosol": 220 * u.um**2 / u.s},
    #clamp={"ecs_top": 15 * u.mmol / u.L}
)
free_buffer = simulation.add_species(
    "free_buffer",
    #diffusivity={"cytosol": 95 * u.um**2 / u.s} #BAPTA
    diffusivity={"cytosol": 113 * u.um**2 / u.s} #EGTA
)
bound_buffer = simulation.add_species(
    "bound_buffer",
    #diffusivity={"cytosol": 95 * u.um**2 / u.s} #BAPTA
    diffusivity={"cytosol": 113 * u.um**2 / u.s} #EGTA
)
simulation.add_reaction(
    reactants=(calcium, free_buffer),
    products=bound_buffer,
    #kf={"cytosol": 450 / (u.s * u.umol / u.L)}, #BAPTA
    #kr={"cytosol": 80 / u.s} #BAPTA
    kf={"cytosol": 2.7 / (u.s * u.umol / u.L)}, #EGTA
    kr={"cytosol": 0.5 / u.s} #EGTA
)
simulation.add_channel_flux(
    left="ecs",
    right="cytosol",
    boundary="channel",
    rate = 0.65 * u.um / u.ms
    # old values:
    # rate= 7.04e+03 * u.mmol / (u.L * u.s)  # VOLUME 1
    # rate= 1.86e+09 * u.mmol / (u.L * u.s) #VOLUME 2
)

# %%
# Internally set up all finite element infrastructure
simulation.setup_problem()


# %%
# Time stepping - define a function that pre-computes all timesteps
def time_stepping(simulation, n_samples):
    sample_int = int(ceil(simulation.n_time_steps / n_samples))
    u_ca_t = GridFunction(simulation._fes, multidim=0)
    u_buf_t = GridFunction(simulation._fes, multidim=0)
    u_com_t = GridFunction(simulation._fes, multidim=0)
    u_ca_t.AddMultiDimComponent(simulation.concentrations["calcium"].vec)
    u_buf_t.AddMultiDimComponent(simulation.concentrations["free_buffer"].vec)
    u_com_t.AddMultiDimComponent(simulation.concentrations["bound_buffer"].vec)
    
    for i in trange(simulation.n_time_steps):
        simulation.time_step()
        if i % sample_int == 0:
            u_ca_t.AddMultiDimComponent(simulation.concentrations["calcium"].vec)
            u_buf_t.AddMultiDimComponent(simulation.concentrations["free_buffer"].vec)
            u_com_t.AddMultiDimComponent(simulation.concentrations["bound_buffer"].vec)
    return u_ca_t, u_buf_t, u_com_t


# %%
# Time stepping - set initial conditions and do time stepping
SetNumThreads(12)
with TaskManager():
    simulation.init_concentrations(
        calcium={"ecs": 1.5 * u.mmol / u.L, "cytosol": 0.1 * u.umol / u.L},
        #free_buffer={"cytosol": 1 * u.mmol / u.L}, # BAPTA
        free_buffer={"cytosol": 4.5 * u.mmol / u.L}, # low EGTA
        #free_buffer={"cytosol": 40 * u.mmol / u.L}, # high EGTA
        bound_buffer={"cytosol": 0 * u.mmol / u.L}
    )
    ca_t, buffer_t, complex_t = time_stepping(simulation, n_samples=100)

# %%
# Visualize (because of the product structure of the FESpace, the usual
# visualization of time-dependent functions via multidim is not possible)
visualization = mesh.MaterialCF({"ecs": ca_t.components[0], "cytosol": ca_t.components[1]})
clipping = {"function": True,  "pnt": (0, 0, 1.5), "vec": (0, 1, 0)}
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}, "Colormap": {"ncolors": 32, "autoscale": False, "max": 15}}
Draw(ca_t.components[1], clipping=clipping, settings=settings, interpolate_multidim=True, animate=True)

# %%
import numpy as np
line_evaluator_ecs = LineEvaluator(
    mesh,
    (0.0, 0.0, 0.1005),  # Start point (x, y, z)
    (2.0, 0.0, 0.1005),  # End point (x, y, z)
    120  # Number of points to evaluate
)

# Evaluate the concentration in the cytosol
step = 100
ca_ecs_total = [Integrate(ca_t.components[0].MDComponent(step), simulation.mesh, definedon=mesh.Material("ecs")) for step in range(100)]
# ca_cyt = np.array([line_evaluator_cyt.evaluate(ca_t.components[1].MDComponent(step)) for step in range(101)])
# #buffer_cyt = line_evaluator_cyt.evaluate(buffer_t.components[1].MDComponent(step))
# buffer_cyt = np.array([line_evaluator_cyt.evaluate(buffer_t.components[1].MDComponent(step))for step in range(101)])
# #complex_cyt = line_evaluator_cyt.evaluate(complex_t.components[1].MDComponent(step))
# complex_cyt = np.array([line_evaluator_cyt.evaluate(complex_t.components[1].MDComponent(step))for step in range(101)])

# Get the x-coordinates for the plot
x_coords_ecs = line_evaluator_ecs.raw_points[:, 0]  # Extract the x-coordinates

# %%
n_samples=100
sample_interval = int(ceil(simulation.n_time_steps / n_samples))
sample_interval


# %%
import numpy as np

# Calcular el intervalo de muestreo (sample_interval) y convertir time_step a segundos
sample_interval = int(ceil(simulation.n_time_steps / 100))  # 100 es el número de muestras actuales (n_samples)
time_step_size_seconds = float(simulation._time_step_size) * 1e-3  # Convertir de milisegundos a segundos

# Calcular el vector de tiempo ajustado usando el intervalo de muestreo
# Cada paso en `ca_cyt` corresponde a sample_interval pasos en la simulación original
time_vector = np.arange(ca_cyt.shape[0]) * sample_interval * time_step_size_seconds

# Imprimir el vector de tiempo ajustado
print(f"Vector de tiempo ajustado:\n{time_vector}")


# %%
import matplotlib.pyplot as plt

# Crear una figura para la gráfica
plt.figure(figsize=(10, 6))

# Graficar cada línea de `ca_cyt` con respecto a `time_vector`
# Seleccionamos 10 curvas de `ca_cyt` distribuidas uniformemente
num_samples_to_plot = 10  # Número de curvas que quieres graficar
indices_to_plot = np.linspace(0, ca_cyt.shape[1] - 1, num_samples_to_plot, dtype=int)  # Índices de las columnas a graficar

# Graficar solo las concentraciones correspondientes a los índices seleccionados
for i in indices_to_plot:
    plt.plot(time_vector, ca_cyt[:, i], marker='o', linestyle='-', label=f'x = {x_coords[i]:.3f} µm')  # Graficar cada columna seleccionada de `ca_cyt`

# Configurar etiquetas y título
plt.title("Calcium Concentration vs Time")
plt.xlabel("Time (s)")  # Eje X representa el tiempo en segundos
plt.ylabel("Calcium Concentration (mM)")  # Eje Y representa la concentración de calcio
plt.legend(loc='upper right')  # Mostrar la leyenda en la esquina superior derecha
plt.grid(True)  # Activar cuadrícula
plt.show()  # Mostrar la gráfica


# %%
# Create a line evaluator that evaluates a line in the extracellular space (ECS)
line_evaluator_ecs = LineEvaluator(
    mesh,
    (0.0, 0.0, 0.3005),  # Start point (x, y, z)
    (0.06, 0.0, 0.3005),  # End point (x, y, z)
    120  # Number of points to evaluate
)

# Evaluate the concentration in the extracellular space (ECS)
#concentrations_ecs = line_evaluator_ecs.evaluate(simulation.concentrations["calcium"].components[0])
concentrations_ecs = np.array([line_evaluator_ecs.evaluate(simulation.concentrations["calcium"].components[0])for step in range(101)])

# Get the x-coordinates for the plot
x_coords_ecs = line_evaluator_ecs.raw_points[:, 0]

# %%
# print(f"Último valor del vector de tiempo ajustado: {time_vector[-1]} segundos")  # Debe ser aproximadamente 0.02


# %%

# %%
import csv

# Nombre del archivo CSV donde se guardarán los datos
output_filename = 'ca_cyt_concentrations_vs_time_300nm_300nm_200nm.csv'

# Crear la cabecera para el archivo CSV
header = ['Time (s)']  # Primera columna será el tiempo
header.extend([f'x = {x:.3f} µm' for x in x_coords])  # Agregar columnas para cada coordenada x

# Abrir un archivo CSV para escritura
with open(output_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Escribir la cabecera en el archivo
    csvwriter.writerow(header)
    
    # Escribir los datos fila por fila
    # Cada fila representa un instante de tiempo, por lo que se escribe el tiempo y todas las concentraciones correspondientes
    for i, time in enumerate(time_vector):
        row = [time]  # Agregar el tiempo como la primera entrada de la fila
        row.extend(ca_cyt[i, :])  # Agregar las concentraciones de ca_cyt en todas las posiciones x
        csvwriter.writerow(row)

print(f"Datos guardados exitosamente en {output_filename}")


# %%
# Importar numpy y matplotlib (si no están importados)
import numpy as np
import matplotlib.pyplot as plt

# Crear un array para almacenar las concentraciones en cada paso de tiempo
# Array de dimensiones: (n_time_steps, número de puntos en la línea)
concentrations_ecs = np.array([line_evaluator_ecs.evaluate(ca_t.components[0].MDComponent(step)) for step in range(101)])

# Calcular el intervalo de muestreo (sample_interval) y convertir time_step a segundos
sample_interval = int(ceil(simulation.n_time_steps / 100))  # 100 es el número de muestras actuales (n_samples)
time_step_size_seconds = float(simulation._time_step_size) * 1e-3  # Convertir de milisegundos a segundos

# Calcular el vector de tiempo ajustado usando el intervalo de muestreo
# Cada paso en `concentrations_ecs` corresponde a sample_interval pasos en la simulación original
time_vector = np.arange(concentrations_ecs.shape[0]) * sample_interval * time_step_size_seconds

# Crear una figura para la gráfica
plt.figure(figsize=(10, 6))

# Seleccionar cuántas curvas quieres graficar de `concentrations_ecs`
num_samples_to_plot = 10  # Número de curvas a graficar
indices_to_plot = np.linspace(0, concentrations_ecs.shape[1] - 1, num_samples_to_plot, dtype=int)  # Índices de columnas

# Graficar solo las concentraciones correspondientes a los índices seleccionados
for i in indices_to_plot:
    plt.plot(time_vector, concentrations_ecs[:, i], marker='o', linestyle='-', label=f'x = {x_coords_ecs[i]:.3f} µm')

# Configurar etiquetas y título
plt.title("Extracellular Calcium Concentration vs Time")
plt.xlabel("Time (s)")  # Eje X representa el tiempo en segundos
plt.ylabel("Calcium Concentration (mM)")  # Eje Y representa la concentración de calcio
plt.legend(loc='upper right')  # Mostrar la leyenda en la esquina superior derecha
plt.grid(True)  # Activar cuadrícula
plt.show()  # Mostrar la gráfica


# %%
import csv

# Nombre del archivo CSV donde se guardarán los datos
output_filename_ecs = 'ecs_calcium_concentrations_vs_time_300nm_300nm_200nm.csv'

# Crear la cabecera para el archivo CSV
header_ecs = ['Time (s)']  # Primera columna será el tiempo
header_ecs.extend([f'x = {x:.3f} µm' for x in x_coords_ecs])  # Agregar columnas para cada coordenada x

# Abrir un archivo CSV para escritura
with open(output_filename_ecs, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Escribir la cabecera en el archivo
    csvwriter.writerow(header_ecs)
    
    # Escribir los datos fila por fila
    # Cada fila representa un instante de tiempo, por lo que se escribe el tiempo y todas las concentraciones correspondientes
    for i, time in enumerate(time_vector):
        row = [time]  # Agregar el tiempo como la primera entrada de la fila
        row.extend(concentrations_ecs[i, :])  # Agregar las concentraciones de concentrations_ecs en todas las posiciones x
        csvwriter.writerow(row)

print(f"Datos guardados exitosamente en {output_filename_ecs}")


# %%
# import matplotlib.pyplot as plt

# # Crear una figura para la gráfica
# plt.figure(figsize=(10, 6))

# # Graficar cada línea de `ca_cyt` con respecto a `time_vector`
# # Seleccionamos 10 curvas de `ca_cyt` distribuidas uniformemente
# num_samples_to_plot = 10  # Número de curvas que quieres graficar
# indices_to_plot = np.linspace(0, concentrations_ecs.shape[1] - 1, num_samples_to_plot, dtype=int)  # Índices de las columnas a graficar

# # Graficar solo las concentraciones correspondientes a los índices seleccionados
# for i in indices_to_plot:
#     plt.plot(time_vector, concentrations_ecs[:, i], marker='o', linestyle='-', label=f'x = {x_coords_ecs[i]:.3f} µm')  # Graficar cada columna seleccionada de `ca_cyt`

# # Configurar etiquetas y título
# plt.title("Calcium Concentration vs Time")
# plt.xlabel("Time (s)")  # Eje X representa el tiempo en segundos
# plt.ylabel("Calcium ECS (mM)")  # Eje Y representa la concentración de calcio
# plt.legend(loc='upper right')  # Mostrar la leyenda en la esquina superior derecha
# plt.grid(True)  # Activar cuadrícula
# plt.show()  # Mostrar la gráfica

# %%
# plt.plot(time_vector, concentrations_ecs[:, 1])

# %%
# concentrations_ecs[:, 1]

# %%
# time_vector = np.arange(ca_cyt.shape[0]) * float(simulation._time_step_size) * 1e-3
# time_vector

# %%
# # Imprimir información relevante
# print(f"Tamaño del paso de tiempo (time_step): {simulation._time_step_size} ms")  # Debe ser 0.001 ms
# print(f"Tiempo total de simulación (t_end): {simulation._t_end} ms")  # Debe ser 20 ms
# print(f"Número total de pasos de tiempo calculados: {simulation.n_time_steps}")  # Debe ser 20000
# print(f"Forma de ca_cyt (número de pasos de tiempo almacenados): {ca_cyt.shape[0]}")  # Debe ser 20000
# print(f"Último valor del vector de tiempo: {time_vector[-1]} segundos")  # Debe ser 0.02 s

# # Confirmar si el vector de tiempo llega a 0.02 s
# if time_vector[-1] == 0.02:
#     print("El vector de tiempo alcanza correctamente el tiempo total de 0.02 segundos.")
# else:
#     print("El vector de tiempo no alcanza el tiempo total esperado. Revisa `ca_cyt.shape[0]` y la implementación.")


# %%
# ca_cyt.shape[0]

# %%
# total_simulation_time = simulation._t_end  # Tiempo total en ms


# %%
# total_simulation_time_seconds = total_simulation_time * 1e-3  # Convertir ms a s


# %%
# import matplotlib.pyplot as plt
# import numpy as np

# # Calcular el vector de tiempo (en segundos) basado en el número de pasos de tiempo y el tamaño del paso
# # Convertir `simulation._time_step_size` de ms a s multiplicando por 1e-3
# time_vector = np.arange(ca_cyt.shape[0]) * float(simulation._time_step_size) * 1e-3  # Vector de tiempo en segundos

# # Tiempo total de simulación en segundos
# total_simulation_time_seconds = time_vector[-1]
# print(f"Tiempo total de simulación: {total_simulation_time_seconds} segundos")

# # Crear una figura para la gráfica
# plt.figure(figsize=(10, 6))

# # Seleccionar 10 índices uniformemente espaciados a lo largo de `x_coords`
# num_samples = 10
# indices = np.linspace(0, len(x_coords) - 1, num_samples, dtype=int)  # Índices de muestreo

# # Graficar solo las concentraciones correspondientes a los índices seleccionados
# for i in indices:
#     plt.plot(time_vector, ca_cyt[:, i], marker='o', linestyle='-', label=f'x = {x_coords[i]:.3f} µm')  # Graficar cada columna seleccionada de `ca_cyt`

# # Añadir etiquetas y leyenda
# plt.title("Calcium Concentration vs Time (Sampled)")
# plt.xlabel("Time (s)")  # Tiempo en segundos
# plt.ylabel("Calcium Concentration (mM)")
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.show()


# %%
# total_simulation_time_seconds = 20 * 1e-3
# total_simulation_time_seconds


# %%
# time_vector

# %%
# print(f"Tiempo total esperado de simulación: {simulation._t_end} ms")
# print(f"Tamaño del paso de tiempo: {simulation._time_step_size} ms")
# print(f"Número total de pasos de tiempo: {simulation.n_time_steps}")
# print(f"Forma del array `ca_cyt`: {ca_cyt.shape}")


# %%
# # Superponer todas las curvas en una sola gráfica
# plt.figure(figsize=(10, 6))
# for i, ca_concentration in enumerate(ca_cyt):
#     plt.plot(x_coords, ca_concentration, marker='o', linestyle='-', label=f'Time step {i}')
# plt.title(r"Concentration vs distance from the channel over time")
# plt.xlabel(r"Distance from the channel ($\mathrm{\mu m}$)")
# plt.ylabel(r"Concentration (mM)")
# plt.legend()
# plt.grid(True)
# plt.show()


# %%
# # Graficar cada 10 pasos de tiempo para no saturar la gráfica con demasiadas curvas
# plt.figure(figsize=(10, 6))
# for i, ca_concentration in enumerate(ca_cyt):
#     if i % 10 == 0:  # Mostrar cada 10 pasos de tiempo
#         plt.plot(x_coords, ca_concentration, marker='o', linestyle='-', label=f'Time step {i}')
# plt.title(r"Concentration vs distance from the channel over time")
# plt.xlabel(r"Distance from the channel ($\mathrm{\mu m}$)")
# plt.ylabel(r"Concentration (mM)")
# plt.legend()
# plt.grid(True)
# plt.show()


# %%
# time_vector = np.arange(ca_cyt.shape[0]) * float(simulation._time_step_size)

# %%
# import matplotlib.pyplot as plt
# import numpy as np

# # Crear una figura
# plt.figure(figsize=(10, 6))

# # Seleccionar 10 índices uniformemente espaciados a lo largo de x_coords
# num_samples = 10
# indices = np.linspace(0, len(x_coords) - 1, num_samples, dtype=int)  # Índices de muestreo

# # Graficar solo las concentraciones correspondientes a los índices seleccionados
# for i in indices:
#     plt.plot(time_vector, ca_cyt[:, i],  marker='o', linestyle='-', label=f'x = {x_coords[i]:.3f} µm')  # Graficar cada columna seleccionada de `ca_cyt`

# # Añadir etiquetas y leyenda
# plt.title("Calcium Concentration vs Time (Sampled)")
# plt.xlabel("Time (s)")
# plt.ylabel("Calcium Concentration (mM)")
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.show()


# %%
# time_vector

# %%
# import matplotlib.pyplot as plt

# # Crear una figura
# plt.figure(figsize=(10, 6))

# # Graficar la concentración a lo largo del tiempo para cada punto en `x_coords`
# for i, x in enumerate(x_coords):
#     plt.plot(time_vector, ca_cyt[:, i], label=f'x = {x:.3f} µm')  # Graficar cada columna de `ca_cyt` a lo largo del tiempo

# # Añadir etiquetas y leyenda
# plt.title("Calcium Concentration vs Time")
# plt.xlabel("Time (s)")
# plt.ylabel("Calcium Concentration (mM)")
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.show()


# %%
# # Create a line evaluator that evaluates a line away from the channel in the cytosol
# import numpy as np
# line_evaluator_cyt = LineEvaluator(
#     mesh,
#     (0.0, 0.0, 2.995),  # Start point (x, y, z)
#     (0.6, 0.0, 2.995),  # End point (x, y, z)
#     120  # Number of points to evaluate
# )

# # Evaluate the concentration in the cytosol
# step = 100
# ca_cyt = np.array([line_evaluator_cyt.evaluate(ca_t.components[1].MDComponent(step)) for step in range(101)])
# buffer_cyt = line_evaluator_cyt.evaluate(buffer_t.components[1].MDComponent(step))
# complex_cyt = line_evaluator_cyt.evaluate(complex_t.components[1].MDComponent(step))

# # Get the x-coordinates for the plot
# x_coords = line_evaluator_cyt.raw_points[:, 0]  # Extract the x-coordinates

# # Plot the results
# plt.plot(x_coords, ca_cyt, marker='o', linestyle='-', color='red', label='$[\mathrm{Ca}^{2+}]_{\mathrm{cyt}}$')
# plt.plot(x_coords, buffer_cyt, marker='x', linestyle='-', color='blue', label='$[\mathrm{BAPTA}]_{\mathrm{cyt}}$')
# plt.plot(x_coords, complex_cyt, marker='s', linestyle='-', color='magenta', label='$[\mathrm{CaBAPTA}]_{\mathrm{cyt}}$')
# plt.title(r"Concentration vs distance from the channel")
# plt.xlabel(r"Distance from the channel ($\mathrm{\mu m}$)")
# plt.ylabel(r"Concentration (mM)")
# plt.legend()
# plt.grid(True)
# plt.show()

# %%
# Save the values in a CSV file
# with open('calcium_concentrations_cyt_egta_high_vol2.csv', 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(['x_coords', 'concentrations_cyt'])
#     for x, conc in zip(x_coords, concentrations_cyt):
#         csvwriter.writerow([x, conc])

# %%
# # Create a line evaluator that evaluates a line in the extracellular space (ECS)
# line_evaluator_ecs = LineEvaluator(
#     mesh,
#     (0.0, 0.0, 3.005),  # Start point (x, y, z)
#     (0.6, 0.0, 3.005),  # End point (x, y, z)
#     120  # Number of points to evaluate
# )

# # Evaluate the concentration in the extracellular space (ECS)
# concentrations_ecs = line_evaluator_ecs.evaluate(simulation.concentrations["calcium"].components[0])

# # Get the x-coordinates for the plot
# x_coords_ecs = line_evaluator_ecs.raw_points[:, 0]

# # Plot the results
# plt.plot(x_coords_ecs, concentrations_ecs, marker='o', linestyle='-', color='red')
# plt.title(r"$[\mathrm{Ca}^{2+}]_{\mathrm{ecs}}$ vs Distance from the channel")
# plt.xlabel(r"Distance from the channel ($\mathrm{\mu m}$)")
# plt.ylabel(r"$[\mathrm{Ca}^{2+}]_{\mathrm{ecs}}$ (mM)")
# plt.grid(True)
# plt.show()

# %%
# # Save cytosolic data to a CSV file
# with open('cytosolic_concentrations_egta_4p5mM_rate_0p65_reflective_10nm_with_time.csv', 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     # Write header
#     csvwriter.writerow(['x_coords', 'ca_cyt', 'buffer_cyt', 'complex_cyt'])
#     # Write data
#     for x, ca, buf, com in zip(x_coords, ca_cyt, buffer_cyt, complex_cyt):
#         csvwriter.writerow([x, ca, buf, com])


# %%
# # Save ECS data to a CSV file
# with open('ecs_concentrations_egta_4p5mM_rate_0p65_reflective_10nm_with_time.csv', 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     # Write header
#     csvwriter.writerow(['x_coords_ecs', 'concentrations_ecs'])
#     # Write data
#     for x_ecs, conc_ecs in zip(x_coords_ecs, concentrations_ecs):
#         csvwriter.writerow([x_ecs, conc_ecs])


# %%
# # %% [markdown]
# # Agregar el siguiente bloque de código al final de tu script `dynamics.py`
# # para calcular y graficar `time vs Ca_cyt`

# # Crear una lista para almacenar el tiempo y otra para la concentración promedio de Ca_cyt
# time_values = []
# ca_cyt_values = []

# # Calcular el vector de tiempo y la concentración promedio de Ca_cyt en cada muestra
# for step in range(100):  # Aquí 100 es el número de muestras tomadas en la simulación
#     current_time = step * simulation._time_step_size.to(u.ms).value
#     time_values.append(current_time)
    
#     # Evaluar la concentración de calcio en el citosol en cada paso de tiempo
#     ca_cyt = line_evaluator_cyt.evaluate(ca_t.components[1].MDComponent(step))
#     average_ca_cyt = ca_cyt.mean()
#     ca_cyt_values.append(average_ca_cyt)

# # Graficar el tiempo vs la concentración de Ca_cyt
# plt.figure(figsize=(8, 5))
# plt.plot(time_values, ca_cyt_values, marker='o', linestyle='-', color='red', label='$[\mathrm{Ca}^{2+}]_{\mathrm{cyt}}$')
# plt.title("Concentration of Ca2+ in Cytosol Over Time")
# plt.xlabel("Time (ms)")
# plt.ylabel(r"Concentration of $\mathrm{Ca}^{2+}$ (mM)")
# plt.grid(True)
# plt.legend()
# plt.show()




# %%
# # Guardar los resultados en un archivo CSV
# with open('time_vs_ca_cyt.csv', 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(['time (ms)', 'Ca_cyt (mM)'])
#     for t, ca in zip(time_values, ca_cyt_values):
#         csvwriter.writerow([t, ca])
