import astropy.units as u
import bmbcsim

mesh = bmbcsim.geometry.create_sphere_geometry(radius=10 * u.um, mesh_size=1 * u.um)
sim = bmbcsim.Simulation("demo", mesh, result_root="results")
cell = sim.simulation_geometry.compartments["sphere"]


ca = sim.add_species("ca")
cell.add_diffusion(ca, diffusivity=0.2 * u.um**2 / u.ms)
buffer = sim.add_species("buffer")
cell.add_diffusion(buffer, diffusivity=0.1 * u.um**2 / u.ms)
ca_buffer = sim.add_species("ca_buffer")
cell.add_diffusion(ca_buffer, diffusivity=0.05 * u.um**2 / u.ms)
cell.add_reaction(
    reactants=[ca, buffer], products=[ca_buffer],
    k_f=0.1 / (u.ms * u.um**3), k_r=0.05 / u.ms,
)

sim.run(end_time=10.0 * u.ms, time_step=0.1 * u.ms, record_interval=1.0 * u.ms)
