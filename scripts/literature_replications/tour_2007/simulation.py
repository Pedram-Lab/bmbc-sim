"""Tour et al. (2007), "Calcium Green FlAsH as a genetically targeted
small-molecule calcium indicator".

Calcium is transported from the extracellular space into the cytosol through a
cluster of calcium channels, where it binds one of three buffers. Each buffer is
a config variant in ``configs/`` -- the initial concentration, diffusivity and
reaction rates all belong to the same buffer, so switching one switches all:

    uv run scripts/literature_replications/tour_2007/simulation.py --config-name egta_low
    uv run scripts/literature_replications/tour_2007/simulation.py --config-name egta_high
    uv run scripts/literature_replications/tour_2007/simulation.py --config-name bapta

``evaluation.py`` plots all three side by side.
"""
import astropy.units as u

import bmbcsim
from bmbcsim.geometry import create_ca_depletion_mesh
from bmbcsim.simulation import transport
from bmbcsim.config import (
    ConfigGroup,
    Quantity,
    SimulationConfig,
    dump_resolved,
    run_from_cli,
)


class Buffer(ConfigGroup):
    """The cytosolic Ca2+ buffer; ``name`` also labels the result directory."""

    name: str = "EGTA_low"
    initial_concentration: Quantity("mM") = "4.5 mM"
    diffusivity: Quantity("um2 / s") = "113 um2 / s"
    kf: Quantity("1 / (uM s)") = "2.7 / (uM s)"
    kr: Quantity("1 / s") = "0.5 / s"


class Geometry(ConfigGroup):
    """Slab of cytosol under a thin extracellular sheet, pierced by one channel."""

    side: Quantity("um") = "3.0 um"
    cytosol_height: Quantity("um") = "3.0 um"
    ecs_height: Quantity("um") = "0.1 um"
    channel_radius: Quantity("nm") = "50 nm"
    mesh_size: Quantity("nm") = "100 nm"


class Config(SimulationConfig):
    """Full config for the Tour et al. replication."""

    simulation_name: str = "tour"
    buffer: Buffer = Buffer()
    geometry: Geometry = Geometry()
    # Calcium
    ca_ecs: Quantity("mM") = "15 mM"  # also the concentration outside ecs_top
    ca_cytosol: Quantity("uM") = "0.1 uM"
    diffusivity_ecs: Quantity("um2 / s") = "600 um2 / s"
    diffusivity_cytosol: Quantity("um2 / s") = "220 um2 / s"
    channel_rate: Quantity("um / ms") = "10 um / ms"
    # Timing
    end_time: Quantity("ms") = "20 ms"
    time_step: Quantity("ms") = "1 us"
    record_interval: Quantity("ms") = "1 ms"

    def derived_postfix(self) -> str:
        """The buffer *is* the variant here (see configs/), so it names the run."""
        return self.buffer.name.lower()


def run(cfg: Config) -> None:
    """Run the simulation from a validated config."""
    geom, buffer_cfg = cfg.geometry, cfg.buffer
    result_dir = bmbcsim.timestamped_directory(cfg.result_root, cfg.run_name)
    dump_resolved(cfg, result_dir)
    print(f"Results and config -> {result_dir}")

    mesh = create_ca_depletion_mesh(
        side_length_x=geom.side,
        side_length_y=geom.side,
        cytosol_height=geom.cytosol_height,
        ecs_height=geom.ecs_height,
        channel_radius=geom.channel_radius,
        mesh_size=geom.mesh_size,
        channel_mesh_size=geom.channel_radius,
    )

    simulation = bmbcsim.Simulation(mesh, result_directory=result_dir)
    geometry = simulation.simulation_geometry

    ecs = geometry.compartments["ecs"]
    cytosol = geometry.compartments["cytosol"]
    channel = geometry.membranes["channel"]
    ecs_top = geometry.membranes["ecs_top"]

    # Species
    ca = simulation.add_species("Ca", valence=2)
    buffer = simulation.add_species(buffer_cfg.name, valence=-2)
    ca_buffer = simulation.add_species(f"Ca_{buffer_cfg.name}", valence=0)

    # Initial conditions
    ecs.initialize_species(ca, value=cfg.ca_ecs)
    cytosol.initialize_species(ca, value=cfg.ca_cytosol)
    cytosol.initialize_species(buffer, value=buffer_cfg.initial_concentration)
    cytosol.initialize_species(ca_buffer, value=0 * u.umol / u.L)

    # Diffusion
    ecs.add_diffusion(ca, diffusivity=cfg.diffusivity_ecs)
    cytosol.add_diffusion(ca, diffusivity=cfg.diffusivity_cytosol)
    cytosol.add_diffusion(buffer, diffusivity=buffer_cfg.diffusivity)
    cytosol.add_diffusion(ca_buffer, diffusivity=buffer_cfg.diffusivity)

    # Reaction Ca + Buffer <=> CaBuffer
    cytosol.add_reaction(
        reactants=[ca, buffer],
        products=[ca_buffer],
        k_f=buffer_cfg.kf,
        k_r=buffer_cfg.kr,
    )

    # Transport: into the cytosol through the channel, and out of the top of the
    # ECS sheet, which is held at the bulk extracellular concentration.
    permeability = cfg.channel_rate * channel.area
    channel.add_transport(
        species=ca,
        transport=transport.Passive(permeability=permeability),
        source=ecs,
        target=cytosol,
    )
    ecs_top.add_transport(
        species=ca,
        transport=transport.Passive(
            permeability=permeability, outside_concentration=cfg.ca_ecs
        ),
        source=ecs,
        target=None,
    )

    simulation.run(
        end_time=cfg.end_time,
        time_step=cfg.time_step,
        record_interval=cfg.record_interval,
        n_threads=cfg.n_threads,
    )


if __name__ == "__main__":
    run_from_cli(Config, run, __file__)
