import astropy.units as u
import ngsolve as ngs

from bmbcsim.simulation.geometry.compartment import Compartment
from bmbcsim.simulation.transport.transport import Transport
from bmbcsim.simulation.simulation_agents import ChemicalSpecies
from bmbcsim.units import BASE_UNITS


class Membrane:
    """A membrane is a boundary between two compartments in the simulation
    geometry. It can have different properties on the two sides and can be
    used to model diffusion and reactions that occur at the boundary.
    """
    def __init__(
            self,
            name: str,
            mesh: ngs.Mesh,
            connects: set[tuple[Compartment, Compartment]],
            area: float
    ):
        """Create a new membrane.

        :param mesh: NGSolve mesh object that represents the geometry.
        :param name: Name of the membrane.
        :param connects: List of tuples containing the compartments that this membrane connects.
        :param area: Area of the membrane.
        """
        self._mesh = mesh
        self.name = name
        self.connects = connects
        self._area_parameter = ngs.Parameter(area)
        self._transport = []

    @property
    def area(self) -> u.Quantity:
        """Get the area of the membrane."""
        return self._area_parameter.Get() * BASE_UNITS['length'] ** 2


    def add_transport(
            self,
            species: ChemicalSpecies,
            transport: Transport,
            source: Compartment | None,
            target: Compartment | None
    ) -> None:
        """Add a transport mechanism to the membrane, transporting material from
        the source to the target compartment. Either of the compartments can be
        None if the membrane is a boundary.

        :param species: The chemical species to which the transport applies.
        :param transport: The transport mechanism to apply.
        :param source: The compartment from which the species is transported.
        :param target: The compartment to which the species is transported.
        """
        if source is None and target is None:
            raise ValueError("At least one of source or target must be a compartment.")
        elif source is None and self.neighbor(target) is not None:
            raise ValueError("'{target}' is not a boundary compartment.")
        elif target is None and self.neighbor(source) is not None:
            raise ValueError("'{source}' is not a boundary compartment.")
        elif source is not None and target is not None:
            other = self.neighbor(source)
            if target != other:
                raise ValueError(f"Membrane {self.name} does not connect "
                                 f"{source.name} and {target.name}.")

        # Add the transport to the respective compartments
        self._transport.append((species, source, target, transport))


    def neighbor(self, compartment: Compartment) -> Compartment:
        """Get the compartment on the opposite side of the membrane.

        :param compartment: The compartment on one side of the membrane.
        :return: The compartment on the opposite side of the membrane.
        """
        for left, right in self.connects:
            if compartment == left:
                return right
            elif compartment == right:
                return left
        raise ValueError(f"Compartment '{compartment.name}' "
                         f"is not connected to membrane '{self.name}'.")


    def get_transport(self) -> tuple[ChemicalSpecies, Compartment, Compartment, Transport]:
        """Get the transport mechanisms associated with this membrane.

        :return: A list of (species, source compartment, target compartment, transport).
        """
        return self._transport

    def __str__(self) -> str:
        return f"Membrane {self.name} connecting {[comp.name for comp in self.connects]}"


    def __repr__(self) -> str:
        return f"Membrane(name={self.name}, connects={self.connects}, area={self.area})"


    def __eq__(self, value):
        if not isinstance(value, Membrane):
            return False
        return self.name == value.name and self.connects == value.connects


    def __hash__(self):
        return hash((self.name, tuple(self.connects)))
