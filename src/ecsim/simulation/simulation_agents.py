from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class ChemicalSpecies:
    """A chemical species that participates in a simulation.
    """
    name: str
    valence: int
