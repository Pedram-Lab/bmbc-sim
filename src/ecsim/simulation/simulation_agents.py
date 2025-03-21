from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class ChemicalSpecies:
    """A chemical species that participates in a simulation.
    """
    name: str
    valence: int

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise ValueError(f"Species name must be a string, got {type(self.name)}")
        if not isinstance(self.valence, int):
            raise ValueError(f"Species valence must be an integer, got {type(self.valence)}")
