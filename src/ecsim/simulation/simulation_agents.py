from dataclasses import dataclass
from typing import Dict, Iterable

import astropy.units as au


@dataclass
class ChemicalSpecies:
    name: str
    diffusivity: Dict[str, au.Quantity]
    clamp: Iterable[str]
    # TODO: charge, ...?

    @property
    def compartments(self):
        return self.diffusivity.keys()


@dataclass
class Reaction:
    reactants: Iterable[ChemicalSpecies]
    products: Iterable[ChemicalSpecies]
    kf: Dict[str, au.Quantity]
    kr: Dict[str, au.Quantity]


@dataclass
class ChannelFlux:
    left: str
    right: str
    boundary: str
    rate: au.Quantity
