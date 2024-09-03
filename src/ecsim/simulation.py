from dataclasses import dataclass
from typing import Dict, Iterable

import astropy.units as au
from ngsolve import Mesh, FESpace, H1, Compress, VOL, BND, BilinearForm, LinearForm, grad, dx, ds, GridFunction, CoefficientFunction
from pyngcore import BitArray


# Define common units for the simulation
LENGTH_UNIT = au.um
TIME_UNIT = au.s
CONCENTRATION_UNIT = au.amol / au.um ** 3  # equivalent to mM = mmol / L


@dataclass
class ChemicalSpecies:
    name: str
    diffusivity: Dict[str, au.Quantity]
    clamp: Iterable[str]
    valence: int

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

class Simulation:
    def __init__(self, mesh: Mesh, time_step: au.Quantity, order: int = 2):
        self.mesh = mesh
        self._time_step_size = time_step
        self._compartments = {name: i for i, name in enumerate(mesh.GetMaterials())}
        self._fes = FESpace([Compress(H1(mesh, order=order, definedon=mesh.Materials(name))) for name in mesh.GetMaterials()])
        self._species = {}
        self._reactions = []
        self._channels = []
        self._diffusion_matrix = {}
        self._time_stepping_matrix = {}
        self._source_terms = {}
        self.concentrations = {}

    def add_species(
            self,
            name: str,
            *,
            diffusivity: Dict[str, au.Quantity],
            valence: int = 0,
            clamp: str | Iterable[str] = None
    ) -> ChemicalSpecies:
        """
        Add a new :class:`ChemicalSpecies` to the simulation.
        :param name: Name of the species.
        :param diffusivity: Diffusivity in different compartments; if not given, the species is not present in the compartment.
        :param valence: Valence (electrical charge) of the species.
        :param clamp: Clamp concentration to initial value on the given boundaries.
        :return:
        """
        if name in self._species:
            raise ValueError(f"Species {name} already exists.") from None
        for compartment in diffusivity:
            if compartment not in self._compartments:
                raise ValueError(f"Compartment {compartment} not found in the mesh.")
        clamp = clamp or []
        if not isinstance(clamp, Iterable):
            clamp = [clamp]
        for boundary in clamp:
            if boundary not in self.mesh.GetBoundaries():
                raise ValueError(f"Boundary {boundary} not found in the mesh.")

        self._species[name] = ChemicalSpecies(name, diffusivity, clamp, valence)
        return self._species[name]

    def add_reaction(
            self,
            *,
            reactants: ChemicalSpecies | Iterable[ChemicalSpecies],
            products: ChemicalSpecies | Iterable[ChemicalSpecies],
            kf: Dict[str, au.Quantity],
            kr: Dict[str, au.Quantity]
    ):
        """
        Add a reaction r_1 + r_2 + ... <-> p_1 + p_2 + ... with forward and reverse rate constants.
        :param reactants: Participating :class:`ChemicalSpecies` in the reaction.
        :param products: Products of the reaction.
        :param kf: Forward rate constant per compartment.
        :param kr: Reverse rate constant per compartment.
        """
        if not isinstance(reactants, Iterable):
            reactants = [reactants]
        if not isinstance(products, Iterable):
            products = [products]
        reaction = Reaction(reactants, products, kf, kr)
        # TODO: check if compartments exist and all species are present in the compartments
        self._reactions.append(reaction)

    def add_channel_flux(self, left: str, right: str, boundary: str, rate: au.Quantity):
        """
        Add a channel flux between two compartments.
        :param left: Name of the left compartment.
        :param right: Name of the right compartment.
        :param boundary: Name of the boundary where the flux occurs.
        :param rate: Rate of the flux.
        """
        if left not in self._compartments:
            raise ValueError(f"Compartment {left} not found in the mesh.")
        if right not in self._compartments:
            raise ValueError(f"Compartment {right} not found in the mesh.")
        if boundary not in self.mesh.GetBoundaries():
            raise ValueError(f"Boundary {boundary} not found in the mesh.")
        # TODO: check if the compartments are adjacent and the boundary is between them

        self._channels.append(ChannelFlux(left, right, boundary, rate))

    def setup_problem(self):
        self.concentrations = {name: GridFunction(self._fes) for name in self._species}

        for name, species in self._species.items():
                a, m_star_inv = self._setup_matrices(species)
                self._diffusion_matrix[name] = a
                self._time_stepping_matrix[name] = m_star_inv

        self._source_terms = {name: LinearForm(self._fes) for name in self._species}
        for reaction in self._reactions:
            self._add_reaction_to_source_terms(reaction)

    def _setup_matrices(self, species):
        relevant_dofs = BitArray(self._fes.ndof)
        relevant_dofs[:] = True

        for compartment in self._compartments:
            if not compartment in species.compartments:
                # If no diffusivity is given, the species is not present in this compartment (even if the concentration is zero)
                self._set_dofs(relevant_dofs, self.mesh.Region(VOL, compartment), False)
        for boundary in self.mesh.GetBoundaries():
            if boundary in species.clamp:
                self._set_dofs(relevant_dofs, self.mesh.Region(BND, boundary), False)

        # Set up diffusion and mass matrix (set check_unused=False to avoid warnings about unused DOFs)
        a = BilinearForm(self._fes, check_unused=False)
        m = BilinearForm(self._fes, check_unused=False)
        u, v = self._fes.TnT()
        for compartment, diffusivity in species.diffusivity.items():
            i = self._compartments[compartment]
            D = diffusivity.to(LENGTH_UNIT ** 2 / TIME_UNIT).value
            a += D * grad(u[i]) * grad(v[i]) * dx(compartment)
            m += u[i] * v[i] * dx(compartment)

        for channel in self._channels:
            i = self._compartments[channel.left]
            j = self._compartments[channel.right]
            rate = channel.rate.to(CONCENTRATION_UNIT / TIME_UNIT).value
            a += rate * (u[i] - u[j]) * (v[i] - v[j]) * ds(channel.boundary)

        a.Assemble()
        m.Assemble()

        dt = self._time_step_size.to(TIME_UNIT).value
        m.mat.AsVector().data += dt * a.mat.AsVector()

        return a, m.mat.Inverse(relevant_dofs)

    def _set_dofs(self, dof_array, region, value):
        for el in region.Elements():
            for dof in self._fes.GetDofNrs(el):
                dof_array[dof] = value

    def _add_reaction_to_source_terms(self, reaction):
        _, v = self._fes.TnT()
        for compartment in reaction.kf:
            i = self._compartments[compartment]
            kf = reaction.kf[compartment].to(CONCENTRATION_UNIT / TIME_UNIT).value

            # TODO: find a better default value (or skip if reactants / products are not present in the compartment)
            forward_reaction = CoefficientFunction(1.0)
            for reactant in reaction.reactants:
                forward_reaction *= self.concentrations[reactant.name].components[i]
            for reactant in reaction.reactants:
                self._source_terms[reactant.name] += -kf * forward_reaction.Compile() * v[i] * dx(compartment)
            for product in reaction.products:
                self._source_terms[product.name] += kf * forward_reaction.Compile() * v[i] * dx(compartment)

        for compartment in reaction.kr:
            i = self._compartments[compartment]
            kr = reaction.kr[compartment].to(1 / TIME_UNIT).value

            reverse_reaction = CoefficientFunction(1.0)
            for product in reaction.products:
                reverse_reaction *= self.concentrations[product.name].components[i]
            for reactant in reaction.reactants:
                self._source_terms[reactant.name] += kr * reverse_reaction.Compile() * v[i] * dx(compartment)
            for product in reaction.products:
                self._source_terms[product.name] += -kr * reverse_reaction.Compile() * v[i] * dx(compartment)

    def time_step(self):
        residual = {}
        dt = self._time_step_size.to(TIME_UNIT).value
        for name, f in self._source_terms.items():
            f.Assemble()
            a = self._diffusion_matrix[name]
            u = self.concentrations[name]
            residual[name] = dt * (f.vec - a.mat * u.vec)
        for name, u in self.concentrations.items():
            u.vec.data += self._time_stepping_matrix[name] * residual[name]

    def init_concentrations(self, **initial_concentrations):
        for name, values in initial_concentrations.items():
            for compartment, value in values.items():
                i = self._compartments[compartment]
                v = value.to(CONCENTRATION_UNIT).value
                self.concentrations[name].components[i].Set(v)