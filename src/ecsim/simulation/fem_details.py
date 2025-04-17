import ngsolve as ngs
import astropy.units as u
import astropy.constants as const

from ecsim.simulation.simulation_agents import ChemicalSpecies
from ecsim.units import to_simulation_units


class FemLhs:
    """Left-hand side of the finite element equations for a single species."""

    def __init__(self, a, transport, m_star, pre):
        self._a = a
        self._transport = transport
        self._m_star = m_star
        self._pre = pre


    @classmethod
    def for_all_species(
            cls,
            species,
            fes,
            simulation_geometry,
            concentrations,
            dt
    ) -> dict[ChemicalSpecies, 'FemLhs']:
        """Set up the left-hand side of the finite element equations for all
        species.
        """
        species_to_lhs = {}
        for s in species:
            species_to_lhs[s] = cls._for_single_species(
                s,
                fes,
                simulation_geometry,
                concentrations[s],
                dt
            )
        return species_to_lhs


    @classmethod
    def _for_single_species(
            cls,
            species,
            fes,
            simulation_geometry,
            concentration,
            dt
    ):
        """Set up the left-hand side of the finite element equations for a given species.
        """
        compartments = simulation_geometry.compartments.values()
        mass = ngs.BilinearForm(fes, check_unused=False)
        stiffness = ngs.BilinearForm(fes, check_unused=False)
        trial_and_test = list(zip(*fes.TnT()))
        compartment_to_index = {compartment: i for i, compartment in enumerate(compartments)}

        for i, compartment in enumerate(compartments):
            coefficients = compartment.coefficients
            trial, test = trial_and_test[i]

            # Set up stiffness matrix (diffusion terms)
            if species in coefficients.diffusion and \
                    (diffusivity := coefficients.diffusion[species]) is not None:
                stiffness += diffusivity * ngs.grad(trial) * ngs.grad(test) * ngs.dx

            # Set up mass matrix
            mass += trial * test * ngs.dx

        # Handle implicit transport terms
        transport_term = ngs.BilinearForm(fes, check_unused=False)
        for membrane in simulation_geometry.membranes.values():
            for s, source, target, transport in membrane.get_transport():
                if s != species:
                    continue

                def select(compartment, concentration, tnt):
                    if compartment is None:
                        return None, None, None
                    idx = compartment_to_index[compartment]
                    return concentration.components[idx], *tnt[idx]

                src_c, src_trial, src_test = select(source, concentration, trial_and_test)
                trg_c, trg_trial, trg_test = select(target, concentration, trial_and_test)

                # Calculate the flux density through the membrane
                flux = transport.flux_lhs(src_c, trg_c, src_trial, trg_trial)

                if flux is not None:
                    area = to_simulation_units(membrane.area, 'area')
                    flux_density = (flux / area).Compile()
                    ds = ngs.ds(membrane.name)
                    if src_trial is not None:
                        transport_term += -flux_density * src_test * ds
                    if trg_trial is not None:
                        transport_term += flux_density * trg_test * ds

        # Assemble the mass and stiffness matrices
        mass.Assemble()
        stiffness.Assemble()

        # Invert the mass matrix and the matrix for the implicit Euler rule
        mass.mat.AsVector().data += dt * stiffness.mat.AsVector()
        smoother = mass.mat.CreateSmoother(fes.FreeDofs())

        return cls(
            stiffness.mat,
            transport_term,
            mass.mat,
            smoother
        )


    def assemble(self) -> 'FemLhs':
        """Update the transport matrix."""
        self._transport.Assemble()
        return self


    @property
    def stiffness(self) -> ngs.BaseMatrix:
        """The stiffness matrix."""
        return self._a - self._transport.mat


    @property
    def time_stepping(self) -> ngs.BaseMatrix:
        """The matrix for the implicit Euler rule."""
        return ngs.CGSolver(self._m_star - self._transport.mat, self._pre, printrates=False)


class FemRhs:
    """Right-hand side of the finite element equations for a single species."""

    def __init__(self, source_term):
        self._source_term = source_term

    @classmethod
    def for_all_species(
            cls,
            species,
            fes,
            simulation_geometry,
            concentrations,
            potential
    ):
        """Set up the right-hand side of the finite element equations for a given species.
        """
        source_terms = {s: ngs.LinearForm(fes) for s in species}
        test_functions = fes.TestFunction()
        compartments = list(simulation_geometry.compartments.values())
        compartment_to_index = {compartment: i for i, compartment in enumerate(compartments)}

        # Handle reaction terms
        for i, compartment in enumerate(compartments):
            coefficients = compartment.coefficients
            test = test_functions[i]

            for (reactants, products), (kf, kr) in coefficients.reactions.items():
                all_reactants = ngs.CoefficientFunction(1.0)
                for reactant in reactants:
                    all_reactants *= concentrations[reactant].components[i]
                forward_reaction = (kf * all_reactants * test).Compile()
                for reactant in reactants:
                    source_terms[reactant] += -forward_reaction * ngs.dx
                for product in products:
                    source_terms[product] += forward_reaction * ngs.dx

                all_products = ngs.CoefficientFunction(1.0)
                for product in products:
                    all_products *= concentrations[product].components[i]
                reverse_reaction = (kr * all_products * test).Compile()
                for reactant in reactants:
                    source_terms[reactant] += reverse_reaction * ngs.dx
                for product in products:
                    source_terms[product] += -reverse_reaction * ngs.dx

        # Handle explicit transport terms
        for membrane in simulation_geometry.membranes.values():
            for s, source, target, transport in membrane.get_transport():
                concentration = concentrations[s]

                def select(compartment, concentration, test):
                    if compartment is None:
                        return None, None
                    idx = compartment_to_index[compartment]
                    return concentration.components[idx], test[idx]

                src_c, src_test = select(source, concentration, test_functions)
                trg_c, trg_test = select(target, concentration, test_functions)

                # Calculate the flux density through the membrane
                flux = transport.flux_rhs(src_c, trg_c)

                if flux is not None:
                    area = to_simulation_units(membrane.area, 'area')
                    flux_density = (flux / area).Compile()
                    ds = ngs.ds(membrane.name)
                    if src_test is not None:
                        source_terms[s] += -flux_density * src_test * ds
                    if trg_test is not None:
                        source_terms[s] += flux_density * trg_test * ds

        # Handle potential terms
        if potential is not None:
            beta = to_simulation_units(const.e.si / (const.k_B * 310 * u.K))
            for i, compartment in enumerate(compartments):
                for s in species:
                    if s.valence == 0:
                        continue

                    d = compartment.coefficients.diffusion[s]
                    c = concentrations[s].components[i]
                    drift = ngs.InnerProduct(ngs.grad(potential[i]), ngs.grad(test_functions[i]))
                    source_terms[s] += (d * beta * s.valence * c * drift).Compile() * ngs.dx

        return {s: cls(source_terms[s]) for s in species}


    def assemble(self) -> 'FemRhs':
        """Update the source term."""
        self._source_term.Assemble()
        return self


    @property
    def vec(self) -> ngs.BaseVector:
        """The vector of the right-hand side."""
        return self._source_term.vec


class PnpPotential:
    """FEM structures for Poisson-Nernst-Planck equations."""

    def __init__(self, stiffness, inverse, source_term, potential):
        self._stiffness = stiffness
        self._inverse = inverse
        self._source_term = source_term
        self._potential = potential

    @classmethod
    def for_all_species(
            cls,
            species,
            fes,
            simulation_geometry,
            concentrations,
    ):
        """Set up the right-hand side of the finite element equations for a given species.
        """
        compartments = list(simulation_geometry.compartments.values())
        faraday_const = to_simulation_units(96485.3365 * u.C / u.mol)

        # Set up potential matrix and source term
        trial, test = fes.TnT()
        offset = len(species)
        a = ngs.BilinearForm(fes, check_unused=False)
        f = ngs.LinearForm(fes)
        for k, compartment in enumerate(compartments):
            eps = compartment.coefficients.permittivity
            a += eps * ngs.grad(trial[k]) * ngs.grad(test[k]) * ngs.dx
            a += trial[k + offset] * test[k] * ngs.dx
            a += trial[k] * test[k + offset] * ngs.dx

            for s in species:
                c = concentrations[s]
                f += faraday_const * s.valence * c.components[k] * test[k] * ngs.dx

        a.Assemble()
        smoother = a.mat.CreateSmoother(fes.FreeDofs())
        inverse = ngs.GMRESSolver(a.mat, pre=smoother, printrates=False)
        potential = ngs.GridFunction(fes)

        return cls(a.mat, inverse, f, potential)


    def update(self):
        """Update the potential given the current status of chemical concentrations."""
        self._source_term.Assemble()
        self._potential.vec.data = self._inverse * self._source_term.vec


    def __getitem__(self, k: int) -> ngs.CoefficientFunction:
        """Returns the k-th component of the potential."""
        return self._potential.components[k]
