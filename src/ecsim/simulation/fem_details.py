import ngsolve as ngs

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
        test_and_trial = list(zip(*fes.TnT()))
        compartment_to_index = {compartment: i for i, compartment in enumerate(compartments)}

        for i, compartment in enumerate(compartments):
            coefficients = compartment.coefficients
            test, trial = test_and_trial[i]

            # Set up stiffness matrix (diffusion terms)
            if species in coefficients.diffusion and \
                    (diffusivity := coefficients.diffusion[species]) is not None:
                stiffness += diffusivity * ngs.grad(trial) * ngs.grad(test) * ngs.dx

            # Set up mass matrix
            mass += test * trial * ngs.dx

        # Handle implicit transport terms
        transport_term = ngs.BilinearForm(fes, check_unused=False)
        for membrane in simulation_geometry.membranes.values():
            for s, source, target, transport in membrane.get_transport():
                if s != species:
                    continue

                def get_index_and_concentration(compartment, concentration):
                    if compartment is None:
                        return None, None
                    idx = compartment_to_index[compartment]
                    return idx, concentration.components[idx]

                src_idx, src_c = get_index_and_concentration(source, concentration)
                trg_idx, trg_c = get_index_and_concentration(target, concentration)
                src_test, src_trial = None, None if src_idx is None else test_and_trial[src_idx]
                trg_test, trg_trial = None, None if trg_idx is None else test_and_trial[trg_idx]

                # Calculate the flux density through the membrane
                flux = transport.flux_lhs(src_c, trg_c, src_test, trg_test)

                if flux is not None:
                    area = to_simulation_units(membrane.area, 'area')
                    flux_density = (flux / area).Compile()
                    ds = ngs.ds(membrane.name)
                    if src_idx is not None:
                        transport_term += -flux_density * src_trial * ds
                    if trg_idx is not None:
                        transport_term += flux_density * trg_trial * ds

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
        return self._a + self._transport.mat


    @property
    def time_stepping(self) -> ngs.BaseMatrix:
        """The matrix for the implicit Euler rule."""
        return ngs.CGSolver(self._m_star + self._transport.mat, self._pre, printrates=False)


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

        # Handle transport terms
        for membrane in simulation_geometry.membranes.values():
            for s, source, target, transport in membrane.get_transport():
                concentration = concentrations[s]

                def get_index_and_concentration(compartment, concentration):
                    if compartment is None:
                        return None, None
                    idx = compartment_to_index[compartment]
                    return idx, concentration.components[idx]

                src_idx, src_c = get_index_and_concentration(source, concentration)
                trg_idx, trg_c = get_index_and_concentration(target, concentration)

                # Calculate the flux density through the membrane
                flux = transport.flux_rhs(src_c, trg_c)

                if flux is not None:
                    area = to_simulation_units(membrane.area, 'area')
                    flux_density = (flux / area).Compile()
                    ds = ngs.ds(membrane.name)
                    if src_idx is not None:
                        source_terms[s] += -flux_density * test_functions[src_idx] * ds
                    if trg_idx is not None:
                        source_terms[s] += flux_density * test_functions[trg_idx] * ds

        return {s: cls(source_terms[s]) for s in species}


    def assemble(self) -> 'FemRhs':
        """Update the source term."""
        self._source_term.Assemble()
        return self


    @property
    def vec(self) -> ngs.BaseVector:
        """The vector of the right-hand side."""
        return self._source_term.vec
