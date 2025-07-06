import ngsolve as ngs
import astropy.units as u
import astropy.constants as const
import numpy as np

from ecsim.simulation.simulation_agents import ChemicalSpecies
from ecsim.units import to_simulation_units


class DiffusionSolver:
    """FEM solver for diffusion and transport equations."""

    def __init__(self, a, transport, m_star, pre, source_term, dt):
        self._a = a
        self._transport = transport
        self._m_star = m_star
        self._pre = pre
        self._source_term = source_term
        self._dt = dt


    @classmethod
    def for_all_species(
            cls,
            species,
            fes,
            simulation_geometry,
            concentrations,
            potential,
            dt
    ) -> dict[ChemicalSpecies, 'DiffusionSolver']:
        """Set up the solver for all given species."""
        species_to_solver = {}
        for s in species:
            species_to_solver[s] = cls._for_single_species(
                s, fes, simulation_geometry, concentrations[s], potential, dt
            )
        return species_to_solver


    @classmethod
    def _for_single_species(
            cls,
            species,
            fes,
            simulation_geometry,
            concentration,
            potential,
            dt
    ):
        """Set up the solver for a single given species."""
        compartments = simulation_geometry.compartments.values()
        mass = ngs.BilinearForm(fes, check_unused=False)
        stiffness = ngs.BilinearForm(fes, check_unused=False)
        trial_and_test = tuple(zip(*fes.TnT()))
        compartment_to_index = {compartment: i for i, compartment in enumerate(compartments)}
        source_term = ngs.LinearForm(fes)

        for i, compartment in enumerate(compartments):
            coefficients = compartment.coefficients
            trial, test = trial_and_test[i]

            # Set up diffusion stiffness matrix
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

                def select_i(compartment, concentration, tnt):
                    if compartment is None:
                        return None, None, None
                    idx = compartment_to_index[compartment]
                    return concentration.components[idx], *tnt[idx]

                src_c, src_trial, src_test = select_i(source, concentration, trial_and_test)
                trg_c, trg_trial, trg_test = select_i(target, concentration, trial_and_test)

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

        # Handle explicit transport terms
        for membrane in simulation_geometry.membranes.values():
            for s, source, target, transport in membrane.get_transport():
                if s != species:
                    continue

                def select_e(compartment, concentration, tnt):
                    if compartment is None:
                        return None, None
                    idx = compartment_to_index[compartment]
                    return concentration.components[idx], tnt[idx][1]

                src_c, src_test = select_e(source, concentration, trial_and_test)
                trg_c, trg_test = select_e(target, concentration, trial_and_test)

                # Calculate the flux density through the membrane
                flux = transport.flux_rhs(src_c, trg_c)

                if flux is not None:
                    area = to_simulation_units(membrane.area, 'area')
                    flux_density = (flux / area).Compile()
                    ds = ngs.ds(membrane.name)
                    if src_test is not None:
                        source_term += -flux_density * src_test * ds
                    if trg_test is not None:
                        source_term += flux_density * trg_test * ds

        # Handle potential terms
        if potential is not None and species.valence != 0:
            beta = to_simulation_units(const.e.si / (const.k_B * 310 * u.K))
            h = ngs.specialcf.mesh_size
            for i, compartment in enumerate(compartments):
                trial, test = trial_and_test[i]
                d = compartment.coefficients.diffusion[species]

                # Drift term D * β * valence * u * ∇φ·∇v
                grad_phi = ngs.grad(potential[i])
                directional_test = ngs.InnerProduct(grad_phi, ngs.grad(test))
                drift = beta * species.valence * trial

                # SUPG regularization D * τ * (∇φ·∇u)(∇φ·∇v) with parameter τ ~ h/(2|∇φ|)
                tau = h / (2 * grad_phi.Norm() + 1e-6)
                supg = tau * (grad_phi * ngs.grad(trial))

                transport_term += (-d * (supg + drift) * directional_test).Compile() * ngs.dx

        # Assemble the mass and stiffness matrices
        mass.Assemble()
        stiffness.Assemble()

        # Invert the matrix for the implicit Euler integrator
        # Use GMRes with a Gauss-Seidel smoother (Jacobi converges to wrong solution!)
        mass.mat.AsVector().data += dt * stiffness.mat.AsVector()
        mass = mass.mat.DeleteZeroElements(1e-10)
        smoother = mass.CreateSmoother(fes.FreeDofs(), GS=True)
        stiffness = stiffness.mat.DeleteZeroElements(1e-10)

        return cls(
            stiffness,
            transport_term,
            mass,
            smoother,
            source_term,
            dt
        )

    def compute_residual(self, c: ngs.GridFunction) -> ngs.la.DynamicVectorExpression:
        """Compute the residual vector for the implicit Euler integrator."""
        # Update the transport terms (implicit and explicit)
        self._transport.Assemble()
        self._source_term.Assemble()
        stiffness = self._a - self._transport.mat

        return self._dt * (self._source_term.vec - stiffness * c.vec)

    def step(self, c: ngs.GridFunction, res: ngs.la.DynamicVectorExpression):
        """Apply the implicit Euler step to the concentration vector."""
        # Scale the transport terms
        scaled_transport = self._transport.mat.DeleteZeroElements(1e-10)
        scaled_transport.AsVector().FV().NumPy()[:] *= self._dt
        mstar_inv = ngs.GMRESSolver(self._m_star - scaled_transport, self._pre, printrates=False)

        # Update the concentration
        c.vec.data += mstar_inv * res


class ReactionSolver:
    """FEM solver for the reaction terms."""

    def __init__(self, source_term, derivative, lumped_mass_inv, dt):
        self._source_term = source_term
        self._lumped_mass_inv = lumped_mass_inv
        self._derivative = derivative
        self._dt = dt

    @classmethod
    def for_all_species(
            cls,
            species,
            fes,
            simulation_geometry,
            concentrations,
            dt
    ):
        """Set up the solver for all given species."""
        source_terms = {s: ngs.LinearForm(fes) for s in species}
        derivatives = {s: ngs.LinearForm(fes) for s in species}
        test_functions = fes.TestFunction()
        compartments = list(simulation_geometry.compartments.values())

        # Make the concentrations variables so one can differentiate in their direction
        concentrations = {s: concentrations[s].MakeVariable() for s in species}

        # To decouple the reactions, use mass lumping. NGSolve does not account for the
        # volume of the reference element in the integration rule, so we need to adjust
        # the usual [1/4, 1/4, 1/4, 1/4] weights by the volume of the 3D unit simplex.
        mass_lumping = {
            ngs.ET.TET: ngs.IntegrationRule(
                points=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                weights=[1 / 24, 1 / 24, 1 / 24, 1 / 24],
            )
        }

        # Set up the reaction terms for each compartment and species
        for i, compartment in enumerate(compartments):
            coefficients = compartment.coefficients
            test = test_functions[i]

            cf = {s: ngs.CoefficientFunction(0.0) for s in species}
            for (reactants, products), (kf, kr) in coefficients.reactions.items():
                all_reactants = ngs.CoefficientFunction(1.0)
                for reactant in reactants:
                    all_reactants *= concentrations[reactant].components[i]
                forward_reaction = kf * all_reactants
                for reactant in reactants:
                    cf[reactant] += -forward_reaction
                for product in products:
                    cf[product] += forward_reaction

                all_products = ngs.CoefficientFunction(1.0)
                for product in products:
                    all_products *= concentrations[product].components[i]
                reverse_reaction = kr * all_products
                for reactant in reactants:
                    cf[reactant] += reverse_reaction
                for product in products:
                    cf[product] += -reverse_reaction

            for s in species:
                c = concentrations[s]
                source_terms[s] += (cf[s] * test).Compile() * ngs.dx(intrules=mass_lumping)
                derivatives[s] += (cf[s].Diff(c) * test).Compile() * ngs.dx(intrules=mass_lumping)

        # Create a lumped mass matrix for decoupled time stepping
        trial_and_test = tuple(zip(*fes.TnT()))
        mass = ngs.BilinearForm(fes, check_unused=False)
        for trial, test in trial_and_test:
            mass += trial * test * ngs.dx
        mass.Assemble()

        v = mass.mat.CreateVector()
        w = mass.mat.CreateVector()
        v.FV().NumPy()[:] = 1
        w.data = mass.mat * v
        lumped_mass_inv = 1 / w.FV().NumPy()

        return {s: cls(source_terms[s], derivatives[s], lumped_mass_inv, dt) for s in species}

    def assemble_linearization(self) -> ngs.BaseVector:
        """Assemble function value and derivative from the current state."""
        self._source_term.Assemble()
        self._derivative.Assemble()

    def diagonal_newton_step(
        self,
        c: ngs.GridFunction,
        cumulative_update: np.ndarray,
    ) -> np.ndarray:
        """Apply one step of a diagonal Newton method to the concentration vector."""
        dt_m_inv = self._dt * self._lumped_mass_inv
        jac = 1 - dt_m_inv * self._derivative.vec.FV().NumPy()
        res = dt_m_inv * self._source_term.vec.FV().NumPy() - cumulative_update
        delta = res / jac
        c.vec.FV().NumPy()[:] += delta

        return delta


class PnpSolver:
    """FEM solver for Poisson-Nernst-Planck equations, computing the potential."""

    def __init__(self, stiffness, inverse, source_term, potential):
        self._stiffness = stiffness
        self._inverse = inverse
        self._source_term = source_term
        self.potential = potential

    @classmethod
    def for_all_species(
            cls,
            species,
            fes,
            simulation_geometry,
            concentrations,
    ):
        """Set up the solver for all given species."""
        compartments = list(simulation_geometry.compartments.values())
        faraday_const = to_simulation_units(96485.3365 * u.C / u.mol)

        # Set up potential matrix and source term
        trial, test = fes.TnT()
        offset = len(compartments)
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
        a = a.mat.DeleteZeroElements(1e-10)

        # Assemble preconditioner
        p = ngs.BilinearForm(fes, check_unused=False)
        for k, compartment in enumerate(compartments):
            p += trial[k + offset] * test[k + offset] * ngs.dx

        p.Assemble()
        p = p.mat.DeleteZeroElements(1e-10)

        free_dofs = fes.FreeDofs()
        for i in range(len(species)):
            free_dofs[-i - 1] = False

        smoother = a.CreateSmoother(free_dofs, GS=True) + p
        inverse = ngs.CGSolver(a, pre=smoother, printrates=False)
        potential = ngs.GridFunction(fes)

        return cls(a, inverse, f, potential)


    def step(self):
        """Update the potential given the current status of chemical concentrations."""
        self._source_term.Assemble()
        self.potential.vec.data = self._inverse * self._source_term.vec


    def __getitem__(self, k: int) -> ngs.CoefficientFunction:
        """Returns the k-th component of the potential."""
        return self.potential.components[k]
