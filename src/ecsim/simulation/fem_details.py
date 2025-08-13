import itertools

import ngsolve as ngs
import astropy.units as u
import astropy.constants as const
import numpy as np
import sympy

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

    def step(self, c: ngs.GridFunction, res: ngs.la.DynamicVectorExpression) -> int:
        """Apply the implicit Euler step to the concentration vector."""
        # Scale the transport terms
        scaled_transport = self._transport.mat.DeleteZeroElements(1e-10)
        scaled_transport.AsVector().FV().NumPy()[:] *= self._dt
        mstar_inv = ngs.GMRESSolver(self._m_star - scaled_transport, self._pre, printrates=False)

        # Update the concentration
        c.vec.data += mstar_inv * res
        return mstar_inv.GetSteps()


class ReactionSolver:
    """FEM solver for the reaction terms."""

    def __init__(self, source_terms, derivatives, rates, dt):
        self._source_terms = source_terms
        self._derivatives = derivatives
        self._rates = rates
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
        compartments = list(simulation_geometry.compartments.values())

        # Make the concentrations variables so one can differentiate in their direction
        concentrations = {s: concentrations[s].MakeVariable() for s in species}

        variables = {s.name: sympy.Symbol(s.name) for s in species}
        reactions = {}
        rates = {}

        # Consolidate and evaluate coefficients
        for i, compartment in enumerate(compartments):
            coefficients = compartment.coefficients

            for (reactants, products), (kf, kr) in coefficients.reactions.items():
                if (reactants, products) not in rates:
                    # Create new symbols and vectors for the reaction rates
                    rates[(reactants, products)] = (ngs.GridFunction(fes), ngs.GridFunction(fes))
                    n = len(rates)
                    variables[f"kf_{n}"] = sympy.Symbol(f"kf_{n}")
                    variables[f"kr_{n}"] = sympy.Symbol(f"kr_{n}")
                    reactions[(reactants, products)] = (
                        variables[f"kf_{n}"],
                        variables[f"kr_{n}"],
                    )

                kf_gf, kr_gf = rates[(reactants, products)]
                kf_gf.components[i].Set(kf)
                kr_gf.components[i].Set(kr)

        # Set up the reaction terms for each reaction
        source_terms = {s.name: 0.0 for s in species}
        derivatives = {s.name: 0.0 for s in species}
        for (reactants, products), (kf, kr) in reactions.items():
            forward_reaction = kf
            for r in reactants:
                forward_reaction *= variables[r.name]
            for r in reactants:
                source_terms[r.name] -= forward_reaction
                derivatives[r.name] -= forward_reaction.diff(variables[r.name])
            for p in products:
                source_terms[p.name] += forward_reaction
                derivatives[p.name] += forward_reaction.diff(variables[p.name])

            reverse_reaction = kr
            for p in products:
                reverse_reaction *= variables[p.name]
            for r in reactants:
                source_terms[r.name] += reverse_reaction
                derivatives[r.name] += reverse_reaction.diff(variables[r.name])
            for p in products:
                source_terms[p.name] += -reverse_reaction
                derivatives[p.name] += -reverse_reaction.diff(variables[p.name])

        source_terms = sympy.lambdify(
            list(variables.values()),
            list(source_terms.values()),
            modules=['numpy'],
            cse=True
        )
        derivatives = sympy.lambdify(
            list(variables.values()),
            list(derivatives.values()),
            modules=['numpy'],
            cse=True
        )

        return cls(source_terms, derivatives, rates, dt)

    def diagonal_newton_step(
        self,
        concentrations: dict[ChemicalSpecies, ngs.GridFunction]
    ) -> np.array:
        """Apply one step of a diagonal Newton method to the concentration vector."""
        c = [concentrations[s].vec.FV().NumPy() for s in concentrations]
        r = list(map(lambda x: x.vec.FV().NumPy(), itertools.chain(*self._rates.values())))
        res = np.zeros((len(concentrations), c[0].size))
        jac = np.zeros((len(concentrations), c[0].size))
        source = self._source_terms(*c, *r)
        deriv = self._derivatives(*c, *r)
        for i, _ in enumerate(concentrations):
            res[i, :] = self._dt * source[i]
            jac[i, :] = 1 - self._dt * deriv[i]
        delta = res / jac
        for i, (_, c) in enumerate(concentrations.items()):
            c.vec.FV().NumPy()[:] += delta[i, :]

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


    def step(self) -> int:
        """Update the potential given the current status of chemical concentrations."""
        self._source_term.Assemble()
        self.potential.vec.data = self._inverse * self._source_term.vec
        return self._inverse.GetSteps()


    def __getitem__(self, k: int) -> ngs.CoefficientFunction:
        """Returns the k-th component of the potential."""
        return self.potential.components[k]
