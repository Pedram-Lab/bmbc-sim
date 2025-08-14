import itertools

import ngsolve as ngs
import astropy.units as u
import astropy.constants as const
import numpy as np
import sympy
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, LinearOperator, spilu

from ecsim.simulation.simulation_agents import ChemicalSpecies
from ecsim.units import to_simulation_units


def ngs_to_csr(mat: ngs.Matrix) -> sp.csr_matrix:
    """Convert an NGSolve matrix to a SciPy CSR matrix."""
    # Extract the matrix data
    data = mat.CSR()
    val, col, ind = (v.NumPy().copy() for v in data)
    return sp.csr_matrix((val, col, ind), shape=mat.shape)


class ILUPreconditioner:
    """Incomplete LU preconditioner as a linear operator for scipy.sparse.linalg.gmres."""

    def __init__(self, matrix, fill_factor=10, drop_tol=1e-4):
        """Initialize ILU preconditioner.

        :param matrix: Sparse matrix (scipy.sparse.csr)
        :param fill_factor: Fill factor for ILU
        :param drop_tol: Drop tolerance for ILU
        """
        self.shape = matrix.shape

        try:
            # Compute incomplete LU factorization
            self.ilu = spilu(matrix.tocsc(), fill_factor=fill_factor, drop_tol=drop_tol)
        except RuntimeError as e:
            # Fallback to Jacobi if ILU fails
            print(f"Warning: ILU factorization failed ({e}), falling back to Jacobi preconditioner")
            self.ilu = None
            self.diag_inv = 1.0 / matrix.diagonal()
            self.diag_inv[~np.isfinite(self.diag_inv)] = 0.0

    def _ilu_matvec(self, x):
        """Apply the ILU preconditioner."""
        return self.ilu.solve(x)

    def _jac_matvec(self, x):
        """Apply the fallback Jacobi preconditioner."""
        return self.diag_inv * x

    def as_linear_operator(self):
        """Return as scipy LinearOperator."""
        if self.ilu is not None:
            return LinearOperator(self.shape, matvec=self._ilu_matvec)
        else:
            return LinearOperator(self.shape, matvec=self._jac_matvec)


class MatrixSum:
    """Class to compute the sum of two matrices."""

    def __init__(self, a, b):
        if a.shape != b.shape:
            raise ValueError("Incompatible matrix shapes")
        self.a = a
        self.b = b

    def _matvec(self, x):
        """Compute the sum of the two matrices."""
        return self.a @ x + self.b @ x

    def as_linear_operator(self):
        """Return as scipy LinearOperator."""
        return LinearOperator(self.a.shape, matvec=self._matvec)


class DiffusionSolver:
    """FEM solver for diffusion and transport equations."""

    def __init__(self, a, transport, m_star, preconditioner, source_term, dt):
        self._a = a
        self._transport = transport
        self._m_star = m_star
        self._preconditioner = preconditioner
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

        # Invert the matrix for the implicit Euler integrator (M* = M + dt * A)
        mass.mat.AsVector().data += dt * stiffness.mat.AsVector()
        mass = mass.mat.DeleteZeroElements(1e-10)
        mass_csr = ngs_to_csr(mass)

        # Create preconditioner for the system matrix
        preconditioner = ILUPreconditioner(mass_csr).as_linear_operator()

        stiffness = stiffness.mat.DeleteZeroElements(1e-10)
        stiffness_csr = ngs_to_csr(stiffness)

        return cls(
            stiffness_csr,
            transport_term,
            mass_csr,
            preconditioner,
            source_term,
            dt
        )

    def step(self, concentration: ngs.GridFunction):
        """Apply the implicit Euler step to the concentration vector."""
        # Assemble matrices
        self._transport.Assemble()
        self._source_term.Assemble()
        transport = self._transport.mat.DeleteZeroElements(1e-10)
        transport_csr = ngs_to_csr(transport)

        # Compute the residual
        c = concentration.vec.FV().NumPy().copy()
        res = self._dt * (self._source_term.vec.FV().NumPy() - self._a * c + transport_csr * c)

        # Create the system matrix: M* - dt * transport
        transport_csr.data *= -self._dt
        system_matrix_csr = MatrixSum(self._m_star, transport_csr).as_linear_operator()

        # Solve using scipy GMRES
        solution, info = gmres(
            system_matrix_csr,
            res,
            M=self._preconditioner,
            rtol=1e-8,
            atol=1e-12,
            maxiter=1000,
        )

        if info > 0:
            print(f"Warning: GMRES did not converge after {info} iterations")
        elif info < 0:
            print(f"Error: GMRES failed with error code {info}")

        # Update the concentration
        concentration.vec.FV().NumPy()[:] += solution


class ReactionSolver:
    """FEM solver for the reaction terms."""

    def __init__(self, source_terms, derivatives, rates, dt):
        self._source_terms = source_terms
        self._derivatives = derivatives
        self._rates = rates
        self._dt = dt
        self._res = None
        self._jac = None

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

                # Interpolate coefficient functions to FE grid to obtain node values
                kf_gf, kr_gf = rates[(reactants, products)]
                kf_gf.components[i].Set(kf)
                kr_gf.components[i].Set(kr)

        # Unpack rates to only store numpy vectors
        rates = list(map(lambda x: x.vec.FV().NumPy().copy(), itertools.chain(*rates.values())))

        # Set up the reaction terms for each reaction
        source_terms = {s.name: sympy.Float(0.0) for s in species}
        for (reactants, products), (kf, kr) in reactions.items():
            forward_reaction = kf
            for r in reactants:
                forward_reaction *= variables[r.name]
            for r in reactants:
                source_terms[r.name] -= forward_reaction
            for p in products:
                source_terms[p.name] += forward_reaction

            reverse_reaction = kr
            for p in products:
                reverse_reaction *= variables[p.name]
            for r in reactants:
                source_terms[r.name] += reverse_reaction
            for p in products:
                source_terms[p.name] -= reverse_reaction

        # Symbolically differentiate the source terms
        derivatives = [
            [source_terms[si.name].diff(variables[sj.name]) for sj in species]
            for si in species
        ]

        # Convert source terms and derivatives to callable functions
        source_terms = sympy.lambdify(
            list(variables.values()),
            list(source_terms.values()),
            modules=['numpy'],
            cse=True
        )
        derivatives = sympy.lambdify(
            list(variables.values()),
            derivatives,
            modules=['numpy'],
            cse=True
        )

        return cls(source_terms, derivatives, rates, dt)

    def newton_step(
        self,
        concentrations: np.ndarray,
        max_it: int,
        tol: float
    ) -> tuple[np.ndarray, int, bool]:
        """Apply one step of a Newton method with adaptive damping to the concentration vector."""
        nc, nn = concentrations.shape
        c_cur = concentrations.copy()
        if self._res is None or self._jac is None:
            self._res = np.zeros((nc, 1, nn))
            self._jac = np.zeros((nc, nc, nn))

        iteration = 0
        is_converged = False
        atol, rtol = tol, np.sqrt(tol)
        c_old_norm = np.linalg.norm(concentrations, ord=np.inf, axis=1)

        # Update the concentrations, stop when the updates are small
        while not is_converged and iteration < max_it:
            is_converged = True
            iteration += 1
            c_cur_norm = np.linalg.norm(c_cur, ord=np.inf, axis=1)

            # Evaluate source terms and derivatives
            args = [c_cur[i, :] for i in range(nc)] + self._rates
            source = self._source_terms(*args)
            deriv = self._derivatives(*args)

            # Assemble residual and Jacobian
            for i in range(nc):
                self._res[i, 0, :] = c_cur[i, :] - concentrations[i, :] - self._dt * source[i]
                self._jac[i, i, :] = 1 - self._dt * deriv[i][i]
            for i in range(nc):
                for j in range(i + 1, nc):
                    self._jac[i, j, :] = -self._dt * deriv[i][j]
                    self._jac[j, i, :] = -self._dt * deriv[j][i]

            # Compute Newton update
            delta = np.linalg.solve(self._jac.transpose((2, 0, 1)), self._res.transpose(2, 0, 1))
            delta = np.reshape(delta, (nn, nc)).T

            # Are residual and step small enough?
            upper_bound = atol + rtol * np.maximum(c_old_norm, c_cur_norm)
            is_converged &= np.all(np.linalg.norm(delta, ord=np.inf, axis=1) < upper_bound)
            is_converged &= np.all(np.linalg.norm(self._res, ord=np.inf, axis=(1, 2)) < upper_bound)

            # Only take a fractional step if full step would make a concentration negative
            eps = 1e-15
            with np.errstate(divide='ignore', invalid='ignore'):
                positivity_cap = np.where(delta > 0.0, (c_cur - eps) / delta, np.inf)
            alpha = np.minimum(1.0, (1 - eps) * np.min(positivity_cap))
            if alpha < 1:
                c_cur -= alpha * delta
            else:
                c_cur -= delta

        return c_cur, iteration, is_converged


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
