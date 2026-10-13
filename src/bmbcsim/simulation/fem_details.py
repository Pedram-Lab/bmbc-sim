import itertools

import ngsolve as ngs
import astropy.units as u
import astropy.constants as const
import numpy as np
import sympy
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from bmbcsim.simulation.simulation_agents import ChemicalSpecies
from bmbcsim.units import to_simulation_units


def ngs_to_csr(mat: ngs.Matrix) -> sps.csr_matrix:
    """Convert an NGSolve matrix to a SciPy CSR matrix."""
    # Extract the matrix data
    data = mat.CSR()
    val, col, ind = (v.NumPy().copy() for v in data)
    return sps.csr_matrix((val, col, ind), shape=mat.shape)


class DiffusionSolver:
    """FEM solver for diffusion and transport equations."""

    def __init__(
            self,
            mass_form,
            stiffness_form,
            transport_form,
            drift,
            dt,
            reassemble,
    ):
        self._mass_form = mass_form
        self._stiffness_form = stiffness_form
        self._transport_form = transport_form
        self._drift = drift
        self._dt = dt
        self._reassemble = reassemble

        # Operators are (optionally) rebuilt on every step when reassemble=True
        self._stiffness = None
        self._lumped_mass = None
        self._m_star = None
        self._preconditioner = None
        self._prepare_operators()


    @classmethod
    def for_all_species(
            cls,
            species,
            fes,
            simulation_geometry,
            concentrations,
            potential,
            dt,
            reassemble=False,
    ) -> dict[ChemicalSpecies, 'DiffusionSolver']:
        """Set up the solver for all given species."""
        species_to_solver = {}
        for s in species:
            species_to_solver[s] = cls._for_single_species(
                s, fes, simulation_geometry, concentrations[s], potential, dt, reassemble
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
            dt,
            reassemble,
    ):
        """Set up the solver for a single given species."""
        compartments = simulation_geometry.compartments.values()
        mass = ngs.BilinearForm(fes, check_unused=False)
        stiffness = ngs.BilinearForm(fes, check_unused=False)
        trial_and_test = tuple(zip(*fes.TnT()))
        compartment_to_index = {compartment: i for i, compartment in enumerate(compartments)}

        for i, compartment in enumerate(compartments):
            coefficients = compartment.coefficients
            trial, test = trial_and_test[i]

            # Set up diffusion stiffness matrix
            if species in coefficients.diffusion and \
                    (diffusivity := coefficients.diffusion[species]) is not None:
                stiffness += diffusivity * ngs.grad(trial) * ngs.grad(test) * ngs.dx

            # Set up mass matrix
            mass += trial * test * ngs.dx

        # Handle membrane transport terms
        transport_form = ngs.LinearForm(fes)
        for membrane in simulation_geometry.membranes.values():
            for s, source, target, transport in membrane.get_transport():
                if s != species:
                    continue

                def select(compartment, concentration, tnt):
                    if compartment is None:
                        return None, None
                    idx = compartment_to_index[compartment]
                    porosity = compartment.coefficients.porosity
                    if porosity is None:
                        return concentration.components[idx], tnt[idx][1]
                    else:
                        return (
                            concentration.components[idx],
                            tnt[idx][1] / porosity,
                        )

                src_c, src_test = select(source, concentration, trial_and_test)
                trg_c, trg_test = select(target, concentration, trial_and_test)

                # Calculate the flux density through the membrane
                # Note: area normalization is handled in Transport.finalize_coefficients
                flux_density = transport.flux(src_c, trg_c)

                if flux_density is not None:
                    flux_density = flux_density.Compile()
                    ds = ngs.ds(membrane.name)
                    if src_test is not None:
                        transport_form += -flux_density * src_test * ds
                    if trg_test is not None:
                        transport_form += flux_density * trg_test * ds

        # Handle potential terms (electrostatic drift; implicit in diffusion half-step)
        drift = None
        if potential is not None and species.valence != 0:
            drift = ngs.BilinearForm(fes, check_unused=False)
            beta = to_simulation_units(const.e.si / (const.k_B * 310 * u.K))
            h = ngs.specialcf.mesh_size
            for i, compartment in enumerate(compartments):
                trial, test = trial_and_test[i]
                d = compartment.coefficients.diffusion[species]

                # Drift term D * β * valence * u * ∇φ·∇v
                grad_phi = ngs.grad(potential[i])
                directional_test = ngs.InnerProduct(grad_phi, ngs.grad(test))
                drift_term = beta * species.valence * trial

                # SUPG regularization D * τ * (∇φ·∇u)(∇φ·∇v) with parameter τ ~ h/(2|∇φ|)
                tau = h / (2 * grad_phi.Norm() + 1e-6)
                supg = tau * (grad_phi * ngs.grad(trial))

                drift += (-d * (supg + drift_term) * directional_test).Compile() * ngs.dx

        # Assemble the mass and stiffness matrices
        return cls(
            mass,
            stiffness,
            transport_form,
            drift,
            dt,
            reassemble,
        )

    def _prepare_operators(self):
        """Assemble matrices and compute the system/preconditioner."""
        self._stiffness_form.Assemble()
        stiffness_mat = self._stiffness_form.mat.DeleteZeroElements(1e-15)
        self._stiffness = ngs_to_csr(stiffness_mat)

        self._mass_form.Assemble()
        self._lumped_mass = np.asarray(ngs_to_csr(self._mass_form.mat).sum(axis=1)).flatten()

        # M* uses dt/2 for Strang splitting half-steps
        # Use lumped (diagonal) mass to preserve the discrete maximum principle
        half_dt = self._dt / 2
        m_star = sps.diags(self._lumped_mass) + half_dt * self._stiffness
        m_ilu = spla.spilu(m_star.T, fill_factor=5)
        self._preconditioner = spla.LinearOperator(
            m_star.shape, matvec=m_ilu.solve, dtype=np.float64
        )
        self._m_star = m_star

    def prepare(self):
        """Reassemble diffusion operators if needed (e.g., after mesh deformation).

        Call once per timestep from the simulation loop, before the first
        diffusion half-step.
        """
        if self._reassemble:
            self._prepare_operators()

    def diffusion_half_step(self, concentration: ngs.GridFunction):
        """Apply a half-step of implicit diffusion (with drift if applicable)."""
        half_dt = self._dt / 2
        c = concentration.vec.FV().NumPy().copy()

        if self._drift is not None:
            # Assemble drift (potential may have been updated since last call)
            self._drift.Assemble()
            drift_csr = ngs_to_csr(self._drift.mat.DeleteZeroElements(1e-10))

            rhs = half_dt * (drift_csr @ c - self._stiffness @ c)
            system = spla.LinearOperator(
                self._m_star.shape,
                matvec=lambda x: self._m_star @ x - half_dt * (drift_csr @ x),
                dtype=np.float64,
            )
        else:
            rhs = -half_dt * (self._stiffness @ c)
            system = self._m_star

        solution, info = spla.gmres(
            system,
            rhs,
            M=self._preconditioner,
            rtol=1e-6,
            atol=1e-12,
            maxiter=1000,
        )

        if info > 0:
            print(f"Warning: GMRES did not converge after {info} iterations")
        elif info < 0:
            print(f"Error: GMRES failed with error code {info}")

        concentration.vec.FV().NumPy()[:] += solution

    def transport_step(self, concentration: ngs.GridFunction):
        """Apply explicit membrane transport using lumped mass."""
        self._transport_form.Assemble()

        c = concentration.vec.FV().NumPy()
        rhs = self._dt * self._transport_form.vec.FV().NumPy()
        c[:] += rhs / self._lumped_mass


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
            c_cur -= delta
            c_cur_norm = np.linalg.norm(c_cur, ord=np.inf, axis=1)

            # Are residual and step small enough?
            upper_bound = atol + rtol * np.maximum(c_old_norm, c_cur_norm)
            is_converged &= np.all(np.linalg.norm(delta, ord=np.inf, axis=1) < upper_bound)
            is_converged &= np.all(np.linalg.norm(self._res, ord=np.inf, axis=(1, 2)) < upper_bound)

        return c_cur, iteration, is_converged


class PnpSolver:
    """FEM solver for Poisson-Nernst-Planck equations, computing the potential."""

    def __init__(
            self,
            a_form,
            b_form,
            source_term,
            potential,
            n_space,
            shape,
            reassemble,
    ):
        self._a_form = a_form
        self._b_form = b_form
        self._source_term = source_term
        self.potential = potential
        self._n_space = n_space
        self._shape = shape
        self._reassemble = reassemble

        self._prepare_matrix()

    @classmethod
    def for_all_species(
            cls,
            species,
            fes,
            simulation_geometry,
            concentrations,
            reassemble=False,
    ):
        """Set up the solver for all given species."""
        compartments = list(simulation_geometry.compartments.values())
        faraday_const = to_simulation_units(96485.3365 * u.C / u.mol)
        n_space = sum(fes.components[k].ndof for k in range(len(compartments)))
        n_compartments = len(compartments)
        shape = (n_space + n_compartments, n_space + n_compartments)

        # Set up potential matrix [[a, b], [b^T, 0]] and source term
        trial, test = fes.TnT()
        a = ngs.BilinearForm(fes, check_unused=False)
        b = ngs.BilinearForm(fes, check_unused=False)
        f = ngs.LinearForm(fes)
        for k, compartment in enumerate(compartments):
            eps = compartment.coefficients.permittivity
            a += eps * ngs.grad(trial[k]) * ngs.grad(test[k]) * ngs.dx
            b += trial[k + n_compartments] * test[k] * ngs.dx

            for s in species:
                c = concentrations[s]
                f += faraday_const * s.valence * c.components[k] * test[k] * ngs.dx

        potential = ngs.GridFunction(fes)

        return cls(a, b, f, potential, n_space, shape, reassemble)

    def _prepare_matrix(self):
        """Assemble the saddle-point matrix for the electrostatic potential."""
        self._a_form.Assemble()
        a_mat = self._a_form.mat.DeleteZeroElements(1e-10)
        a = ngs_to_csr(a_mat)
        a = a[:self._n_space, :self._n_space]

        self._b_form.Assemble()
        b_mat = self._b_form.mat.DeleteZeroElements(1e-10)
        b = ngs_to_csr(b_mat)
        b = b[:self._n_space, self._n_space:]

        # Augmented Lagrangian formulation: [[a + tau * b * bT, b], [bT, -I / tau]]
        tau = np.mean(a.diagonal())
        tau_inv = 1 / tau

        def matvec_full(x):
            f, g = x[:self._n_space], tau_inv * x[self._n_space:]
            r = b.T @ f
            s = a @ f + tau * (b @ (r + g))
            return np.concatenate([s, r - g])

        self._matrix = spla.LinearOperator(self._shape, matvec=matvec_full, dtype=np.float64)


    def step(self):
        """Update the potential given the current status of chemical concentrations."""
        if self._reassemble:
            self._prepare_matrix()

        self._source_term.Assemble()

        # Minres without preconditioning seemed to yield the best results
        solution, info = spla.minres(
            self._matrix,
            self._source_term.vec.FV().NumPy(),
            x0=self.potential.vec.FV().NumPy(),
            rtol=1e-8,
            maxiter=1000,
        )

        if info > 0:
            print(f"Warning: Minres did not converge in {info} iterations")
        elif info < 0:
            print(f"Error: Minres failed with error code {info}")

        self.potential.vec.FV().NumPy()[:] = solution


    def __getitem__(self, k: int) -> ngs.CoefficientFunction:
        """Returns the k-th component of the potential."""
        return self.potential.components[k]


# Globalization parameters for the mechanics Newton solve.
_ARMIJO_C1 = 1e-4           # sufficient-decrease constant for the line search
_MAX_BACKTRACK = 30         # max line-search halvings before declaring no progress
_MIN_LOAD_INCREMENT = 1e-3  # below this, conclude the mesh cannot represent the swell
_MIN_SWELLING = 1e-3        # floor on the target volume ratio J_g
_LOAD_EPS = 1e-9            # tolerance for "load factor has reached 1"


class MechanicSolver:
    """FEM solver for (non-linear) elasticity on the current mesh deformation."""

    def __init__(self, mesh, concentration_fes, simulation_geometry, concentrations):
        """
        :param mesh: The mesh to deform.
        :param concentration_fes: The finite element space for concentration fields.
        :param simulation_geometry: The simulation geometry containing compartments
            with elastic parameters.
        :param concentrations: Dictionary mapping species to concentration GridFunctions.
        """
        self._mesh = mesh
        self._fes = ngs.VectorH1(mesh, order=1)
        characteristic_length = np.ptp(mesh.ngmesh.Coordinates()) / np.sqrt(3)

        # Build per-region Lamé parameters from elastic properties
        young_modulus_values = {}
        mu_values = {}
        lam_values = {}
        compartments = list(simulation_geometry.compartments.values())
        for compartment in compartments:
            elasticity = compartment.coefficients.elasticity
            if elasticity is None:
                raise ValueError(f"Elasticity not defined for compartment '{compartment.name}'")

            young_raw, nu_raw = elasticity
            region_names = compartment.get_region_names()
            full_names = compartment.get_region_names(full_names=True)

            for region, full_name in zip(region_names, full_names):
                # Get young's modulus and poisson ratio for this region (either from dict or scalar)
                young = young_raw[region] if isinstance(young_raw, dict) else young_raw
                poisson = nu_raw[region] if isinstance(nu_raw, dict) else nu_raw

                young_modulus_values[full_name] = young
                mu_values[full_name] = young / (2 * (1 + poisson))
                lam_values[full_name] = young * poisson / ((1 + poisson) * (1 - 2 * poisson))

        young = mesh.MaterialCF(young_modulus_values)
        mu = mesh.MaterialCF(mu_values)
        lam = mesh.MaterialCF(lam_values)

        # Elastic energy of an order-one strain over the whole body; sets the
        # load-independent convergence floor in :meth:`step`.
        self._energy_scale = ngs.Integrate(mu, mesh)

        # Chemical swelling: the driving species sets the local stress-free
        # volume J_g rather than applying a pressure. This is what makes the
        # problem unconditionally solvable -- a pressure term `p * det(F)` is a
        # dead load whose energy falls linearly in J while the strain energy
        # grows only like J^(2/3), so beyond |p| ~ 0.64 mu no stationary point
        # exists. Rescaling the reference state instead keeps the energy coercive
        # (it tends to +infinity as J -> 0 and as J -> infinity) for every J_g.
        #
        # `_load_factor` interpolates J_g from 1 to its target so :meth:`step` can
        # reach a large swell incrementally; it stays at 1 for a full-load solve.
        self._load_factor = ngs.Parameter(1.0)
        # The driver is the amount of species per *reference* volume: the
        # concentration times the nodal volume ratio V / V_ref that
        # :meth:`adjust_concentrations` divides it by. The concentration alone is
        # per current volume and is diluted by the very swelling it drives, so
        # comparing it against a fixed baseline is a positive feedback loop:
        # swelling -> dilution -> "depletion" -> more swelling, with loop gain
        # k * c_eq * transmission. Above gain 1 the undeformed state is unstable
        # and the ECS locks into a swollen state (J = k * c_eq * T) that no
        # chemistry can undo. The same molecules pull the same whatever space
        # they occupy; only reaction and transport change the driver. Using the
        # nodal ratio (not det F of an element) makes the product exactly the
        # quantity the dilution conserves, also on slivers whose J differs from
        # their patch.
        self._volume_ratio = ngs.GridFunction(concentration_fes)
        self._volume_ratio.vec[:] = 1.0
        swelling = {}
        for i, compartment in enumerate(compartments):
            driving = compartment.coefficients.driving_species
            if driving is not None:
                species, strength, baseline = driving
                amount = concentrations[species].components[i] * self._volume_ratio.components[i]
                target = 1 + self._load_factor * strength * (amount - baseline)
                # J_g <= 0 is not a volume; unclamped it yields NaN rather than
                # any diagnosable failure.
                target = ngs.IfPos(target - _MIN_SWELLING, target, _MIN_SWELLING)
                for full_name in compartment.get_region_names(full_names=True):
                    swelling[full_name] = target
        growth = mesh.MaterialCF(swelling, default=1)

        # Neo-Hookean energy of the elastic part of F: F = F_e F_g with
        # F_g = J_g^(1/3) I. The density is per unit *grown* volume, hence the J_g.
        self._stiffness = ngs.BilinearForm(self._fes, symmetric=False)
        trial = self._fes.TrialFunction()
        deformation_tensor = ngs.Id(mesh.dim) + ngs.Grad(trial)
        elastic_tensor = deformation_tensor / growth ** (1 / 3)
        self._stiffness += ngs.Variation(
            (growth * neo_hooke(elastic_tensor, mu, lam)).Compile() * ngs.dx
        )

        # Set up boundary conditions (spring anchoring, "local compliant embedding")
        # Use BoundaryFromVolumeCF to evaluate MaterialCF on boundary elements
        exterior_boundaries = simulation_geometry.exterior_boundaries
        if exterior_boundaries:
            young_bnd = ngs.BoundaryFromVolumeCF(young)
            mu_bnd = ngs.BoundaryFromVolumeCF(mu)
            n = ngs.specialcf.normal(3)
            normal_springs = (young_bnd / (2 * characteristic_length)) * ngs.InnerProduct(trial, n) ** 2
            # Tangential part as the projection off the normal. Not
            # specialcf.tangential(3): that is an edge (codim-2) quantity, exactly
            # zero on the facets of a 3D mesh, which would silently drop all shear
            # resistance of the embedding.
            tangential = ngs.InnerProduct(trial, trial) - ngs.InnerProduct(trial, n) ** 2
            tangent_springs = (mu_bnd / (2 * characteristic_length)) * tangential
            for boundary_name in exterior_boundaries:
                self._stiffness += ngs.Variation(
                    (normal_springs + tangent_springs) * ngs.ds(boundary_name)
                )

        self._stiffness.Assemble()

        self.deformation = ngs.GridFunction(self._fes)
        self.deformation.vec[:] = 0

        self._residual = self.deformation.vec.CreateVector()
        self._direction = self.deformation.vec.CreateVector()
        self._trial = self.deformation.vec.CreateVector()

        # Set up volume tracking for concentration adjustment
        self._patch_mass = ngs.LinearForm(concentration_fes)
        for psi in concentration_fes.TestFunction():
            self._patch_mass += psi * ngs.dx
        self._patch_mass.Assemble()

        self._prev_mass = self._patch_mass.vec.FV().NumPy().copy()
        self._ref_mass = self._prev_mass.copy()

    def step(self, max_newton=50, atol=1e-12, rtol=1e-8):
        """Solve for elastic equilibrium under the current swelling.

        Attempts one line-searched Newton solve (:meth:`_newton_solve`) at full
        load, warm-started from the previous step; if that stalls, applies the
        swelling in adaptive increments (:meth:`_solve_by_load_stepping`).
        """
        # Solve on the reference configuration. The mesh still carries the
        # previous step's deformation (applied below so diffusion/reaction run on
        # the deformed geometry), and the energy is total-Lagrangian in
        # `self.deformation`, so integrating over an already-deformed mesh would
        # compound the two and leave a nonzero residual floor.
        self._mesh.UnsetDeformation()
        self._load_factor.Set(1.0)

        # Two convergence criteria; each covers where the other fails.
        #
        # `tol` scales with the applied load: the residual at u = 0, which is
        # exactly the force the swelling exerts (u = 0 is not stress-free once
        # J_g != 1). Scaling instead to the residual Newton happens to start from
        # is self-defeating, since a warm start is already near equilibrium and
        # the threshold would shrink with the quantity it is meant to bound.
        self._trial[:] = 0.0
        self._stiffness.Apply(self._trial, self._residual)
        tol = atol + rtol * np.linalg.norm(self._residual.FV().NumPy())

        # `min_decrement` does not vanish with the load, and `tol` does. Under no
        # load u = 0 is an equilibrium to within assembly round-off, which NGSolve
        # produces in no reproducible order across threads, so a purely
        # load-relative test becomes a coin flip on that noise. A residual
        # displacement below `rtol` times the size of the body is meaningless;
        # in the energy norm the Newton decrement measures, that is this energy.
        min_decrement = rtol ** 2 * self._energy_scale

        if not self._newton_solve(self.deformation.vec, max_newton, tol, min_decrement):
            # Restart undeformed so each increment begins at a finite-energy state.
            self.deformation.vec[:] = 0.0
            self._solve_by_load_stepping(max_newton, tol, min_decrement)
            self._load_factor.Set(1.0)

        # Apply the converged deformation to the mesh.
        self._mesh.SetDeformation(self.deformation)

    def _newton_solve(self, u, max_newton, tol, min_decrement):
        """Damped Newton with Armijo backtracking on the elastic + spring energy.

        The line search rejects any trial step of non-finite energy -- an
        inverted element, det F <= 0 -- so the tangent is only ever factorised at
        a valid configuration. Converged means the residual is below ``tol`` *or*
        the Newton decrement is below ``min_decrement``, whichever comes first.

        :param u: deformation vector; updated in place, must start finite-energy.
        :param tol: absolute residual threshold, set by :meth:`step`.
        :param min_decrement: decrement below which ``u`` counts as converged
            regardless of ``tol``, also set by :meth:`step`.
        :returns: True if either criterion is met, False on a stall, which tells
            the caller to cut the load increment.
        """
        energy = self._stiffness.Energy(u)
        if not np.isfinite(energy):
            return False

        self._stiffness.Apply(u, self._residual)

        for _ in range(max_newton):
            if np.linalg.norm(self._residual.FV().NumPy()) <= tol:
                return True

            # Newton direction d = K(u)^{-1} r; the update is u <- u - alpha*d.
            self._stiffness.AssembleLinearization(u)
            try:
                inv = self._stiffness.mat.Inverse(self._fes.FreeDofs())
            except Exception:
                return False
            self._direction.data = inv * self._residual

            # r . K^{-1} r > 0 for an SPD tangent; <= 0 is an indefinite Hessian,
            # so the direction is not a descent direction.
            slope = self._residual.InnerProduct(self._direction)
            if slope <= 0:
                return False

            # The Newton decrement is twice the energy drop this step predicts:
            # the displacement still missing, in the energy norm. Once negligible,
            # u is the equilibrium and the Armijo test below can no longer tell a
            # real decrease from round-off, so it would reject every trial step.
            # That is convergence, not a stall.
            if slope <= min_decrement:
                return True

            # Armijo backtracking on the energy, guarding against inverted elements.
            alpha = 1.0
            accepted = False
            for _ in range(_MAX_BACKTRACK):
                self._trial.data = u - alpha * self._direction
                trial_energy = self._stiffness.Energy(self._trial)
                if np.isfinite(trial_energy) and \
                        trial_energy <= energy - _ARMIJO_C1 * alpha * slope:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                return False

            u.data = self._trial
            energy = trial_energy
            self._stiffness.Apply(u, self._residual)

        # Exhausted Newton iterations; converged only if the residual is small.
        return np.linalg.norm(self._residual.FV().NumPy()) <= tol

    def _solve_by_load_stepping(self, max_newton, tol, min_decrement):
        """Reach the full swelling through adaptive increments.

        Each increment advances ``self._load_factor`` toward 1 and re-solves,
        warm-started from the previous converged increment; a failed increment is
        rejected and halved.

        This is a homotopy, not a bifurcation detector: the energy is coercive for
        every target volume, so a minimiser always exists, but neo-Hookean energy
        is polyconvex rather than convex and a cold start at a large J_g can fall
        outside the basin of attraction. Continuation walks it in.
        """
        u_safe = self.deformation.vec.CreateVector()
        u_safe.data = self.deformation.vec  # last converged state (at `load`)
        load = 0.0
        increment = 0.5
        while load < 1.0 - _LOAD_EPS:
            trial_load = min(load + increment, 1.0)
            self._load_factor.Set(trial_load)
            if self._newton_solve(self.deformation.vec, max_newton, tol, min_decrement):
                load = trial_load
                u_safe.data = self.deformation.vec
                increment = min(increment * 1.5, 1.0 - load)
            else:
                self.deformation.vec.data = u_safe  # reject and restore
                increment *= 0.5
                if increment < _MIN_LOAD_INCREMENT:
                    raise RuntimeError(
                        "Mechanics solve stalled at load fraction "
                        f"{load:.4g} of the requested swelling. An equilibrium "
                        "exists for every target volume, so this is a mesh "
                        "problem, not a physical limit: elements degenerate or "
                        "invert on the way. Check element quality in the driving "
                        "compartment, or reduce the coupling strength."
                    )

    def adjust_concentrations(self, concentrations: dict[ChemicalSpecies, ngs.GridFunction]):
        """Adjust concentrations based on the volume change due to mesh deformation.

        When the mesh deforms, local volumes change. Since the amount of substance
        is conserved, concentrations must be scaled by the ratio of old to new
        nodal volumes: c_new = c_old * (V_old / V_new).

        This method should be called after `step()` applies the deformation.

        :param concentrations: Dictionary mapping species to their concentration GridFunctions.
        """
        # Recompute nodal volumes on the deformed mesh
        self._patch_mass.Assemble()
        curr_mass = self._patch_mass.vec.FV().NumPy()

        # Compute the volume ratio (old / new) for scaling and store current mass
        volume_ratio = self._prev_mass / curr_mass
        self._prev_mass = curr_mass.copy()
        self._volume_ratio.vec.FV().NumPy()[:] = curr_mass / self._ref_mass

        # Scale all concentration fields
        for concentration in concentrations.values():
            concentration.vec.FV().NumPy()[:] *= volume_ratio


def neo_hooke(f, mu, lam):
    """Neo-Hookean strain energy density, normalised to vanish at F = I.

    Both terms are zero at F = I, so an undeformed body stores 0 rather than a
    constant ``mu * (mu/lam - 1)`` per unit volume. That offset changes no force,
    but it dominates ``BilinearForm.Energy``, which the Newton solve compares
    against in the Armijo line search and the round-off convergence check.

    :param f: Deformation gradient tensor (F = I + grad(u)).
    :param mu: Shear modulus (first Lamé parameter).
    :param lam: Second Lamé parameter.
    """
    det_f = ngs.Det(f)
    return mu * (
        0.5 * ngs.Trace(f.trans * f - ngs.Id(3))
        + mu / lam * (det_f ** (-lam / mu) - 1)
    )
