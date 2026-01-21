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
            transport,
            source_term,
            dt,
            reassemble,
    ):
        self._mass_form = mass_form
        self._stiffness_form = stiffness_form
        self._transport = transport
        self._source_term = source_term
        self._dt = dt
        self._reassemble = reassemble

        # Operators are (optionally) rebuilt on every step when reassemble=True
        self._stiffness = None
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
                    porosity = compartment.coefficients.porosity
                    if porosity is None:
                        return concentration.components[idx], *tnt[idx]
                    else:
                        return (
                            concentration.components[idx],
                            tnt[idx][0],             # For porous flux,
                            tnt[idx][1] / porosity,  # scale test function
                        )

                src_c, src_trial, src_test = select_i(source, concentration, trial_and_test)
                trg_c, trg_trial, trg_test = select_i(target, concentration, trial_and_test)

                # Calculate the flux density through the membrane
                # Note: area normalization is handled in Transport.finalize_coefficients
                flux_density = transport.flux_lhs(src_c, trg_c, src_trial, trg_trial)

                if flux_density is not None:
                    flux_density = flux_density.Compile()
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
                    porosity = compartment.coefficients.porosity
                    if porosity is None:
                        return concentration.components[idx], tnt[idx][1]
                    else:
                        return (
                            concentration.components[idx],
                            tnt[idx][1] / porosity,
                        )

                src_c, src_test = select_e(source, concentration, trial_and_test)
                trg_c, trg_test = select_e(target, concentration, trial_and_test)

                # Calculate the flux density through the membrane
                # Note: area normalization is handled in Transport.finalize_coefficients
                flux_density = transport.flux_rhs(src_c, trg_c)

                if flux_density is not None:
                    flux_density = flux_density.Compile()
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
        return cls(
            mass,
            stiffness,
            transport_term,
            source_term,
            dt,
            reassemble,
        )

    def _prepare_operators(self):
        """Assemble matrices and compute the system/preconditioner."""
        self._stiffness_form.Assemble()
        stiffness_mat = self._stiffness_form.mat.DeleteZeroElements(1e-10)
        self._stiffness = ngs_to_csr(stiffness_mat)

        self._mass_form.Assemble()
        mass_mat = self._mass_form.mat.DeleteZeroElements(1e-10)
        mass_csr = ngs_to_csr(mass_mat)

        m_star = mass_csr + self._dt * self._stiffness
        m_ilu = spla.spilu(m_star.T, fill_factor=5)
        self._preconditioner = spla.LinearOperator(
            m_star.shape, matvec=m_ilu.solve, dtype=np.float64
        )
        self._m_star = m_star

    def step(self, concentration: ngs.GridFunction):
        """Apply the implicit Euler step to the concentration vector."""
        if self._reassemble:
            self._prepare_operators()

        # Assemble matrices
        self._transport.Assemble()
        self._source_term.Assemble()
        transport = self._transport.mat.DeleteZeroElements(1e-10)
        transport_csr = ngs_to_csr(transport)

        # Compute the residual
        c = concentration.vec.FV().NumPy().copy()
        res = self._dt * (
            self._source_term.vec.FV().NumPy() - self._stiffness * c + transport_csr * c
        )

        # Create the system matrix: M^* - dt * transport
        transport_csr.data *= -self._dt
        system_matrix_csr = spla.LinearOperator(
            transport_csr.shape,
            matvec=lambda x: self._m_star @ x + transport_csr @ x,
            dtype=np.float64,
        )

        # Solve using scipy GMRES
        solution, info = spla.gmres(
            system_matrix_csr,
            res,
            M=self._preconditioner,
            rtol=1e-6,
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

        # Build chemical pressure term from driving species
        chemical_pressure = None
        for i, compartment in enumerate(compartments):
            driving = compartment.coefficients.driving_species
            if driving is not None:
                species, strength = driving
                concentration = concentrations[species].components[i]
                term = strength * concentration
                chemical_pressure = term if chemical_pressure is None else chemical_pressure + term

        # Set up bulk term
        self._stiffness = ngs.BilinearForm(self._fes, symmetric=False)
        trial = self._fes.TrialFunction()
        deformation_tensor = ngs.Id(mesh.dim) + ngs.Grad(trial)
        self._stiffness += ngs.Variation(
            neo_hooke(deformation_tensor, mu, lam, chemical_pressure).Compile() * ngs.dx
        )

        # Set up boundary conditions (spring anchoring, "local compliant embedding")
        # Use BoundaryFromVolumeCF to evaluate MaterialCF on boundary elements
        exterior_boundaries = simulation_geometry.exterior_boundaries
        if exterior_boundaries:
            young_bnd = ngs.BoundaryFromVolumeCF(young)
            mu_bnd = ngs.BoundaryFromVolumeCF(mu)
            n = ngs.specialcf.normal(3)
            t = ngs.specialcf.tangential(3)
            normal_springs = (young_bnd / (2 * characteristic_length)) * ngs.InnerProduct(trial, n) ** 2
            tangent_springs = (mu_bnd / (2 * characteristic_length)) * ngs.InnerProduct(trial, t) ** 2
            for boundary_name in exterior_boundaries:
                self._stiffness += ngs.Variation(
                    (normal_springs + tangent_springs) * ngs.ds(boundary_name)
                )

        self._stiffness.Assemble()

        self.deformation = ngs.GridFunction(self._fes)
        self.deformation.vec[:] = 0

        self._residual = self.deformation.vec.CreateVector()

        # Set up volume tracking for concentration adjustment
        self._patch_mass = ngs.LinearForm(concentration_fes)
        for psi in concentration_fes.TestFunction():
            self._patch_mass += psi * ngs.dx
        self._patch_mass.Assemble()

        self._prev_mass = self._patch_mass.vec.FV().NumPy().copy()

    def step(self, n_iter=5):
        """Perform a nonlinear solve via simple Newton iterations."""
        for _ in range(n_iter):
            self._stiffness.Apply(self.deformation.vec, self._residual)
            self._stiffness.AssembleLinearization(self.deformation.vec)
            inv = self._stiffness.mat.Inverse(self._fes.FreeDofs())
            self.deformation.vec.data -= inv * self._residual

        # Apply deformation to mesh
        self._mesh.SetDeformation(self.deformation)

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

        # Scale all concentration fields
        for concentration in concentrations.values():
            concentration.vec.FV().NumPy()[:] *= volume_ratio


def neo_hooke(f, mu, lam, chemical_pressure=None):
    """Neo-Hookean material model with optional chemical pressure.

    :param f: Deformation gradient tensor (F = I + grad(u)).
    :param mu: Shear modulus (first Lamé parameter).
    :param lam: Second Lamé parameter.
    :param chemical_pressure: Optional chemical pressure term (coupling_strength * concentration).
        When provided, adds a pressure-like term that drives volume change.
    """
    det_f = ngs.Det(f)
    energy = mu * (
        0.5 * ngs.Trace(f.trans * f - ngs.Id(3))
        + mu / lam * det_f ** (-lam / mu)
        - 1
    )
    if chemical_pressure is not None:
        energy += chemical_pressure * det_f
    return energy
