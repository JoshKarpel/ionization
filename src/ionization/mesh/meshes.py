import itertools
import logging
from typing import Union, Optional, Iterable, NewType, Tuple, Dict
import abc

import numpy as np
import numpy.fft as nfft
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as sparsealg
import scipy.special as special
import scipy.integrate as integ

import simulacra as si
import simulacra.units as u

from .. import states, core, exceptions
from . import sims, mesh_operators, mesh_plotters

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CoordinateMesh = NewType('CoordinateMesh', np.array)
CoordinateVector = NewType('CoordinateVector', np.array)

ScalarMesh = NewType('ScalarMesh', np.array)
VectorMesh = NewType('VectorMesh', Tuple[ScalarMesh, ...])
GMesh = NewType('GMesh', ScalarMesh)
G2Mesh = NewType('G2Mesh', ScalarMesh)
PsiMesh = NewType('PsiMesh', ScalarMesh)
Psi2Mesh = NewType('Psi2Mesh', ScalarMesh)
WavefunctionMesh = Union[GMesh, G2Mesh, PsiMesh, Psi2Mesh]

GVector = NewType('GVector', np.array)
G2Vector = NewType('G2Vector', np.array)
PsiVector = NewType('PsiVector', np.array)
Psi2Vector = NewType('Psi2Vector', np.array)
WavefunctionVector = Union[GVector, G2Vector, PsiVector, Psi2Vector]

StateOrGMesh = Optional[Union[states.QuantumState, GMesh]]  # None => the current g mesh

OperatorMatrix = NewType('SparseMatrixOperator', sparse.dia_matrix)


def add_to_diagonal_sparse_matrix_diagonal(dia_matrix: sparse.dia_matrix, value = 1) -> sparse.dia_matrix:
    s = dia_matrix.copy()
    s.setdiag(s.diagonal() + value)
    return s


def add_to_diagonal_sparse_matrix_diagonal_inplace(dia_matrix: sparse.dia_matrix, value = 1) -> sparse.dia_matrix:
    dia_matrix.data[1] += value
    return dia_matrix


class WrappingDirection(si.utils.StrEnum):
    X = 'x'
    Y = 'y'
    Z = 'z'
    RHO = 'rho'
    CHI = 'chi'
    THETA = 'theta'
    R = 'r'
    L = 'l'


class QuantumMesh(abc.ABC):
    mesh_plotter_type = mesh_plotters.MeshPlotter

    def __init__(self, sim: 'sims.MeshSimulation'):
        self.sim = sim
        self.spec = sim.spec
        self.operators = self.spec.operators

        self.g = None
        self.inner_product_multiplier = None

        self.plot = self.mesh_plotter_type(self)

    def __eq__(self, other):
        """
        QuantumMeshes should evaluate equal if and only if their Simulations are equal and their g (the only thing which carries state information) are the same.
        """
        return isinstance(other, self.__class__) and self.sim == other.sim and np.array_equal(self.g, other.g)

    def __hash__(self):
        """Return the hash of the QuantumMesh, which is the same as the hash of the associated Simulation."""
        return hash(self.sim)

    def __str__(self):
        return f'{self.__class__.__name__} for {self.sim}'

    def __repr__(self):
        return f'{self.__class__.__name__}(sim = {repr(self.sim)})'

    def flatten_mesh(self, mesh, flatten_along: Optional[WrappingDirection]):
        """Return a mesh flattened along one of the mesh coordinates ('theta' or 'r')."""
        flat = self._wrapping_direction_to_order(flatten_along)

        if flat is None:
            return mesh

        return mesh.flatten(flat)

    def wrap_vector(self, vector, wrap_along: Optional[WrappingDirection]):
        wrap = self._wrapping_direction_to_order(wrap_along)

        if wrap is None:
            return vector

        return np.reshape(vector, self.mesh_shape, wrap)

    @abc.abstractmethod
    def _wrapping_direction_to_order(self, wrapping_direction: Optional[WrappingDirection]) -> Optional[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_g_for_state(self, state: StateOrGMesh) -> GMesh:
        raise NotImplementedError

    def state_to_mesh(self, state_or_mesh: StateOrGMesh) -> GMesh:
        """Return the mesh associated with the given state, or simply passes the mesh through."""
        if state_or_mesh is None:
            return self.g
        elif isinstance(state_or_mesh, states.QuantumState):
            try:
                state_or_mesh = self.analytic_to_numeric[state_or_mesh]
            except (AttributeError, KeyError):
                pass
            return self.get_g_for_state(state_or_mesh)
        else:
            return state_or_mesh

    def get_g_with_states_removed(self, states: Iterable[StateOrGMesh], g: StateOrGMesh = None) -> GMesh:
        """
        Get a g mesh with the contributions from the states removed.

        :param states: a list of states to remove from g
        :param g: a g to remove the state contributions from. Defaults to self.g
        :return:
        """
        if g is None:
            g = self.g

        g = g.copy()  # always act on a copy of g, regardless of source

        for state in states:
            g -= self.inner_product(state, g) * self.get_g_for_state(state)

        return g

    def inner_product(self, a: StateOrGMesh = None, b: StateOrGMesh = None) -> complex:
        """Inner product between two meshes. If either mesh is None, the state on the g is used for that state."""
        return np.sum(np.conj(self.state_to_mesh(a)) * self.state_to_mesh(b)) * self.inner_product_multiplier

    def state_overlap(self, a: StateOrGMesh = None, b: StateOrGMesh = None) -> float:
        """State overlap between two states. If either state is None, the state on the g is used for that state."""
        return np.abs(self.inner_product(a, b)) ** 2

    def expectation_value(self, state: StateOrGMesh = None, operator: mesh_operators.SumOfOperators = mesh_operators.SumOfOperators()) -> float:
        state_after_oper, _ = operator.apply(self, self.state_to_mesh(state), None)
        return self.inner_product(state, state_after_oper).real

    def norm(self, state: StateOrGMesh = None) -> float:
        return self.expectation_value(state)

    def internal_energy_expectation_value(self, state: StateOrGMesh = None):
        return self.expectation_value(state, operator = self.operators.internal_hamiltonian(self))

    def total_energy_expectation_value(self, state: StateOrGMesh = None):
        return self.expectation_value(state, operator = self.operators.total_hamiltonian(self))

    def z_expectation_value(self, state: StateOrGMesh = None):
        return self.expectation_value(state, operator = self.operators.z(self))

    def r_expectation_value(self, state: StateOrGMesh = None):
        return self.expectation_value(state, operator = self.operators.r(self))

    @property
    def psi(self) -> PsiMesh:
        return self.g / self.g_factor

    @property
    def g2(self) -> G2Mesh:
        return np.abs(self.g) ** 2

    @property
    def psi2(self) -> Psi2Mesh:
        return np.abs(self.psi) ** 2

    def evolve(self, time_step: complex):
        self.g = self.spec.evolution_method.evolve(self, self.g, time_step)
        self.g *= self.spec.mask(r = self.r_mesh)

    def get_mesh_slicer(self, plot_limit: float):
        raise NotImplementedError


class LineMesh(QuantumMesh):
    mesh_storage_method = ('z',)
    mesh_plotter_type = mesh_plotters.LineMeshPlotter

    def __init__(self, sim: 'sims.MeshSimulation'):
        super().__init__(sim)

        self.z_mesh = np.linspace(-self.spec.z_bound, self.spec.z_bound, self.spec.z_points)
        self.delta_z = np.abs(self.z_mesh[1] - self.z_mesh[0])
        self.z_center_index = si.utils.find_nearest_entry(self.z_mesh, 0).index
        self.mesh_points = len(self.z_mesh)

        self.inner_product_multiplier = self.delta_z
        self.g_factor = 1

        if self.spec.use_numeric_eigenstates:
            logger.debug('Calculating numeric eigenstates')

            self.analytic_to_numeric = self._get_numeric_eigenstate_basis(self.spec.number_of_numeric_eigenstates)
            self.spec.test_states = sorted(list(self.analytic_to_numeric.values()), key = lambda x: x.energy)
            self.spec.initial_state = self.analytic_to_numeric[self.spec.initial_state]

            logger.warning(f'Replaced test states for {self} with numeric eigenbasis')

        self.g = self.get_g_for_state(self.spec.initial_state)

    @property
    def r_mesh(self) -> CoordinateMesh:
        return self.z_mesh

    def _wrapping_direction_to_order(self, wrapping_direction: Optional[WrappingDirection]) -> Optional[str]:
        return None

    @si.utils.memoize
    def get_g_for_state(self, state: StateOrGMesh) -> GMesh:
        if state.analytic and self.spec.use_numeric_eigenstates:
            try:
                state = self.analytic_to_numeric[state]
            except (AttributeError, KeyError):
                logger.debug(f'Analytic to numeric eigenstate lookup failed for state {state}')

        g = state(self.z_mesh)
        g /= np.sqrt(self.norm(g))
        g *= state.amplitude

        return g

    def _get_numeric_eigenstate_basis(self, number_of_eigenstates: int):
        analytic_to_numeric = {}

        h = self.operators.internal_hamiltonian(self)

        eigenvalues, eigenvectors = sparsealg.eigsh(h, k = number_of_eigenstates, which = 'SA')

        for nn, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
            eigenvector /= np.sqrt(self.inner_product_multiplier * np.sum(np.abs(eigenvector) ** 2))  # normalize

            try:
                bound = True
                analytic_state = self.spec.analytic_eigenstate_type.from_potential(
                    self.spec.internal_potential,
                    self.spec.test_mass,
                    n = nn + self.spec.analytic_eigenstate_type.smallest_n
                )
            except exceptions.IllegalQuantumState:
                bound = False
                analytic_state = states.OneDPlaneWave.from_energy(eigenvalue, mass = self.spec.test_mass)

            numeric_state = states.NumericOneDState(eigenvector, eigenvalue, bound = bound, corresponding_analytic_state = analytic_state)
            if analytic_state is None:
                analytic_state = numeric_state

            analytic_to_numeric[analytic_state] = numeric_state

        logger.debug(f'Generated numeric eigenbasis with {len(analytic_to_numeric)} states')

        return analytic_to_numeric

    def gauge_transformation(self, *, g: GMesh = None, leaving_gauge: core.Gauge) -> GMesh:
        g = self.state_to_mesh(g)
        if leaving_gauge is None:
            leaving_gauge = self.spec.evolution_gauge

        vamp = self.spec.electric_potential.get_vector_potential_amplitude_numeric_cumulative(self.sim.times_to_current)
        integral = integ.simps(
            y = vamp ** 2,
            x = self.sim.times_to_current,
        )

        dipole_to_velocity = np.exp(1j * integral * (self.spec.test_charge ** 2) / (2 * self.spec.test_mass * u.hbar))
        dipole_to_length = np.exp(-1j * self.spec.test_charge * vamp[-1] * self.z_mesh / u.hbar)

        if leaving_gauge == core.Gauge.LENGTH:
            return np.conj(dipole_to_length) * dipole_to_velocity * g
        elif leaving_gauge == core.Gauge.VELOCITY:
            return dipole_to_length * np.conj(dipole_to_velocity) * g

    def get_mesh_slicer(self, plot_limit: Optional[float]) -> slice:
        if plot_limit is None:
            mesh_slicer = slice(None, None, 1)
        else:
            x_lim_points = round(plot_limit / self.delta_z)
            mesh_slicer = slice(int(self.z_center_index - x_lim_points), int(self.z_center_index + x_lim_points + 1), 1)

        return mesh_slicer


class CylindricalSliceMesh(QuantumMesh):
    mesh_storage_method = ('z', 'rho')
    mesh_plotter_type = mesh_plotters.CylindricalSliceMeshPlotter

    def __init__(self, sim: 'sims.MeshSimulation'):
        super().__init__(sim)

        self.z = np.linspace(-self.spec.z_bound, self.spec.z_bound, self.spec.z_points)
        self.rho = np.delete(np.linspace(0, self.spec.rho_bound, self.spec.rho_points + 1), 0)

        self.delta_z = self.z[1] - self.z[0]
        self.delta_rho = self.rho[1] - self.rho[0]
        self.inner_product_multiplier = self.delta_z * self.delta_rho

        self.rho -= self.delta_rho / 2

        self.z_center_index = int(self.spec.z_points // 2)
        self.z_max = np.max(self.z)
        self.rho_max = np.max(self.rho)

        self.g = self.get_g_for_state(self.spec.initial_state)

        self.mesh_points = len(self.z) * len(self.rho)
        self.matrix_operator_shape = (self.mesh_points, self.mesh_points)
        self.mesh_shape = np.shape(self.r_mesh)

    @property
    @si.utils.memoize
    def z_mesh(self) -> CoordinateMesh:
        return np.meshgrid(self.z, self.rho, indexing = 'ij')[0]

    @property
    @si.utils.memoize
    def rho_mesh(self) -> CoordinateMesh:
        return np.meshgrid(self.z, self.rho, indexing = 'ij')[1]

    @property
    def g_factor(self) -> np.array:
        return np.sqrt(u.twopi * self.rho_mesh)

    @property
    def r_mesh(self) -> CoordinateMesh:
        return np.sqrt((self.z_mesh ** 2) + (self.rho_mesh ** 2))

    @property
    def theta_mesh(self) -> CoordinateMesh:
        return np.arccos(self.z_mesh / self.r_mesh)

    @property
    def sin_theta_mesh(self) -> CoordinateMesh:
        return np.sin(self.theta_mesh)

    @property
    def cos_theta_mesh(self) -> CoordinateMesh:
        return np.cos(self.theta_mesh)

    def _wrapping_direction_to_order(self, wrapping_direction: Optional[WrappingDirection]) -> Optional[str]:
        if wrapping_direction is None:
            return None
        elif wrapping_direction == WrappingDirection.Z:
            return 'F'
        elif wrapping_direction == WrappingDirection.RHO:
            return 'C'
        else:
            raise ValueError(f"{wrapping_direction} is not a valid specifier for flatten_mesh (valid specifiers: 'z', 'rho')")

    @si.utils.memoize
    def get_g_for_state(self, state: states.QuantumState) -> GMesh:
        g = self.g_factor * state(self.r_mesh, self.theta_mesh, 0)
        g /= np.sqrt(self.norm(g))
        g *= state.amplitude

        return g

    def get_probability_current_density_vector_field(self, state: StateOrGMesh = None) -> VectorMesh:
        ops = self.operators.probability_current(self)
        dirs_to_op = {op.wrapping_direction: op for op in ops}
        z_op = dirs_to_op[WrappingDirection.Z]
        rho_op = dirs_to_op[WrappingDirection.RHO]

        mesh = self.state_to_mesh(state)

        current_density_mesh_z = np.imag(np.conj(mesh) * mesh_operators.apply_operators_sequentially(self, mesh, [z_op]))
        current_density_mesh_rho = np.imag(np.conj(mesh) * mesh_operators.apply_operators_sequentially(self, mesh, [rho_op]))

        return current_density_mesh_z, current_density_mesh_rho

    def get_spline_for_mesh(self, mesh: ScalarMesh):
        return sp.interp.RectBivariateSpline(self.z, self.rho, mesh)

    @si.utils.memoize
    def get_mesh_slicer(self, plot_limit: Optional[float] = None):
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if plot_limit is None:
            mesh_slicer = (slice(None, None, 1), slice(None, None, 1))
        else:
            z_lim_points = round(plot_limit / self.delta_z)
            rho_lim_points = round(plot_limit / self.delta_rho)
            mesh_slicer = (slice(int(self.z_center_index - z_lim_points), int(self.z_center_index + z_lim_points + 1), 1), slice(0, int(rho_lim_points + 1), 1))

        return mesh_slicer


# class WarpedCylindricalSliceMesh(QuantumMesh):
#     mesh_storage_method = ('z', 'rho')
#     mesh_plotter_type = mesh_plotters.
#
#     def __init__(self, sim: 'sims.MeshSimulation'):
#         super().__init__(sim)
#
#         self.z = np.linspace(-self.spec.z_bound, self.spec.z_bound, self.spec.z_points)
#         self.chi_max = self.spec.rho_bound ** (1 / self.spec.warping)
#         self.chi = np.linspace(0, self.chi_max, self.spec.rho_points + 1)[1:]
#
#         self.delta_z = self.z[1] - self.z[0]
#         self.delta_chi = self.chi[1] - self.chi[0]
#         self.inner_product_multiplier = self.delta_z * self.delta_chi
#
#         self.z_center_index = int(self.spec.z_points // 2)
#
#         self.g = self.get_g_for_state(self.spec.initial_state)
#
#         self.mesh_points = len(self.z) * len(self.chi)
#         self.matrix_operator_shape = (self.mesh_points, self.mesh_points)
#         self.mesh_shape = np.shape(self.r_mesh)
#
#     @property
#     @si.utils.memoize
#     def z_mesh(self) -> CoordinateMesh:
#         return np.meshgrid(self.z, self.chi, indexing = 'ij')[0]
#
#     @property
#     @si.utils.memoize
#     def chi_mesh(self) -> CoordinateMesh:
#         return np.meshgrid(self.z, self.chi, indexing = 'ij')[1]
#
#     @property
#     @si.utils.memoize
#     def rho_mesh(self) -> CoordinateMesh:
#         return self.chi_mesh ** self.spec.warping
#
#     @property
#     def g_factor(self):
#         return np.sqrt(u.twopi * self.spec.warping) * (self.chi_mesh ** (self.spec.warping - 0.5))
#
#     @property
#     def r_mesh(self) -> CoordinateMesh:
#         return np.sqrt((self.z_mesh ** 2) + (self.rho_mesh ** 2))
#
#     @property
#     def theta_mesh(self) -> CoordinateMesh:
#         return np.arccos(self.z_mesh / self.r_mesh)
#
#     @property
#     def sin_theta_mesh(self) -> CoordinateMesh:
#         return np.sin(self.theta_mesh)
#
#     @property
#     def cos_theta_mesh(self) -> CoordinateMesh:
#         return np.cos(self.theta_mesh)
#
#     def _wrapping_direction_to_order(self, wrapping_direction: Optional[WrappingDirection]) -> Optional[str]:
#         if wrapping_direction is None:
#             return None
#         elif wrapping_direction == WrappingDirection.Z:
#             return 'F'
#         elif wrapping_direction == WrappingDirection.CHI:
#             return 'C'
#         else:
#             raise ValueError(f"{wrapping_direction} is not a valid specifier for flatten_mesh (valid specifiers: 'z', 'chi')")
#
#     @si.utils.memoize
#     def get_g_for_state(self, state) -> GMesh:
#         g = self.g_factor * state(self.r_mesh, self.theta_mesh, 0)
#         g /= np.sqrt(self.norm(g))
#         g *= state.amplitude
#
#         return g
#
#     def _get_kinetic_energy_matrix_operators_HAM(self):
#         """Get the mesh kinetic energy operator matrices for z and rho."""
#         z_prefactor = -(u.hbar ** 2) / (2 * self.spec.test_mass * (self.delta_z ** 2))
#         chi_prefactor = -(u.hbar ** 2) / (2 * self.spec.test_mass * (self.delta_chi ** 2))
#
#         z_diagonal = z_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
#         z_offdiagonal = z_prefactor * np.array([1 if (z_index + 1) % self.spec.z_points != 0 else 0 for z_index in range(self.mesh_points - 1)], dtype = np.complex128)
#
#         @si.utils.memoize
#         def c(j):
#             return j / np.sqrt((j ** 2) - 0.25)
#
#         chi_diagonal = chi_prefactor * ((-2 * np.ones(self.mesh_points, dtype = np.complex128)) + ((self.spec.warping - .5) ** 2))
#         chi_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
#         for rho_index in range(self.mesh_points - 1):
#             if (rho_index + 1) % self.spec.rho_points != 0:
#                 j = (rho_index % self.spec.rho_points) + 1  # get j for the upper diagonal
#                 chi_offdiagonal[rho_index] = c(j)
#         chi_offdiagonal *= chi_prefactor
#
#         z_kinetic = sparse.diags([z_offdiagonal, z_diagonal, z_offdiagonal], offsets = (-1, 0, 1))
#         rho_kinetic = sparse.diags([chi_offdiagonal, chi_diagonal, chi_offdiagonal], offsets = (-1, 0, 1))
#
#         return z_kinetic, rho_kinetic
#
#     @si.utils.memoize
#     def get_internal_hamiltonian_matrix_operators(self):
#         """Get the mesh internal Hamiltonian matrix operators.py for z and rho."""
#         kinetic_z, kinetic_rho = self.get_kinetic_energy_matrix_operators()
#         potential_mesh = self.spec.internal_potential(r = self.r_mesh, test_charge = self.spec.test_charge)
#
#         kinetic_z = add_to_diagonal_sparse_matrix_diagonal(kinetic_z, value = 0.5 * self.flatten_mesh(potential_mesh, 'z'))
#         kinetic_rho = add_to_diagonal_sparse_matrix_diagonal(kinetic_rho, value = 0.5 * self.flatten_mesh(potential_mesh, 'rho'))
#
#         return kinetic_z, kinetic_rho
#
#     def _get_interaction_hamiltonian_matrix_operators_LEN(self):
#         """Get the interaction term calculated from the Lagrangian evolution equations."""
#         electric_potential_energy_mesh = self.spec.electric_potential(t = self.sim.time, distance_along_polarization = self.z_mesh, test_charge = self.spec.test_charge)
#
#         interaction_hamiltonian_z = sparse.diags(self.flatten_mesh(electric_potential_energy_mesh, 'z'))
#         interaction_hamiltonian_rho = sparse.diags(self.flatten_mesh(electric_potential_energy_mesh, 'rho'))
#
#         return interaction_hamiltonian_z, interaction_hamiltonian_rho
#
#     def _get_interaction_hamiltonian_matrix_operators_VEL(self):
#         # vector_potential_amplitude = -self.spec.electric_potential.get_electric_field_integral_numeric_cumulative(self.sim.times_to_current)
#         raise NotImplementedError
#
#     def tg_mesh(self, use_abs_g = False):
#         hamiltonian_z, hamiltonian_rho = self.get_kinetic_energy_matrix_operators()
#
#         if use_abs_g:
#             g = np.abs(self.g)
#         else:
#             g = self.g
#
#         g_vector_z = self.flatten_mesh(g, 'z')
#         hg_vector_z = hamiltonian_z.dot(g_vector_z)
#         hg_mesh_z = self.wrap_vector(hg_vector_z, 'z')
#
#         g_vector_rho = self.flatten_mesh(g, 'rho')
#         hg_vector_rho = hamiltonian_rho.dot(g_vector_rho)
#         hg_mesh_rho = self.wrap_vector(hg_vector_rho, 'rho')
#
#         return hg_mesh_z + hg_mesh_rho
#
#     def hg_mesh(self, include_interaction = False):
#         hamiltonian_z, hamiltonian_rho = self.get_internal_hamiltonian_matrix_operators()
#
#         g_vector_z = self.flatten_mesh(self.g, 'z')
#         hg_vector_z = hamiltonian_z.dot(g_vector_z)
#         hg_mesh_z = self.wrap_vector(hg_vector_z, 'z')
#
#         g_vector_rho = self.flatten_mesh(self.g, 'rho')
#         hg_vector_rho = hamiltonian_rho.dot(g_vector_rho)
#         hg_mesh_rho = self.wrap_vector(hg_vector_rho, 'rho')
#
#         if include_interaction:
#             raise NotImplementedError
#
#         return hg_mesh_z + hg_mesh_rho
#
#     @si.utils.memoize
#     def _get_probability_current_matrix_operators(self):
#         """Get the mesh probability current operators.py for z and rho."""
#         z_prefactor = u.hbar / (4 * u.pi * self.spec.test_mass * self.delta_rho * self.delta_z)
#         rho_prefactor = u.hbar / (4 * u.pi * self.spec.test_mass * (self.delta_rho ** 2))
#
#         # construct the diagonals of the z probability current matrix operator
#         z_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
#         for z_index in range(0, self.mesh_points - 1):
#             if (z_index + 1) % self.spec.z_points == 0:  # detect edge of mesh
#                 z_offdiagonal[z_index] = 0
#             else:
#                 j = z_index // self.spec.z_points
#                 z_offdiagonal[z_index] = 1 / (j + 0.5)
#         z_offdiagonal *= z_prefactor
#
#         @si.utils.memoize
#         def d(j):
#             return 1 / np.sqrt((j ** 2) - 0.25)
#
#         # construct the diagonals of the rho probability current matrix operator
#         rho_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
#         for rho_index in range(0, self.mesh_points - 1):
#             if (rho_index + 1) % self.spec.rho_points == 0:  # detect edge of mesh
#                 rho_offdiagonal[rho_index] = 0
#             else:
#                 j = (rho_index % self.spec.rho_points) + 1
#                 rho_offdiagonal[rho_index] = d(j)
#         rho_offdiagonal *= rho_prefactor
#
#         z_current = sparse.diags([-z_offdiagonal, z_offdiagonal], offsets = [-1, 1])
#         rho_current = sparse.diags([-rho_offdiagonal, rho_offdiagonal], offsets = [-1, 1])
#
#         return z_current, rho_current
#
#     def get_probability_current_vector_field(self):
#         z_current, rho_current = self._get_probability_current_matrix_operators()
#
#         g_vector_z = self.flatten_mesh(self.g, 'z')
#         current_vector_z = z_current.dot(g_vector_z)
#         gradient_mesh_z = self.wrap_vector(current_vector_z, 'z')
#         current_mesh_z = np.imag(np.conj(self.g) * gradient_mesh_z)
#
#         g_vector_rho = self.flatten_mesh(self.g, 'rho')
#         current_vector_rho = rho_current.dot(g_vector_rho)
#         gradient_mesh_rho = self.wrap_vector(current_vector_rho, 'rho')
#         current_mesh_rho = np.imag(np.conj(self.g) * gradient_mesh_rho)
#
#         return current_mesh_z, current_mesh_rho
#
#     def get_spline_for_mesh(self, mesh):
#         return sp.interp.RectBivariateSpline(self.z, self.chi, mesh)
#
#     @si.utils.memoize
#     def get_mesh_slicer(self, plot_limit = None):
#         """Returns a slice object that slices a mesh to the given distance of the center."""
#         if plot_limit is None:
#             mesh_slicer = (slice(None, None, 1), slice(None, None, 1))
#         else:
#             z_lim_points = round(plot_limit / self.delta_z)
#             rho_lim_points = round(plot_limit / self.delta_rho)
#             mesh_slicer = (slice(int(self.z_center_index - z_lim_points), int(self.z_center_index + z_lim_points + 1), 1), slice(0, int(rho_lim_points + 1), 1))
#
#         return mesh_slicer


class SphericalSliceMesh(QuantumMesh):
    mesh_storage_method = ('r', 'theta')
    mesh_plotter_type = mesh_plotters.SphericalSliceMeshPlotter

    def __init__(self, sim: 'sims.MeshSimulation'):
        super().__init__(sim)

        self.r = np.linspace(0, self.spec.r_bound, self.spec.r_points)
        self.theta = np.delete(np.linspace(0, u.pi, self.spec.theta_points + 1), 0)

        self.delta_r = self.r[1] - self.r[0]
        self.delta_theta = self.theta[1] - self.theta[0]
        self.inner_product_multiplier = self.delta_r * self.delta_theta

        self.r += self.delta_r / 2
        self.theta -= self.delta_theta / 2

        self.r_max = np.max(self.r)

        self.g = self.get_g_for_state(self.spec.initial_state)

        self.mesh_points = len(self.r) * len(self.theta)
        self.matrix_operator_shape = (self.mesh_points, self.mesh_points)
        self.mesh_shape = np.shape(self.r_mesh)

    @property
    @si.utils.memoize
    def r_mesh(self) -> CoordinateMesh:
        return np.meshgrid(self.r, self.theta, indexing = 'ij')[0]

    @property
    @si.utils.memoize
    def theta_mesh(self) -> CoordinateMesh:
        return np.meshgrid(self.r, self.theta, indexing = 'ij')[1]

    @property
    def z_mesh(self) -> CoordinateMesh:
        return self.r_mesh * np.cos(self.theta_mesh)

    @property
    def g_factor(self) -> np.array:
        return np.sqrt(u.twopi * np.sin(self.theta_mesh)) * self.r_mesh

    def _wrapping_direction_to_order(self, wrapping_direction: Optional[WrappingDirection]) -> Optional[str]:
        if wrapping_direction is None:
            return None
        elif wrapping_direction == WrappingDirection.R:
            return 'F'
        elif wrapping_direction == WrappingDirection.THETA:
            return 'C'
        else:
            raise ValueError(f"{wrapping_direction} is not a valid specifier for flatten_mesh (valid specifiers: 'r', 'theta')")

    @si.utils.memoize
    def get_g_for_state(self, state):
        g = self.g_factor * state(self.r_mesh, self.theta_mesh, 0)
        g /= np.sqrt(self.norm(g))
        g *= state.amplitude

        return g

    def get_spline_for_mesh(self, mesh):
        return sp.interp.RectBivariateSpline(self.r, self.theta, mesh)

    @si.utils.memoize
    def get_mesh_slicer(self, distance_from_center = None):
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if distance_from_center is None:
            mesh_slicer = slice(None, None, 1)
        else:
            r_lim_points = round(distance_from_center / self.delta_r)
            mesh_slicer = slice(0, int(r_lim_points + 1), 1)

        return mesh_slicer


class SphericalHarmonicMesh(QuantumMesh):
    mesh_storage_method = ('l', 'r')
    mesh_plotter_type = mesh_plotters.SphericalHarmonicMeshPlotter

    def __init__(self, sim: 'sims.MeshSimulation'):
        super().__init__(sim)

        self.r = np.linspace(0, self.spec.r_bound, self.spec.r_points)
        self.delta_r = self.r[1] - self.r[0]
        self.r += self.delta_r / 2
        self.r_max = np.max(self.r)
        self.inner_product_multiplier = self.delta_r

        self.l = np.array(range(self.spec.l_bound), dtype = int)
        self.theta_points = self.phi_points = self.spec.theta_points

        self.mesh_points = len(self.r) * len(self.l)
        self.mesh_shape = np.shape(self.r_mesh)

        if self.spec.use_numeric_eigenstates:
            self.analytic_to_numeric = self.get_numeric_eigenstate_basis(self.spec.numeric_eigenstate_max_energy, self.spec.numeric_eigenstate_max_angular_momentum)
            self.spec.test_states = sorted(list(self.analytic_to_numeric.values()), key = lambda x: x.energy)
            if not self.spec.initial_state.numeric:
                self.spec.initial_state = self.analytic_to_numeric[self.spec.initial_state]

            logger.warning(f'Replaced test states for {self} with numeric eigenbasis')

        self.g = self.get_g_for_state(self.spec.initial_state)

    @property
    @si.utils.memoize
    def r_mesh(self) -> CoordinateMesh:
        return np.meshgrid(self.l, self.r, indexing = 'ij')[1]

    @property
    @si.utils.memoize
    def l_mesh(self) -> CoordinateMesh:
        return np.meshgrid(self.l, self.r, indexing = 'ij')[0]

    @property
    def g_factor(self) -> np.array:
        return self.r

    def _wrapping_direction_to_order(self, wrapping_direction: Optional[WrappingDirection]) -> Optional[str]:
        if wrapping_direction is None:
            return None
        elif wrapping_direction == WrappingDirection.L:
            return 'F'
        elif wrapping_direction == WrappingDirection.R:
            return 'C'
        else:
            raise ValueError(f"{wrapping_direction} is not a valid specifier for flatten_mesh (valid specifiers: 'l', 'r')")

    def get_g_for_state(self, state: StateOrGMesh) -> GMesh:
        """
        Get g for a state.
        """
        if isinstance(state, states.QuantumState) and all(hasattr(s, 'spherical_harmonic') for s in state):
            if state.analytic and self.spec.use_numeric_eigenstates:
                try:
                    state = self.analytic_to_numeric[state]
                except (AttributeError, KeyError):
                    logger.debug(f'Analytic to numeric eigenstate lookup failed for state {state}')
            g = np.zeros(self.mesh_shape, dtype = np.complex128)

            for s in state:
                g[s.l, :] += self.get_radial_g_for_state(s)  # fill in g state-by-state to improve runtime

            return g
        else:
            raise NotImplementedError('States with non-definite angular momentum components are not currently supported by SphericalHarmonicMesh')

    @si.utils.memoize
    def get_radial_g_for_state(self, state: states.QuantumState):
        """Return the radial g function evaluated on the radial mesh for a state that has a radial function."""
        g = state.radial_function(self.r) * self.g_factor
        g /= np.sqrt(self.norm(g))
        g *= state.amplitude

        return g

    def inner_product(self, a: StateOrGMesh = None, b: StateOrGMesh = None) -> complex:
        """
        Return the inner product between two states (a and b) on the mesh.

        a and b can be QuantumStates or g_meshes.

        Parameters
        ----------
        a
            A :class:`QuantumState` or a g mesh.
        b
            A :class:`QuantumState` or a g mesh.

        Returns
        -------
        :class:`complex`
            The inner product between `a` and `b`.
        """
        if isinstance(a, states.QuantumState) and all(hasattr(s, 'spherical_harmonic') for s in a) and b is None:  # shortcut
            ip = 0

            for s in a:
                ip += np.sum(np.conj(self.get_radial_g_for_state(s)) * self.g[s.l, :])  # calculate inner product state-by-state to improve runtime

            return ip * self.inner_product_multiplier
        else:
            return super().inner_product(a, b)

    @property
    def norm_by_l(self) -> np.array:
        return np.abs(np.sum(np.conj(self.g) * self.g, axis = 1) * self.delta_r)

    def inner_product_with_plane_waves(self, thetas, wavenumbers, g: Optional[GMesh] = None):
        """
        Return the inner products for each plane wave state in the Cartesian product of thetas and wavenumbers.

        Parameters
        ----------
        thetas
        wavenumbers
        g

        Returns
        -------

        """
        if g is None:
            g = self.g

        l_mesh = self.l_mesh

        multiplier = np.sqrt(2 / u.pi) * self.g_factor * (-1j ** (l_mesh % 4)) * self.inner_product_multiplier * g

        thetas, wavenumbers = np.array(thetas), np.array(wavenumbers)
        theta_mesh, wavenumber_mesh = np.meshgrid(thetas, wavenumbers, indexing = 'ij')

        inner_product_mesh = np.zeros(np.shape(wavenumber_mesh), dtype = np.complex128)

        @si.utils.memoize
        def sph_harm(theta):
            return special.sph_harm(0, l_mesh, 0, theta)

        @si.utils.memoize
        def bessel(wavenumber):
            return special.spherical_jn(l_mesh, np.real(wavenumber * self.r_mesh))

        for (ii, theta), (jj, wavenumber) in itertools.product(enumerate(thetas), enumerate(wavenumbers)):
            inner_product_mesh[ii, jj] = np.sum(multiplier * sph_harm(theta) * bessel(wavenumber))

        return theta_mesh, wavenumber_mesh, inner_product_mesh

    def inner_product_with_plane_waves_at_infinity(self, thetas, wavenumbers, g: Optional[GMesh] = None):
        """
        Return the inner products for each plane wave state in the Cartesian product of thetas and wavenumbers.

        WARNING: NOT WORKING

        Parameters
        ----------
        thetas
        wavenumbers
        g

        Returns
        -------

        """
        raise NotImplementedError
        # l_mesh = self.l_mesh
        #
        # # multiplier = np.sqrt(2 / pi) * self.g_factor * (-1j ** (l_mesh % 4)) * self.inner_product_multiplier * g
        #
        # thetas, wavenumbers = np.array(thetas), np.array(wavenumbers)
        # theta_mesh, wavenumber_mesh = np.meshgrid(thetas, wavenumbers, indexing = 'ij')
        #
        # inner_product_mesh = np.zeros(np.shape(wavenumber_mesh), dtype = np.complex128)
        #
        # # @si.utils.memoize
        # # def sph_harm(theta):
        # #     return special.sph_harm(0, l_mesh, 0, theta)
        # #
        # # @si.utils.memoize
        # # def bessel(wavenumber):
        # #     return special.spherical_jn(l_mesh, np.real(wavenumber * self.r_mesh))
        #
        # # @si.utils.memoize
        # # def poly(l, theta):
        # #     return special.legendre(l)(np.cos(theta))
        # #
        # # @si.utils.memoize
        # # def phase(l, wavenumber):
        # #     return np.exp(1j * states.coulomb_phase_shift(l, wavenumber))
        # #
        # # # sqrt_mesh = np.sqrt((2 * l_mesh) + 1)
        # #
        # # for ii, theta in enumerate(thetas):
        # #     for jj, wavenumber in enumerate(wavenumbers):
        # #         print(ii, jj)
        # #
        # #         total = 0
        # #         for l in self.l:
        # #             total += phase(l, wavenumber) * np.sqrt((2 * l) + 1) * poly(l, theta) * self.inner_product(states.HydrogenCoulombState.from_wavenumber(wavenumber, l), g)
        # #
        # #         inner_product_mesh[ii, jj] = total / np.sqrt(4 * pi * wavenumber)
        #
        # if g is None:
        #     g = self.g
        #
        # sqrt_mesh = np.sqrt((2 * l_mesh) + 1)
        #
        # @si.utils.memoize
        # def poly(theta):
        #     return special.lpn(l_mesh, np.cos(theta))
        #
        # @si.utils.memoize
        # def phase(wavenumber):
        #     return np.exp(1j * states.coulomb_phase_shift(l_mesh, wavenumber))
        #
        # for ii, theta in enumerate(thetas):
        #     for jj, wavenumber in enumerate(wavenumbers):
        #         print(ii, jj)
        #
        #         # total = 0
        #         # for l in self.l:
        #         #     total += phase(l, wavenumber) * np.sqrt((2 * l) + 1) * poly(l, theta) * self.inner_product(states.HydrogenCoulombState.from_wavenumber(wavenumber, l), g)
        #
        #         state = states.HydrogenCoulombState.from_wavenumber(wavenumber, l = 0)
        #         for l in self.l[1:]:
        #             state += states.HydrogenCoulombState.from_wavenumber(wavenumber, l)
        #
        #         print(state)
        #         state_mesh = self.get_g_for_state(state)
        #         ip = self.inner_product(poly(theta) * phase(wavenumber) * sqrt_mesh * state_mesh, g)
        #
        #         inner_product_mesh[ii, jj] = ip / np.sqrt(4 * u.pi * wavenumber)
        #
        # return theta_mesh, wavenumber_mesh, inner_product_mesh

    def get_numeric_eigenstate_basis(self, max_energy: float, max_angular_momentum: int) -> Dict[states.QuantumState, states.NumericSphericalHarmonicState]:
        analytic_to_numeric = {}

        for l in range(max_angular_momentum + 1):
            h = self.operators.internal_hamiltonian_for_single_l(self, l = l).matrix

            estimated_spacing = u.twopi / self.r_max
            wavenumber_max = np.real(core.electron_wavenumber_from_energy(max_energy))
            number_of_eigenvectors = int(wavenumber_max / estimated_spacing)  # generate an initial guess based on roughly linear wavenumber steps between eigenvalues

            max_eigenvectors = h.shape[0] - 2  # can't generate more than this many eigenvectors using sparse linear algebra methods

            while True:
                if number_of_eigenvectors > max_eigenvectors:
                    number_of_eigenvectors = max_eigenvectors  # this will cause the loop to break after this attempt

                eigenvalues, eigenvectors = sparsealg.eigsh(h, k = number_of_eigenvectors, which = 'SA')

                if np.max(eigenvalues) > max_energy or number_of_eigenvectors == max_eigenvectors:
                    break

                number_of_eigenvectors = int(
                    number_of_eigenvectors * 1.1 * np.sqrt(np.abs(max_energy / np.max(eigenvalues))))  # based on approximate sqrt scaling of energy to wavenumber, with safety factor

            for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T):
                eigenvector /= np.sqrt(self.inner_product_multiplier * np.sum(np.abs(eigenvector) ** 2))  # normalize
                eigenvector /= self.g_factor  # go to u from R

                if eigenvalue > max_energy:  # ignore eigenvalues that are too large
                    continue
                elif eigenvalue > 0:
                    analytic_state = states.HydrogenCoulombState(energy = eigenvalue, l = l)
                    binding = states.Binding.FREE
                else:
                    n_guess = int(np.sqrt(u.rydberg / np.abs(eigenvalue)))
                    if n_guess == 0:
                        n_guess = 1
                    analytic_state = states.HydrogenBoundState(n = n_guess, l = l)
                    binding = states.Binding.BOUND

                numeric_state = states.NumericSphericalHarmonicState(
                    radial_wavefunction = eigenvector,
                    l = l,
                    m = 0,
                    energy = eigenvalue,
                    corresponding_analytic_state = analytic_state,
                    binding = binding,
                )

                analytic_to_numeric[analytic_state] = numeric_state

            logger.debug(f'Generated numerical eigenbasis for l = {l}, energy <= {u.uround(max_energy, u.eV)} eV')

        logger.debug(f'Generated numerical eigenbasis for l <= {max_angular_momentum}, energy <= {u.uround(max_energy, u.eV)} eV. Found {len(analytic_to_numeric)} states.')

        return analytic_to_numeric

    def get_radial_probability_current_density_mesh__spatial(self) -> ScalarMesh:
        r_current_operator = self.operators.r_probability_current__spatial()

        g_spatial = self.space_g_calc
        g_spatial_shape = g_spatial.shape

        # do this manually for now because the operations don't correspond to the mesh's own wrapping directions
        g_vector_r = g_spatial.flatten('F')
        gradient_vector_r = r_current_operator.matrix.dot(g_vector_r)
        gradient_mesh_r = np.reshape(gradient_vector_r, g_spatial_shape, 'F')
        current_density_mesh_r = np.imag(np.conj(g_spatial) * gradient_mesh_r)

        return current_density_mesh_r

    def _apply_length_gauge_transformation(self, vector_potential_amplitude: float, g: GMesh) -> GMesh:
        bessel_mesh = special.spherical_jn(self.l_mesh, self.spec.test_charge * vector_potential_amplitude * self.r_mesh / u.hbar)

        g_transformed = np.zeros(np.shape(g), dtype = np.complex128)
        for l_result in self.l:
            for l_outer in self.l:  # l'
                prefactor = np.sqrt(4 * u.pi * ((2 * l_outer) + 1)) * ((1j) ** (l_outer % 4)) * bessel_mesh[l_outer, :]
                for l_inner in self.l:  # l
                    g_transformed[l_result, :] += g[l_inner, :] * prefactor * core.triple_y_integral(l_outer, 0, l_result, 0, l_inner, 0)

        return g_transformed

    def gauge_transformation(self, *, g: Optional[StateOrGMesh] = None, leaving_gauge: Optional[str] = None) -> GMesh:
        g = self.state_to_mesh(g)
        if leaving_gauge is None:
            leaving_gauge = self.spec.evolution_gauge

        vamp = self.spec.electric_potential.get_vector_potential_amplitude_numeric_cumulative(self.sim.times_to_current)
        integral = integ.simps(
            y = vamp ** 2,
            x = self.sim.times_to_current,
        )

        dipole_to_velocity = np.exp(1j * integral * (self.spec.test_charge ** 2) / (2 * self.spec.test_mass * u.hbar))

        if leaving_gauge == core.Gauge.LENGTH:
            return self._apply_length_gauge_transformation(-vamp[-1], dipole_to_velocity * g)
        elif leaving_gauge == core.Gauge.VELOCITY:
            return dipole_to_velocity * self._apply_length_gauge_transformation(vamp[-1], g)

    @si.utils.memoize
    def get_mesh_slicer(self, distance_from_center: Optional[float] = None) -> slice:
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if distance_from_center is None:
            mesh_slicer = (slice(None, None, 1), slice(None, None, 1))
        else:
            r_lim_points = int(distance_from_center / self.delta_r)
            mesh_slicer = (slice(None, None, 1), slice(0, int(r_lim_points + 1), 1))

        return mesh_slicer

    @si.utils.memoize
    def get_mesh_slicer_spatial(self, distance_from_center: Optional[float] = None) -> slice:
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if distance_from_center is None:
            mesh_slicer = slice(None, None, 1)
        else:
            r_lim_points = int(distance_from_center / self.delta_r)
            mesh_slicer = slice(0, int(r_lim_points + 1), 1)

        return mesh_slicer

    @property
    @si.utils.memoize
    def theta_plot(self) -> CoordinateVector:
        return np.linspace(0, u.twopi, self.theta_points)

    @property
    @si.utils.memoize
    def theta_calc(self) -> CoordinateVector:
        return np.linspace(0, u.pi, self.theta_points)

    @property
    @si.utils.memoize
    def theta_plot_mesh(self) -> CoordinateMesh:
        return np.meshgrid(self.r, self.theta_plot, indexing = 'ij')[1]

    @property
    @si.utils.memoize
    def r_theta_mesh(self) -> CoordinateMesh:
        return np.meshgrid(self.r, self.theta_plot, indexing = 'ij')[0]

    @property
    @si.utils.memoize
    def theta_calc_mesh(self) -> CoordinateMesh:
        return np.meshgrid(self.r, self.theta_calc, indexing = 'ij')[1]

    @property
    @si.utils.memoize
    def r_theta_calc_mesh(self) -> CoordinateMesh:
        return np.meshgrid(self.r, self.theta_calc, indexing = 'ij')[0]

    @property
    @si.utils.memoize
    def _sph_harm_l_theta_plot_mesh(self):
        l_mesh, theta_mesh = np.meshgrid(self.l, self.theta_plot, indexing = 'ij')
        return special.sph_harm(0, l_mesh, 0, theta_mesh)

    @property
    @si.utils.memoize
    def _sph_harm_l_theta_calc_mesh(self):
        l_mesh, theta_mesh = np.meshgrid(self.l, self.theta_calc, indexing = 'ij')
        return special.sph_harm(0, l_mesh, 0, theta_mesh)

    def reconstruct_spatial_mesh__plot(self, mesh: WavefunctionMesh) -> GMesh:
        """Reconstruct the spatial (r, theta) representation of a mesh from the (l, r) representation."""
        # l: l, angular momentum index
        # r: r, radial position index
        # t: theta, polar angle index
        return np.einsum('lr,lt->rt', mesh, self._sph_harm_l_theta_plot_mesh)

    def reconstruct_spatial_mesh__calc(self, mesh: WavefunctionMesh) -> GMesh:
        """Reconstruct the spatial (r, theta) representation of a mesh from the (l, r) representation."""
        # l: l, angular momentum index
        # r: r, radial position index
        # t: theta, polar angle index
        return np.einsum('lr,lt->rt', mesh, self._sph_harm_l_theta_calc_mesh)

    @property
    @si.utils.watched_memoize(lambda s: s.sim.time)
    def space_g(self) -> GMesh:
        return self.reconstruct_spatial_mesh__plot(self.g)

    @property
    @si.utils.watched_memoize(lambda s: s.sim.time)
    def space_g_calc(self) -> GMesh:
        return self.reconstruct_spatial_mesh__calc(self.g)

    @property
    def space_psi(self) -> PsiMesh:
        return self.space_g / self.r_theta_mesh

    @property
    def g2(self) -> G2Mesh:
        return np.abs(self.space_g) ** 2

    @property
    def psi2(self) -> Psi2Mesh:
        return np.abs(self.space_psi) ** 2
