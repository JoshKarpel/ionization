import itertools
import logging
from typing import Union, Optional, Iterable, NewType, Tuple, Dict
import abc

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as nfft
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as sparsealg
import scipy.special as special
import scipy.integrate as integ

import simulacra as si
import simulacra.units as u

from .. import states, vis, core, cy, exceptions
from . import sims, evolution_methods

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CoordinateMesh = NewType('CoordinateMesh', np.array)
CoordinateVector = NewType('CoordinateVector', np.array)

GMesh = NewType('GMesh', np.array)
G2Mesh = NewType('G2Mesh', np.array)
PsiMesh = NewType('PsiMesh', np.array)
Psi2Mesh = NewType('Psi2Mesh', np.array)
WavefunctionMesh = Union[GMesh, G2Mesh, PsiMesh, Psi2Mesh]

GVector = NewType('GVector', np.array)
G2Vector = NewType('G2Vector', np.array)
PsiVector = NewType('PsiVector', np.array)
Psi2Vector = NewType('Psi2Vector', np.array)
WavefunctionVector = Union[GVector, G2Vector, PsiVector, Psi2Vector]

StateOrGMesh = Optional[Union[states.QuantumState, GMesh]]  # None => the current g mesh

SparseMatrixOperator = NewType('SparseMatrixOperator', sparse.dia_matrix)


def c_l(l) -> float:
    """a particular set of 3j coefficients for SphericalHarmonicMesh"""
    return (l + 1) / np.sqrt(((2 * l) + 1) * ((2 * l) + 3))


def add_to_diagonal_sparse_matrix_diagonal(dia_matrix: SparseMatrixOperator, value = 1) -> sparse.dia_matrix:
    s = dia_matrix.copy()
    s.setdiag(s.diagonal() + value)
    return s


def add_to_diagonal_sparse_matrix_diagonal_inplace(dia_matrix: SparseMatrixOperator, value = 1) -> sparse.dia_matrix:
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


class MeshOperator(abc.ABC):
    def __init__(self, operator: SparseMatrixOperator, *, wrapping_direction: str):
        self.operator = operator
        self.wrapping_direction = wrapping_direction

    def __repr__(self):
        return f"{self.__class__.__name__}(operator = {repr(self.operator)}, wrapping_direction = '{self.wrapping_direction}')"

    def apply(self, mesh: 'QuantumMesh', g: GVector, current_wrapping_direction):
        if current_wrapping_direction != self.wrapping_direction:
            g = mesh.flatten_mesh(mesh.wrap_vector(g, current_wrapping_direction), self.wrapping_direction)

        result = self._apply(g)

        return result, self.wrapping_direction

    @abc.abstractmethod
    def _apply(self, g: GVector):
        raise NotImplementedError


class DotOperator(MeshOperator):
    def _apply(self, g: GVector) -> GVector:
        return self.operator.dot(g)


class TDMAOperator(MeshOperator):
    def _apply(self, g: GVector) -> GVector:
        return cy.tdma(self.operator, g)


class SimilarityOperator(DotOperator):
    def __init__(self, operator: SparseMatrixOperator, *, wrapping_direction: str, parity: str):
        super().__init__(operator, wrapping_direction = wrapping_direction)

        self.parity = parity
        self.transform = getattr(self, f'u_{self.parity}_g')

    def __repr__(self):
        op_repr = repr(self.operator).replace('\n', '')
        return f"{self.__class__.__name__}({op_repr}, wrapping_direction = '{self.wrapping_direction}', parity = '{self.parity}')"

    def u_even_g(self, g):
        stack = []
        if len(g) % 2 == 0:
            for a, b in si.utils.grouper(g, 2, fill_value = 0):
                stack += (a + b, a - b)
        else:
            for a, b in si.utils.grouper(g[:-1], 2, fill_value = 0):
                stack += (a + b, a - b)
            stack.append(np.sqrt(2) * g[-1])

        return np.hstack(stack) / np.sqrt(2)

    def u_odd_g(self, g):
        stack = [np.sqrt(2) * g[0]]
        if len(g) % 2 == 0:
            for a, b in si.utils.grouper(g[1:-1], 2, fill_value = 0):
                stack += (a + b, a - b)
            stack.append(np.sqrt(2) * g[-1])
        else:
            for a, b in si.utils.grouper(g[1:], 2, fill_value = 0):
                stack += (a + b, a - b)

        return np.hstack(stack) / np.sqrt(2)

    def apply(self, mesh, g, current_wrapping_direction: str):
        g_wrapped = mesh.wrap_vector(g, current_wrapping_direction)
        g_transformed = self.transform(g_wrapped)  # this wraps the mesh along j!
        g_flat = mesh.flatten_mesh(g_transformed, self.wrapping_direction)
        g_flat = self._apply(g_flat)
        g_wrap = mesh.wrap_vector(g_flat, self.wrapping_direction)
        result = self.transform(g_wrap)  # this wraps the mesh along j!

        return result, self.wrapping_direction


def apply_operators(mesh, g: GMesh, *operators: MeshOperator):
    """Operators should be entered in operation (the order they would act on something on their right)"""
    current_wrapping_direction = None

    for operator in operators:
        g, current_wrapping_direction = operator.apply(mesh, g, current_wrapping_direction)

    return mesh.wrap_vector(g, current_wrapping_direction)


class QuantumMesh(abc.ABC):
    def __init__(self, simulation: 'sims.MeshSimulation'):
        self.sim = simulation
        self.spec = simulation.spec

        self.g = None
        self.inner_product_multiplier = None

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
    def _wrapping_direction_to_order(self, wrapping_direction: WrappingDirection) -> Optional[str]:
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

    def norm(self, state: StateOrGMesh = None) -> float:
        return np.abs(self.inner_product(a = state, b = state))

    def __abs__(self) -> float:
        return self.norm()

    def energy_expectation_value(self, include_interaction: bool = False):
        raise NotImplementedError

    def radial_position_expectation_value(self) -> float:
        return np.real(self.inner_product(b = self.r_mesh * self.g)) / self.norm()

    def z_dipole_moment_inner_product(self, a: StateOrGMesh = None, b: StateOrGMesh = None):
        raise NotImplementedError

    @property
    def psi(self) -> PsiMesh:
        return self.g / self.g_factor

    @property
    def g2(self) -> G2Mesh:
        return np.abs(self.g) ** 2

    @property
    def psi2(self) -> Psi2Mesh:
        return np.abs(self.psi) ** 2

    @si.utils.memoize
    def get_kinetic_energy_matrix_operators(self):
        try:
            return getattr(self, f'_get_kinetic_energy_matrix_operators_{self.spec.evolution_equations}')()
        except AttributeError:
            raise NotImplementedError

    def get_internal_hamiltonian_matrix_operators(self):
        raise NotImplementedError

    def get_interaction_hamiltonian_matrix_operators(self):
        try:
            return getattr(self, f'_get_interaction_hamiltonian_matrix_operators_{self.spec.evolution_gauge}')()
        except AttributeError:
            raise NotImplementedError

    def evolve(self, time_step: complex):
        self.g = self.spec.evolution_method.evolve(self, self.g, time_step)
        self.g *= self.spec.mask(r = self.r_mesh)

    def get_mesh_slicer(self, plot_limit: float):
        raise NotImplementedError

    def attach_mesh_to_axis(
            self,
            axis: plt.Axes,
            mesh: WavefunctionMesh,
            distance_unit: u.Unit = 'bohr_radius',
            colormap = plt.get_cmap('inferno'),
            norm = si.vis.AbsoluteRenormalize(),
            shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
            plot_limit: Optional[float] = None,
            slicer: str = 'get_mesh_slicer',
            **kwargs):
        raise NotImplementedError

    def attach_g2_to_axis(
            self,
            axis: plt.Axes,
            **kwargs):
        return self.attach_mesh_to_axis(axis, self.g2, **kwargs)

    def attach_psi2_to_axis(
            self,
            axis,
            **kwargs):
        return self.attach_mesh_to_axis(axis, self.psi2, **kwargs)

    def attach_g_to_axis(
            self,
            axis: plt.Axes,
            colormap = plt.get_cmap('richardson'),
            norm = None,
            **kwargs):
        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        return self.attach_mesh_to_axis(
            axis,
            self.g,
            colormap = colormap,
            norm = norm,
            **kwargs
        )

    def attach_psi_to_axis(
            self,
            axis: plt.Axes,
            colormap = plt.get_cmap('richardson'),
            norm = None,
            **kwargs):
        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.psi) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        return self.attach_mesh_to_axis(
            axis,
            self.psi,
            colormap = colormap,
            norm = norm,
            **kwargs
        )

    def update_mesh(
            self,
            colormesh,
            updated_mesh,
            plot_limit: Optional[float] = None,
            shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
            slicer: str = 'get_mesh_slicer',
            **kwargs):
        _slice = getattr(self, slicer)(plot_limit)
        updated_mesh = updated_mesh[_slice]

        try:
            if shading == si.vis.ColormapShader.FLAT:
                updated_mesh = updated_mesh[:-1, :-1]
            colormesh.set_array(updated_mesh.ravel())
        except AttributeError:  # if the mesh is 1D we can't .ravel() it and instead should just set the y data with the mesh
            colormesh.set_ydata(updated_mesh)

    def update_g2_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.g2, **kwargs)

    def update_psi2_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.psi2, **kwargs)

    def update_g_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.g, **kwargs)

    def update_psi_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.psi, **kwargs)

    def plot_mesh(
            self,
            mesh: WavefunctionMesh,
            name: str = '',
            title: Optional[str] = None,
            distance_unit: str = 'bohr_radius',
            colormap = vis.COLORMAP_WAVEFUNCTION,
            norm = si.vis.AbsoluteRenormalize(),
            shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
            plot_limit: Optional[float] = None,
            slicer: str = 'get_mesh_slicer',
            **kwargs):
        """kwargs go to figman"""
        raise NotImplementedError

    def plot_g2(
            self,
            name_postfix: str = '',
            title: Optional[str] = None,
            **kwargs):
        if title is None:
            title = r'$|g|^2$'
        name = 'g2' + name_postfix

        self.plot_mesh(self.g2, name = name, title = title, **kwargs)

    def plot_psi2(
            self,
            name_postfix: str = '',
            **kwargs):
        title = r'$|\Psi|^2$'
        name = 'psi2' + name_postfix

        self.plot_mesh(self.psi2, name = name, title = title, **kwargs)

    def plot_g(
            self,
            title: Optional[str] = None,
            name_postfix: str = '',
            colormap = plt.get_cmap('richardson'),
            norm = None,
            **kwargs):
        if title is None:
            title = r'$g$'
        name = 'g' + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        self.plot_mesh(
            self.g, name = name, title = title,
            colormap = colormap,
            norm = norm,
            show_colorbar = False,
            **kwargs
        )

    def plot_psi(
            self,
            name_postfix = '',
            colormap = plt.get_cmap('richardson'),
            norm = None,
            **kwargs):
        title = r'$g$'
        name = 'g' + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.psi) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        self.plot_mesh(
            self.psi, name = name, title = title,
            colormap = colormap,
            norm = norm,
            show_colorbar = False,
            **kwargs
        )


class LineMesh(QuantumMesh):
    mesh_storage_method = ['x']

    def __init__(self, simulation: 'sims.MeshSimulation'):
        super().__init__(simulation)

        self.x_mesh = np.linspace(-self.spec.x_bound, self.spec.x_bound, self.spec.x_points)
        self.delta_x = np.abs(self.x_mesh[1] - self.x_mesh[0])
        self.x_center_index = si.utils.find_nearest_entry(self.x_mesh, 0).index

        self.wavenumbers = u.twopi * nfft.fftfreq(len(self.x_mesh), d = self.delta_x)
        self.delta_k = np.abs(self.wavenumbers[1] - self.wavenumbers[0])

        self.inner_product_multiplier = self.delta_x
        self.g_factor = 1

        if self.spec.use_numeric_eigenstates:
            logger.debug('Calculating numeric eigenstates')

            self.analytic_to_numeric = self._get_numeric_eigenstate_basis(self.spec.number_of_numeric_eigenstates)
            self.spec.test_states = sorted(list(self.analytic_to_numeric.values()), key = lambda x: x.energy)
            self.spec.initial_state = self.analytic_to_numeric[self.spec.initial_state]

            logger.warning(f'Replaced test states for {self} with numeric eigenbasis')

        self.g = self.get_g_for_state(self.spec.initial_state)

        self.free_evolution_prefactor = -1j * (u.hbar / (2 * self.spec.test_mass)) * (self.wavenumbers ** 2)  # hbar^2/2m / hbar
        self.wavenumber_mask = np.where(np.abs(self.wavenumbers) < self.spec.fft_cutoff_wavenumber, 1, 0)

    @property
    def r_mesh(self):
        return self.x_mesh

    @property
    def energies(self):
        return ((self.wavenumbers * u.hbar) ** 2) / (2 * self.spec.test_mass)

    def _wrapping_direction_to_order(self, wrapping_direction: WrappingDirection) -> Optional[str]:
        return None

    @si.utils.memoize
    def get_g_for_state(self, state):
        if state.analytic and self.spec.use_numeric_eigenstates:
            try:
                state = self.analytic_to_numeric[state]
            except (AttributeError, KeyError):
                logger.debug(f'Analytic to numeric eigenstate lookup failed for state {state}')

        g = state(self.x_mesh)
        g /= np.sqrt(self.norm(g))
        g *= state.amplitude

        return g

    def energy_expectation_value(self, include_interaction = False):
        potential = self.inner_product(b = self.spec.internal_potential(r = self.x_mesh, distance = self.x_mesh, test_charge = self.spec.test_charge) * self.g)

        power_spectrum = np.abs(self.fft(self.g)) ** 2
        kinetic = np.sum((((u.hbar * self.wavenumbers) ** 2) / (2 * self.spec.test_mass)) * power_spectrum) / np.sum(power_spectrum)

        energy = potential + kinetic

        if include_interaction:
            energy += self.inner_product(
                b = self.spec.electric_potential(t = self.sim.time, r = self.x_mesh, distance = self.x_mesh, distance_along_polarization = self.x_mesh, test_charge = self.spec.test_charge) * self.g)

        return np.real(energy) / self.norm()

    def z_dipole_moment_inner_product(self, a: StateOrGMesh = None, b: StateOrGMesh = None):
        return self.spec.test_charge * self.inner_product(a = a, b = self.x_mesh * self.state_to_mesh(b))

    def fft(self, mesh = None):
        if mesh is None:
            mesh = self.g

        return nfft.fft(mesh, norm = 'ortho')

    def ifft(self, mesh):
        return nfft.ifft(mesh, norm = 'ortho')

    def _evolve_potential(self, time_step: complex):
        pot = self.spec.internal_potential(t = self.sim.time, r = self.x_mesh, distance = self.x_mesh, test_charge = self.spec.test_charge)
        pot += self.spec.electric_potential(t = self.sim.time, r = self.x_mesh, distance = self.x_mesh, distance_along_polarization = self.x_mesh, test_charge = self.spec.test_charge)
        self.g *= np.exp(-1j * time_step * pot / u.hbar)

    def _evolve_free(self, time_step: complex):
        self.g = self.ifft(self.fft(self.g) * np.exp(self.free_evolution_prefactor * time_step) * self.wavenumber_mask)

    def _get_kinetic_energy_matrix_operators_HAM(self):
        prefactor = -(u.hbar ** 2) / (2 * self.spec.test_mass * (self.delta_x ** 2))

        diag = -2 * prefactor * np.ones(len(self.x_mesh), dtype = np.complex128)
        off_diag = prefactor * np.ones(len(self.x_mesh) - 1, dtype = np.complex128)

        return sparse.diags((off_diag, diag, off_diag), (-1, 0, 1))

    @si.utils.memoize
    def get_internal_hamiltonian_matrix_operators(self):
        kinetic_x = self.get_kinetic_energy_matrix_operators().copy()

        kinetic_x = add_to_diagonal_sparse_matrix_diagonal(kinetic_x, self.spec.internal_potential(t = self.sim.time, r = self.x_mesh, distance = self.x_mesh, test_charge = self.spec.test_charge))

        return kinetic_x

    def _get_interaction_hamiltonian_matrix_operators_LEN(self):
        epot = self.spec.electric_potential(t = self.sim.time, distance_along_polarization = self.x_mesh, test_charge = self.spec.test_charge)

        return sparse.diags([epot], offsets = [0])

    @si.utils.memoize
    def _get_interaction_hamiltonian_matrix_operators_without_field_VEL(self):
        prefactor = 1j * u.hbar * (self.spec.test_charge / self.spec.test_mass) / (2 * self.delta_x)
        offdiag = prefactor * np.ones(self.spec.x_points - 1, dtype = np.complex128)

        return sparse.diags([-offdiag, offdiag], offsets = [-1, 1])

    def _get_interaction_hamiltonian_matrix_operators_VEL(self):
        return self._get_interaction_hamiltonian_matrix_operators_without_field_VEL() * self.spec.electric_potential.get_vector_potential_amplitude_numeric(self.sim.times_to_current)

    def _evolve_CN(self, time_step):
        """Crank-Nicholson evolution in the Length gauge."""
        tau = time_step / (2 * u.hbar)

        interaction_operator = self.get_interaction_hamiltonian_matrix_operators()

        hamiltonian_x = self.get_internal_hamiltonian_matrix_operators()
        hamiltonian = 1j * tau * sparse.dia_matrix(hamiltonian_x + interaction_operator)

        ham_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian, 1)
        ham_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian, 1)

        operators = [
            DotOperator(ham_explicit, wrapping_direction = None),
            TDMAOperator(ham_implicit, wrapping_direction = None),
        ]

        self.g = apply_operators(self, self.g, *operators)

    def _make_split_operator_evolution_operators(self, interaction_hamiltonians_matrix_operators, tau: float):
        return getattr(self, f'_make_split_operator_evolution_operators_{self.spec.evolution_gauge}')(interaction_hamiltonians_matrix_operators, tau)

    def _make_split_operator_evolution_operators_LEN(self, interaction_hamiltonians_matrix_operators, tau: float):
        return [DotOperator(sparse.diags([np.exp(-1j * interaction_hamiltonians_matrix_operators.data[0] * tau)], offsets = [0]), wrapping_direction = None)]

    def _make_split_operator_evolution_operators_VEL(self, interaction_hamiltonians_matrix_operators, tau: float):
        a = interaction_hamiltonians_matrix_operators.data[-1][1:] * tau * (-1j)

        a_even, a_odd = a[::2], a[1::2]

        even_diag = np.zeros(len(a) + 1, dtype = np.complex128)
        even_offdiag = np.zeros(len(a), dtype = np.complex128)
        odd_diag = np.zeros(len(a) + 1, dtype = np.complex128)
        odd_offdiag = np.zeros(len(a), dtype = np.complex128)

        if len(self.x_mesh) % 2 != 0:
            even_diag[:-1] = np.cos(a_even).repeat(2)
            even_diag[-1] = 1

            even_offdiag[::2] = np.sin(a_even)

            odd_diag[0] = 1
            odd_diag[1:] = np.cos(a_odd).repeat(2)

            odd_offdiag[1::2] = np.sin(a_odd)
        else:
            even_diag[:] = np.cos(a_even).repeat(2)

            even_offdiag[::2] = np.sin(a_even)

            odd_diag[0] = odd_diag[-1] = 1
            odd_diag[1:-1] = np.cos(a_odd).repeat(2)

            odd_offdiag[1::2] = np.sin(a_odd)

        even = sparse.diags([-even_offdiag, even_diag, even_offdiag], offsets = [-1, 0, 1])
        odd = sparse.diags([-odd_offdiag, odd_diag, odd_offdiag], offsets = [-1, 0, 1])

        return (
            DotOperator(even, wrapping_direction = None),
            DotOperator(odd, wrapping_direction = None),
        )

    def _evolve_SO(self, time_step: complex):
        """Split-Operator evolution in the Length gauge."""
        tau = time_step / (2 * u.hbar)

        hamiltonian_x = self.get_internal_hamiltonian_matrix_operators()

        ham_x_explicit = add_to_diagonal_sparse_matrix_diagonal(-1j * tau * hamiltonian_x, 1)
        ham_x_implicit = add_to_diagonal_sparse_matrix_diagonal(1j * tau * hamiltonian_x, 1)

        split_operators = self._make_split_operator_evolution_operators(self.get_interaction_hamiltonian_matrix_operators(), tau)

        operators = [
            *split_operators,
            DotOperator(ham_x_explicit, wrapping_direction = None),
            TDMAOperator(ham_x_implicit, wrapping_direction = None),
            *reversed(split_operators),
        ]

        self.g = apply_operators(self, self.g, *operators)

    def _evolve_S(self, time_step: complex):
        """Spectral evolution in the Length gauge."""
        self._evolve_potential(time_step / 2)
        self._evolve_free(time_step)  # splitting order chosen for computational efficiency (only one FFT per time step)
        self._evolve_potential(time_step / 2)

    def _get_numeric_eigenstate_basis(self, number_of_eigenstates: int):
        analytic_to_numeric = {}

        h = self.get_internal_hamiltonian_matrix_operators()

        eigenvalues, eigenvectors = sparsealg.eigsh(h, k = number_of_eigenstates, which = 'SA')

        for nn, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
            eigenvector /= np.sqrt(self.inner_product_multiplier * np.sum(np.abs(eigenvector) ** 2))  # normalize

            try:
                bound = True
                analytic_state = self.spec.analytic_eigenstate_type.from_potential(self.spec.internal_potential, self.spec.test_mass,
                                                                                   n = nn + self.spec.analytic_eigenstate_type.smallest_n)
            except states.IllegalQuantumState:
                bound = False
                analytic_state = states.OneDFreeParticle.from_energy(eigenvalue, mass = self.spec.test_mass)

            numeric_state = states.NumericOneDState(eigenvector, eigenvalue, bound = bound, analytic_state = analytic_state)
            if analytic_state is None:
                analytic_state = numeric_state

            analytic_to_numeric[analytic_state] = numeric_state

        logger.debug(f'Generated numerical eigenbasis with {len(analytic_to_numeric)} states')

        return analytic_to_numeric

    def gauge_transformation(self, *, g: GMesh = None, leaving_gauge: Optional[str] = None):
        if g is None:
            g = self.g
        if leaving_gauge is None:
            leaving_gauge = self.spec.evolution_gauge

        vamp = self.spec.electric_potential.get_vector_potential_amplitude_numeric_cumulative(self.sim.times_to_current)
        integral = integ.simps(y = vamp ** 2,
                               x = self.sim.times_to_current)

        dipole_to_velocity = np.exp(1j * integral * (self.spec.test_charge ** 2) / (2 * self.spec.test_mass * u.hbar))
        dipole_to_length = np.exp(-1j * self.spec.test_charge * vamp[-1] * self.x_mesh / u.hbar)

        if leaving_gauge == 'LEN':
            return np.conj(dipole_to_length) * dipole_to_velocity * g
        elif leaving_gauge == 'VEL':
            return dipole_to_length * np.conj(dipole_to_velocity) * g

    def get_mesh_slicer(self, plot_limit: Optional[float]):
        if plot_limit is None:
            mesh_slicer = slice(None, None, 1)
        else:
            x_lim_points = round(plot_limit / self.delta_x)
            mesh_slicer = slice(int(self.x_center_index - x_lim_points), int(self.x_center_index + x_lim_points + 1), 1)

        return mesh_slicer

    def attach_mesh_to_axis(
            self,
            axis: plt.Axes,
            mesh,
            distance_unit: u.Unit = 'bohr_radius',
            colormap = plt.get_cmap('inferno'),
            norm = si.vis.AbsoluteRenormalize(),
            shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
            plot_limit = None,
            slicer = 'get_mesh_slicer',
            **kwargs):
        unit_value, _ = u.get_unit_value_and_latex_from_unit(distance_unit)

        _slice = getattr(self, slicer)(plot_limit)

        line, = axis.plot(self.x_mesh[_slice] / unit_value, norm(mesh[_slice]), **kwargs)

        return line

    def plot_mesh(self, mesh, distance_unit: u.Unit = 'nm', **kwargs):
        si.vis.xy_plot(self.sim.name + '_' + kwargs.pop('name'),
                       self.x_mesh,
                       mesh,
                       x_label = 'Distance $x$',
                       x_unit_value = distance_unit,
                       **kwargs)

    def update_mesh(self, colormesh, updated_mesh, norm = None, **kwargs):
        if norm is not None:
            updated_mesh = norm(updated_mesh)

        super().update_mesh(colormesh, updated_mesh, **kwargs)

    def plot_fft(self):
        raise NotImplementedError

    def attach_fft_to_axis(
            self,
            axis,
            distance_unit = 'per_nm',
            norm = si.vis.AbsoluteRenormalize(),
            plot_limit = None,
            slicer = 'get_mesh_slicer',
            **kwargs):
        unit_value, _ = u.get_unit_value_and_latex_from_unit(distance_unit)

        _slice = getattr(self, slicer)(plot_limit)

        line, = axis.plot(self.wavenumbers[_slice] / unit_value, norm(self.fft()[_slice]), **kwargs)

        return line

    def update_fft_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.fft(), **kwargs)


class CylindricalSliceMesh(QuantumMesh):
    mesh_storage_method = ('z', 'rho')

    def __init__(self, simulation: 'sims.MeshSimulation'):
        super().__init__(simulation)

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
    def z_mesh(self):
        return np.meshgrid(self.z, self.rho, indexing = 'ij')[0]

    @property
    @si.utils.memoize
    def rho_mesh(self):
        return np.meshgrid(self.z, self.rho, indexing = 'ij')[1]

    @property
    def g_factor(self):
        return np.sqrt(u.twopi * self.rho_mesh)

    @property
    def r_mesh(self):
        return np.sqrt((self.z_mesh ** 2) + (self.rho_mesh ** 2))

    @property
    def theta_mesh(self):
        return np.arccos(self.z_mesh / self.r_mesh)

    @property
    def sin_theta_mesh(self):
        return np.sin(self.theta_mesh)

    @property
    def cos_theta_mesh(self):
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
    def get_g_for_state(self, state):
        g = self.g_factor * state(self.r_mesh, self.theta_mesh, 0)
        g /= np.sqrt(self.norm(g))
        g *= state.amplitude

        return g

    def z_dipole_moment_inner_product(self, a = None, b = None):
        return self.spec.test_charge * self.inner_product(a = a, b = self.z_mesh * self.state_to_mesh(b))

    def _get_kinetic_energy_matrix_operators_HAM(self):
        """Get the mesh kinetic energy operator matrices for z and rho."""
        z_prefactor = -(u.hbar ** 2) / (2 * self.spec.test_mass * (self.delta_z ** 2))
        rho_prefactor = -(u.hbar ** 2) / (2 * self.spec.test_mass * (self.delta_rho ** 2))

        z_diagonal = z_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        z_offdiagonal = z_prefactor * np.array([1 if (z_index + 1) % self.spec.z_points != 0 else 0 for z_index in range(self.mesh_points - 1)], dtype = np.complex128)

        @si.utils.memoize
        def c(j):
            return j / np.sqrt((j ** 2) - 0.25)

        rho_diagonal = rho_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        rho_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for rho_index in range(self.mesh_points - 1):
            if (rho_index + 1) % self.spec.rho_points != 0:
                j = (rho_index % self.spec.rho_points) + 1  # get j for the upper diagonal
                rho_offdiagonal[rho_index] = c(j)
        rho_offdiagonal *= rho_prefactor

        z_kinetic = sparse.diags([z_offdiagonal, z_diagonal, z_offdiagonal], offsets = (-1, 0, 1))
        rho_kinetic = sparse.diags([rho_offdiagonal, rho_diagonal, rho_offdiagonal], offsets = (-1, 0, 1))

        return z_kinetic, rho_kinetic

    @si.utils.memoize
    def get_internal_hamiltonian_matrix_operators(self):
        """Get the mesh internal Hamiltonian matrix operators for z and rho."""
        kinetic_z, kinetic_rho = self.get_kinetic_energy_matrix_operators()
        potential_mesh = self.spec.internal_potential(r = self.r_mesh, test_charge = self.spec.test_charge)

        kinetic_z = add_to_diagonal_sparse_matrix_diagonal(kinetic_z, value = 0.5 * self.flatten_mesh(potential_mesh, 'z'))
        kinetic_rho = add_to_diagonal_sparse_matrix_diagonal(kinetic_rho, value = 0.5 * self.flatten_mesh(potential_mesh, 'rho'))

        return kinetic_z, kinetic_rho

    def _get_interaction_hamiltonian_matrix_operators_LEN(self):
        """Get the interaction term calculated from the Lagrangian evolution equations."""
        electric_potential_energy_mesh = self.spec.electric_potential(t = self.sim.time, distance_along_polarization = self.z_mesh, test_charge = self.spec.test_charge)

        interaction_hamiltonian_z = sparse.diags(self.flatten_mesh(electric_potential_energy_mesh, 'z'))
        interaction_hamiltonian_rho = sparse.diags(self.flatten_mesh(electric_potential_energy_mesh, 'rho'))

        return interaction_hamiltonian_z, interaction_hamiltonian_rho

    def _get_interaction_hamiltonian_matrix_operators_VEL(self):
        # vector_potential_amplitude = -self.spec.electric_potential.get_electric_field_integral_numeric_cumulative(self.sim.times_to_current)
        raise NotImplementedError

    def tg_mesh(self, use_abs_g = False):
        hamiltonian_z, hamiltonian_rho = self.get_kinetic_energy_matrix_operators()

        if use_abs_g:
            g = np.abs(self.g)
        else:
            g = self.g

        g_vector_z = self.flatten_mesh(g, 'z')
        hg_vector_z = hamiltonian_z.dot(g_vector_z)
        hg_mesh_z = self.wrap_vector(hg_vector_z, 'z')

        g_vector_rho = self.flatten_mesh(g, 'rho')
        hg_vector_rho = hamiltonian_rho.dot(g_vector_rho)
        hg_mesh_rho = self.wrap_vector(hg_vector_rho, 'rho')

        return hg_mesh_z + hg_mesh_rho

    def hg_mesh(self, include_interaction = False):
        hamiltonian_z, hamiltonian_rho = self.get_internal_hamiltonian_matrix_operators()

        g_vector_z = self.flatten_mesh(self.g, 'z')
        hg_vector_z = hamiltonian_z.dot(g_vector_z)
        hg_mesh_z = self.wrap_vector(hg_vector_z, 'z')

        g_vector_rho = self.flatten_mesh(self.g, 'rho')
        hg_vector_rho = hamiltonian_rho.dot(g_vector_rho)
        hg_mesh_rho = self.wrap_vector(hg_vector_rho, 'rho')

        if include_interaction:
            raise NotImplementedError

        return hg_mesh_z + hg_mesh_rho

    def energy_expectation_value(self, include_interaction = False):
        return np.real(self.inner_product(b = self.hg_mesh(include_interaction = include_interaction))) / self.norm()

    @si.utils.memoize
    def _get_probability_current_matrix_operators(self):
        """Get the mesh probability current operators for z and rho."""
        z_prefactor = u.hbar / (4 * u.pi * self.spec.test_mass * self.delta_rho * self.delta_z)
        rho_prefactor = u.hbar / (4 * u.pi * self.spec.test_mass * (self.delta_rho ** 2))

        # construct the diagonals of the z probability current matrix operator
        z_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for z_index in range(0, self.mesh_points - 1):
            if (z_index + 1) % self.spec.z_points == 0:  # detect edge of mesh
                z_offdiagonal[z_index] = 0
            else:
                j = z_index // self.spec.z_points
                z_offdiagonal[z_index] = 1 / (j + 0.5)
        z_offdiagonal *= z_prefactor

        @si.utils.memoize
        def d(j):
            return 1 / np.sqrt((j ** 2) - 0.25)

        # construct the diagonals of the rho probability current matrix operator
        rho_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for rho_index in range(0, self.mesh_points - 1):
            if (rho_index + 1) % self.spec.rho_points == 0:  # detect edge of mesh
                rho_offdiagonal[rho_index] = 0
            else:
                j = (rho_index % self.spec.rho_points) + 1
                rho_offdiagonal[rho_index] = d(j)
        rho_offdiagonal *= rho_prefactor

        z_current = sparse.diags([-z_offdiagonal, z_offdiagonal], offsets = [-1, 1])
        rho_current = sparse.diags([-rho_offdiagonal, rho_offdiagonal], offsets = [-1, 1])

        return z_current, rho_current

    def get_probability_current_vector_field(self):
        z_current, rho_current = self._get_probability_current_matrix_operators()

        g_vector_z = self.flatten_mesh(self.g, 'z')
        current_vector_z = z_current.dot(g_vector_z)
        gradient_mesh_z = self.wrap_vector(current_vector_z, 'z')
        current_mesh_z = np.imag(np.conj(self.g) * gradient_mesh_z)

        g_vector_rho = self.flatten_mesh(self.g, 'rho')
        current_vector_rho = rho_current.dot(g_vector_rho)
        gradient_mesh_rho = self.wrap_vector(current_vector_rho, 'rho')
        current_mesh_rho = np.imag(np.conj(self.g) * gradient_mesh_rho)

        return current_mesh_z, current_mesh_rho

    def get_spline_for_mesh(self, mesh):
        return sp.interp.RectBivariateSpline(self.z, self.rho, mesh)

    @si.utils.memoize
    def get_mesh_slicer(self, plot_limit = None):
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if plot_limit is None:
            mesh_slicer = (slice(None, None, 1), slice(None, None, 1))
        else:
            z_lim_points = round(plot_limit / self.delta_z)
            rho_lim_points = round(plot_limit / self.delta_rho)
            mesh_slicer = (slice(int(self.z_center_index - z_lim_points), int(self.z_center_index + z_lim_points + 1), 1), slice(0, int(rho_lim_points + 1), 1))

        return mesh_slicer

    def attach_mesh_to_axis(
            self,
            axis: plt.Axes,
            mesh,
            distance_unit: u.Unit = 'bohr_radius',
            colormap = plt.get_cmap('inferno'),
            norm = si.vis.AbsoluteRenormalize(),
            shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
            plot_limit = None,
            slicer = 'get_mesh_slicer',
            **kwargs):
        unit_value, _ = u.get_unit_value_and_latex_from_unit(distance_unit)

        _slice = getattr(self, slicer)(plot_limit)

        color_mesh = axis.pcolormesh(
            self.z_mesh[_slice] / unit_value,
            self.rho_mesh[_slice] / unit_value,
            mesh[_slice],
            shading = shading,
            cmap = colormap,
            norm = norm,
            **kwargs
        )

        return color_mesh

    def attach_probability_current_to_axis(self, axis: plt.Axes, plot_limit = None, distance_unit: u.Unit = 'bohr_radius'):
        unit_value, _ = u.get_unit_value_and_latex_from_unit(distance_unit)

        current_mesh_z, current_mesh_rho = self.get_probability_current_vector_field()

        current_mesh_z *= self.delta_z
        current_mesh_rho *= self.delta_rho

        skip_count = int(self.z_mesh.shape[0] / 50), int(self.z_mesh.shape[1] / 50)
        skip = (slice(None, None, skip_count[0]), slice(None, None, skip_count[1]))
        normalization = np.max(np.sqrt(current_mesh_z ** 2 + current_mesh_rho ** 2)[skip])
        if normalization == 0 or normalization is np.NaN:
            normalization = 1

        quiv = axis.quiver(
            self.z_mesh[self.get_mesh_slicer(plot_limit)][skip] / unit_value,
            self.rho_mesh[self.get_mesh_slicer(plot_limit)][skip] / unit_value,
            current_mesh_z[self.get_mesh_slicer(plot_limit)][skip] / normalization,
            current_mesh_rho[self.get_mesh_slicer(plot_limit)][skip] / normalization,
            pivot = 'middle',
            units = 'width',
            scale = 10,
            scale_units = 'width',
            width = 0.0015,
            alpha = 0.5
        )

        return quiv

    def plot_mesh(
            self,
            mesh,
            name: str = '',
            title: Optional[str] = None,
            distance_unit: u.Unit = 'bohr_radius',
            colormap = vis.COLORMAP_WAVEFUNCTION,
            norm = si.vis.AbsoluteRenormalize(),
            shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
            plot_limit = None,
            slicer = 'get_mesh_slicer',
            show_colorbar = True,
            show_title = True,
            show_axes = True,
            title_size = 12,
            axis_label_size = 12,
            tick_label_size = 10,
            grid_kwargs = None,
            title_y_adjust = 1.1,
            # overlay_probability_current = False, probability_current_time_step = 0,
            **kwargs):
        if grid_kwargs is None:
            grid_kwargs = {}

        with si.vis.FigureManager(name = f'{self.spec.name}__{name}', **kwargs) as figman:
            fig = figman.fig
            fig.set_tight_layout(True)
            axis = plt.subplot(111)

            unit_value, unit_name = u.get_unit_value_and_latex_from_unit(distance_unit)

            color_mesh = self.attach_mesh_to_axis(
                axis, mesh,
                distance_unit = distance_unit,
                colormap = colormap,
                norm = norm,
                shading = shading,
                plot_limit = plot_limit,
                slicer = slicer
            )
            # if overlay_probability_current:
            #     quiv = self.attach_probability_current_to_axis(axis, plot_limit = plot_limit, distance_unit = distance_unit)

            axis.set_xlabel(rf'$z$ (${unit_name}$)', fontsize = axis_label_size)
            axis.set_ylabel(rf'$\rho$ (${unit_name}$)', fontsize = axis_label_size)
            if title is not None and title != '' and show_axes and show_title:
                title = axis.set_title(title, fontsize = title_size)
                title.set_y(title_y_adjust)  # move title up a bit

            # make a colorbar
            if show_colorbar and show_axes:
                cbar = fig.colorbar(mappable = color_mesh, ax = axis, pad = .1)
                cbar.ax.tick_params(labelsize = tick_label_size)

            axis.axis('tight')  # removes blank space between color mesh and axes

            axis.grid(True, color = si.vis.CMAP_TO_OPPOSITE[colormap], **{**si.vis.COLORMESH_GRID_KWARGS, **grid_kwargs})  # change grid color to make it show up against the colormesh

            axis.tick_params(labelright = True, labeltop = True)  # ticks on all sides
            axis.tick_params(axis = 'both', which = 'major', labelsize = tick_label_size)  # increase size of tick labels
            # axis.tick_params(axis = 'both', which = 'both', length = 0)

            # set upper and lower y ticks to not display to avoid collisions with the x ticks at the edges
            # y_ticks = axis.yaxis.get_major_ticks()
            # y_ticks[0].label1.set_visible(False)
            # y_ticks[0].label2.set_visible(False)
            # y_ticks[-1].label1.set_visible(False)
            # y_ticks[-1].label2.set_visible(False)

            if not show_axes:
                axis.axis('off')


class WarpedCylindricalSliceMesh(QuantumMesh):
    mesh_storage_method = ('z', 'rho')

    def __init__(self, simulation: 'sims.MeshSimulation'):
        super().__init__(simulation)

        self.z = np.linspace(-self.spec.z_bound, self.spec.z_bound, self.spec.z_points)
        self.chi_max = self.spec.rho_bound ** (1 / self.spec.warping)
        self.chi = np.linspace(0, self.chi_max, self.spec.rho_points + 1)[1:]

        self.delta_z = self.z[1] - self.z[0]
        self.delta_chi = self.chi[1] - self.chi[0]
        self.inner_product_multiplier = self.delta_z * self.delta_chi

        self.z_center_index = int(self.spec.z_points // 2)

        self.g = self.get_g_for_state(self.spec.initial_state)

        self.mesh_points = len(self.z) * len(self.chi)
        self.matrix_operator_shape = (self.mesh_points, self.mesh_points)
        self.mesh_shape = np.shape(self.r_mesh)

    @property
    @si.utils.memoize
    def z_mesh(self):
        return np.meshgrid(self.z, self.chi, indexing = 'ij')[0]

    @property
    @si.utils.memoize
    def chi_mesh(self):
        return np.meshgrid(self.z, self.chi, indexing = 'ij')[1]

    @property
    @si.utils.memoize
    def rho_mesh(self):
        return self.chi_mesh ** self.spec.warping

    @property
    def g_factor(self):
        return np.sqrt(u.twopi * self.spec.warping) * (self.chi_mesh ** (self.spec.warping - 0.5))

    @property
    def r_mesh(self):
        return np.sqrt((self.z_mesh ** 2) + (self.rho_mesh ** 2))

    @property
    def theta_mesh(self):
        return np.arccos(self.z_mesh / self.r_mesh)

    @property
    def sin_theta_mesh(self):
        return np.sin(self.theta_mesh)

    @property
    def cos_theta_mesh(self):
        return np.cos(self.theta_mesh)

    def _wrapping_direction_to_order(self, wrapping_direction: Optional[WrappingDirection]) -> Optional[str]:
        if wrapping_direction is None:
            return None
        elif wrapping_direction == WrappingDirection.Z:
            return 'F'
        elif wrapping_direction == WrappingDirection.CHI:
            return 'C'
        else:
            raise ValueError(f"{wrapping_direction} is not a valid specifier for flatten_mesh (valid specifiers: 'z', 'chi')")

    @si.utils.memoize
    def get_g_for_state(self, state):
        g = self.g_factor * state(self.r_mesh, self.theta_mesh, 0)
        g /= np.sqrt(self.norm(g))
        g *= state.amplitude

        return g
    #
    # def z_dipole_moment_inner_product(self, a = None, b = None):
    #     return self.spec.test_charge * self.inner_product(a = a, b = self.z_mesh * self.state_to_mesh(b))
    #
    # def _get_kinetic_energy_matrix_operators_HAM(self):
    #     """Get the mesh kinetic energy operator matrices for z and rho."""
    #     z_prefactor = -(u.hbar ** 2) / (2 * self.spec.test_mass * (self.delta_z ** 2))
    #     chi_prefactor = -(u.hbar ** 2) / (2 * self.spec.test_mass * (self.delta_chi ** 2))
    #
    #     z_diagonal = z_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
    #     z_offdiagonal = z_prefactor * np.array([1 if (z_index + 1) % self.spec.z_points != 0 else 0 for z_index in range(self.mesh_points - 1)], dtype = np.complex128)
    #
    #     @si.utils.memoize
    #     def c(j):
    #         return j / np.sqrt((j ** 2) - 0.25)
    #
    #     chi_diagonal = chi_prefactor * ((-2 * np.ones(self.mesh_points, dtype = np.complex128)) + ((self.spec.warping - .5) ** 2))
    #     chi_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
    #     for rho_index in range(self.mesh_points - 1):
    #         if (rho_index + 1) % self.spec.rho_points != 0:
    #             j = (rho_index % self.spec.rho_points) + 1  # get j for the upper diagonal
    #             chi_offdiagonal[rho_index] = c(j)
    #     chi_offdiagonal *= chi_prefactor
    #
    #     z_kinetic = sparse.diags([z_offdiagonal, z_diagonal, z_offdiagonal], offsets = (-1, 0, 1))
    #     rho_kinetic = sparse.diags([chi_offdiagonal, chi_diagonal, chi_offdiagonal], offsets = (-1, 0, 1))
    #
    #     return z_kinetic, rho_kinetic
    #
    # @si.utils.memoize
    # def get_internal_hamiltonian_matrix_operators(self):
    #     """Get the mesh internal Hamiltonian matrix operators for z and rho."""
    #     kinetic_z, kinetic_rho = self.get_kinetic_energy_matrix_operators()
    #     potential_mesh = self.spec.internal_potential(r = self.r_mesh, test_charge = self.spec.test_charge)
    #
    #     kinetic_z = add_to_diagonal_sparse_matrix_diagonal(kinetic_z, value = 0.5 * self.flatten_mesh(potential_mesh, 'z'))
    #     kinetic_rho = add_to_diagonal_sparse_matrix_diagonal(kinetic_rho, value = 0.5 * self.flatten_mesh(potential_mesh, 'rho'))
    #
    #     return kinetic_z, kinetic_rho
    #
    # def _get_interaction_hamiltonian_matrix_operators_LEN(self):
    #     """Get the interaction term calculated from the Lagrangian evolution equations."""
    #     electric_potential_energy_mesh = self.spec.electric_potential(t = self.sim.time, distance_along_polarization = self.z_mesh, test_charge = self.spec.test_charge)
    #
    #     interaction_hamiltonian_z = sparse.diags(self.flatten_mesh(electric_potential_energy_mesh, 'z'))
    #     interaction_hamiltonian_rho = sparse.diags(self.flatten_mesh(electric_potential_energy_mesh, 'rho'))
    #
    #     return interaction_hamiltonian_z, interaction_hamiltonian_rho
    #
    # def _get_interaction_hamiltonian_matrix_operators_VEL(self):
    #     # vector_potential_amplitude = -self.spec.electric_potential.get_electric_field_integral_numeric_cumulative(self.sim.times_to_current)
    #     raise NotImplementedError
    #
    # def tg_mesh(self, use_abs_g = False):
    #     hamiltonian_z, hamiltonian_rho = self.get_kinetic_energy_matrix_operators()
    #
    #     if use_abs_g:
    #         g = np.abs(self.g)
    #     else:
    #         g = self.g
    #
    #     g_vector_z = self.flatten_mesh(g, 'z')
    #     hg_vector_z = hamiltonian_z.dot(g_vector_z)
    #     hg_mesh_z = self.wrap_vector(hg_vector_z, 'z')
    #
    #     g_vector_rho = self.flatten_mesh(g, 'rho')
    #     hg_vector_rho = hamiltonian_rho.dot(g_vector_rho)
    #     hg_mesh_rho = self.wrap_vector(hg_vector_rho, 'rho')
    #
    #     return hg_mesh_z + hg_mesh_rho
    #
    # def hg_mesh(self, include_interaction = False):
    #     hamiltonian_z, hamiltonian_rho = self.get_internal_hamiltonian_matrix_operators()
    #
    #     g_vector_z = self.flatten_mesh(self.g, 'z')
    #     hg_vector_z = hamiltonian_z.dot(g_vector_z)
    #     hg_mesh_z = self.wrap_vector(hg_vector_z, 'z')
    #
    #     g_vector_rho = self.flatten_mesh(self.g, 'rho')
    #     hg_vector_rho = hamiltonian_rho.dot(g_vector_rho)
    #     hg_mesh_rho = self.wrap_vector(hg_vector_rho, 'rho')
    #
    #     if include_interaction:
    #         raise NotImplementedError
    #
    #     return hg_mesh_z + hg_mesh_rho
    #
    # def energy_expectation_value(self, include_interaction = False):
    #     return np.real(self.inner_product(b = self.hg_mesh())) / self.norm()
    #
    # @si.utils.memoize
    # def _get_probability_current_matrix_operators(self):
    #     """Get the mesh probability current operators for z and rho."""
    #     z_prefactor = u.hbar / (4 * u.pi * self.spec.test_mass * self.delta_rho * self.delta_z)
    #     rho_prefactor = u.hbar / (4 * u.pi * self.spec.test_mass * (self.delta_rho ** 2))
    #
    #     # construct the diagonals of the z probability current matrix operator
    #     z_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
    #     for z_index in range(0, self.mesh_points - 1):
    #         if (z_index + 1) % self.spec.z_points == 0:  # detect edge of mesh
    #             z_offdiagonal[z_index] = 0
    #         else:
    #             j = z_index // self.spec.z_points
    #             z_offdiagonal[z_index] = 1 / (j + 0.5)
    #     z_offdiagonal *= z_prefactor
    #
    #     @si.utils.memoize
    #     def d(j):
    #         return 1 / np.sqrt((j ** 2) - 0.25)
    #
    #     # construct the diagonals of the rho probability current matrix operator
    #     rho_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
    #     for rho_index in range(0, self.mesh_points - 1):
    #         if (rho_index + 1) % self.spec.rho_points == 0:  # detect edge of mesh
    #             rho_offdiagonal[rho_index] = 0
    #         else:
    #             j = (rho_index % self.spec.rho_points) + 1
    #             rho_offdiagonal[rho_index] = d(j)
    #     rho_offdiagonal *= rho_prefactor
    #
    #     z_current = sparse.diags([-z_offdiagonal, z_offdiagonal], offsets = [-1, 1])
    #     rho_current = sparse.diags([-rho_offdiagonal, rho_offdiagonal], offsets = [-1, 1])
    #
    #     return z_current, rho_current
    #
    # def get_probability_current_vector_field(self):
    #     z_current, rho_current = self._get_probability_current_matrix_operators()
    #
    #     g_vector_z = self.flatten_mesh(self.g, 'z')
    #     current_vector_z = z_current.dot(g_vector_z)
    #     gradient_mesh_z = self.wrap_vector(current_vector_z, 'z')
    #     current_mesh_z = np.imag(np.conj(self.g) * gradient_mesh_z)
    #
    #     g_vector_rho = self.flatten_mesh(self.g, 'rho')
    #     current_vector_rho = rho_current.dot(g_vector_rho)
    #     gradient_mesh_rho = self.wrap_vector(current_vector_rho, 'rho')
    #     current_mesh_rho = np.imag(np.conj(self.g) * gradient_mesh_rho)
    #
    #     return current_mesh_z, current_mesh_rho
    #
    # def get_spline_for_mesh(self, mesh):
    #     return sp.interp.RectBivariateSpline(self.z, self.chi, mesh)
    #
    # @si.utils.memoize
    # def get_mesh_slicer(self, plot_limit = None):
    #     """Returns a slice object that slices a mesh to the given distance of the center."""
    #     if plot_limit is None:
    #         mesh_slicer = (slice(None, None, 1), slice(None, None, 1))
    #     else:
    #         z_lim_points = round(plot_limit / self.delta_z)
    #         rho_lim_points = round(plot_limit / self.delta_rho)
    #         mesh_slicer = (slice(int(self.z_center_index - z_lim_points), int(self.z_center_index + z_lim_points + 1), 1), slice(0, int(rho_lim_points + 1), 1))
    #
    #     return mesh_slicer


class SphericalSliceMesh(QuantumMesh):
    mesh_storage_method = ('r', 'theta')

    def __init__(self, simulation: 'sims.MeshSimulation'):
        super().__init__(simulation)

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
    def r_mesh(self):
        return np.meshgrid(self.r, self.theta, indexing = 'ij')[0]

    @property
    @si.utils.memoize
    def theta_mesh(self):
        return np.meshgrid(self.r, self.theta, indexing = 'ij')[1]

    @property
    def g_factor(self):
        return np.sqrt(u.twopi * np.sin(self.theta_mesh)) * self.r_mesh

    @property
    def z_mesh(self):
        return self.r_mesh * np.cos(self.theta_mesh)

    def _wrapping_direction_to_order(self, wrapping_direction: Optional[WrappingDirection]) -> Optional[str]:
        if wrapping_direction is None:
            return None
        elif wrapping_direction == WrappingDirection.R:
            return 'F'
        elif wrapping_direction == WrappingDirection.THETA:
            return 'C'
        else:
            raise ValueError(f"{wrapping_direction} is not a valid specifier for flatten_mesh (valid specifiers: 'r', 'theta')")

    def state_overlap(self, state_a = None, state_b = None):
        """State overlap between two states. If either state is None, the state on the g is used for that state."""
        if state_a is None:
            mesh_a = self.g
        else:
            mesh_a = self.get_g_for_state(state_a)
        if state_b is None:
            b = self.g
        else:
            b = self.get_g_for_state(state_b)

        return np.abs(self.inner_product(mesh_a, b)) ** 2

    @si.utils.memoize
    def get_g_for_state(self, state):
        g = self.g_factor * state(self.r_mesh, self.theta_mesh, 0)
        g /= np.sqrt(self.norm(g))
        g *= state.amplitude

        return g

    def z_dipole_moment_inner_product(self, a = None, b = None):
        return self.spec.test_charge * self.inner_product(a = a, b = self.z_mesh * self.state_to_mesh(b))

    def _get_kinetic_energy_matrix_operators_HAM(self):
        r_prefactor = -(u.hbar ** 2) / (2 * u.electron_mass_reduced * (self.delta_r ** 2))
        theta_prefactor = -(u.hbar ** 2) / (2 * u.electron_mass_reduced * ((self.delta_r * self.delta_theta) ** 2))

        r_diagonal = r_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        r_offdiagonal = r_prefactor * np.array([1 if (z_index + 1) % self.spec.r_points != 0 else 0 for z_index in range(self.mesh_points - 1)], dtype = np.complex128)

        @si.utils.memoize
        def theta_j_prefactor(x):
            return 1 / (x + 0.5) ** 2

        @si.utils.memoize
        def sink(x):
            return np.sin(x * self.delta_theta)

        @si.utils.memoize
        def sqrt_sink_ratio(x_num, x_den):
            return np.sqrt(sink(x_num) / sink(x_den))

        @si.utils.memoize
        def cotank(x):
            return 1 / np.tan(x * self.delta_theta)

        theta_diagonal = (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        for theta_index in range(self.mesh_points):
            j = theta_index // self.spec.theta_points
            theta_diagonal[theta_index] *= theta_j_prefactor(j)
        theta_diagonal *= theta_prefactor

        theta_upper_diagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        theta_lower_diagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for theta_index in range(self.mesh_points - 1):
            if (theta_index + 1) % self.spec.theta_points != 0:
                j = theta_index // self.spec.theta_points
                k = theta_index % self.spec.theta_points
                k_p = k + 1  # add 1 because the entry for the lower diagonal is really for the next point (k -> k + 1), not this one
                theta_upper_diagonal[theta_index] = theta_j_prefactor(j) * (1 + (self.delta_theta / 2) * cotank(k + 0.5)) * sqrt_sink_ratio(k + 0.5, k + 1.5)
                theta_lower_diagonal[theta_index] = theta_j_prefactor(j) * (1 - (self.delta_theta / 2) * cotank(k_p + 0.5)) * sqrt_sink_ratio(k_p + 0.5, k_p - 0.5)
        theta_upper_diagonal *= theta_prefactor
        theta_lower_diagonal *= theta_prefactor

        r_kinetic = sparse.diags([r_offdiagonal, r_diagonal, r_offdiagonal], offsets = (-1, 0, 1))
        theta_kinetic = sparse.diags([theta_lower_diagonal, theta_diagonal, theta_upper_diagonal], offsets = (-1, 0, 1))

        return r_kinetic, theta_kinetic

    @si.utils.memoize
    def get_internal_hamiltonian_matrix_operators(self):
        kinetic_r, kinetic_theta = self.get_kinetic_energy_matrix_operators()
        potential_mesh = self.spec.internal_potential(r = self.r_mesh, test_charge = self.spec.test_charge)

        kinetic_r = add_to_diagonal_sparse_matrix_diagonal(kinetic_r, value = 0.5 * self.flatten_mesh(potential_mesh, 'r'))
        kinetic_theta = add_to_diagonal_sparse_matrix_diagonal(kinetic_theta, value = 0.5 * self.flatten_mesh(potential_mesh, 'theta'))

        return kinetic_r, kinetic_theta

    def _get_interaction_hamiltonian_matrix_operators_LEN(self):
        """Get the angular momentum interaction term calculated from the Lagrangian evolution equations."""
        electric_potential_energy_mesh = self.spec.electric_potential(t = self.sim.time, distance_along_polarization = self.z_mesh, test_charge = self.spec.test_charge)

        interaction_hamiltonian_r = sparse.diags(self.flatten_mesh(electric_potential_energy_mesh, 'r'))
        interaction_hamiltonian_theta = sparse.diags(self.flatten_mesh(electric_potential_energy_mesh, 'theta'))

        return interaction_hamiltonian_r, interaction_hamiltonian_theta

    def _get_interaction_hamiltonian_matrix_operators_VEL(self):
        # vector_potential_amplitude = -self.spec.electric_potential.get_electric_field_integral_numeric_cumulative(self.sim.times_to_current)

        raise NotImplementedError

    def tg_mesh(self, use_abs_g: bool = False):
        hamiltonian_r, hamiltonian_theta = self.get_kinetic_energy_matrix_operators()

        if use_abs_g:
            g = np.abs(self.g)
        else:
            g = self.g

        g_vector_r = self.flatten_mesh(g, 'r')
        hg_vector_r = hamiltonian_r.dot(g_vector_r)
        hg_mesh_r = self.wrap_vector(hg_vector_r, 'r')

        g_vector_theta = self.flatten_mesh(g, 'theta')
        hg_vector_theta = hamiltonian_theta.dot(g_vector_theta)
        hg_mesh_theta = self.wrap_vector(hg_vector_theta, 'theta')

        return hg_mesh_r + hg_mesh_theta

    def hg_mesh(self, include_interaction: bool = False):
        hamiltonian_r, hamiltonian_theta = self.get_internal_hamiltonian_matrix_operators()

        g_vector_r = self.flatten_mesh(self.g, 'r')
        hg_vector_r = hamiltonian_r.dot(g_vector_r)
        hg_mesh_r = self.wrap_vector(hg_vector_r, 'r')

        g_vector_theta = self.flatten_mesh(self.g, 'theta')
        hg_vector_theta = hamiltonian_theta.dot(g_vector_theta)
        hg_mesh_theta = self.wrap_vector(hg_vector_theta, 'theta')

        if include_interaction:
            raise NotImplementedError
        # TODO: not including interaction yet

        return hg_mesh_r + hg_mesh_theta

    def energy_expectation_value(self, include_interaction: bool = False):
        return np.real(self.inner_product(b = self.hg_mesh(include_interaction = include_interaction))) / self.norm()

    @si.utils.memoize
    def get_probability_current_matrix_operators(self):
        raise NotImplementedError

    def get_probability_current_vector_field(self):
        raise NotImplementedError

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

    def attach_mesh_to_axis(
            self,
            axis,
            mesh,
            distance_unit: u.Unit = 'bohr_radius',
            colormap = plt.get_cmap('inferno'),
            norm = si.vis.AbsoluteRenormalize(),
            shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
            plot_limit = None,
            slicer = 'get_mesh_slicer',
            **kwargs):
        unit_value, _ = u.get_unit_value_and_latex_from_unit(distance_unit)

        _slice = getattr(self, slicer)(plot_limit)

        color_mesh = axis.pcolormesh(
            self.theta_mesh[_slice],
            self.r_mesh[_slice] / unit_value,
            mesh[_slice],
            shading = shading,
            cmap = colormap,
            norm = norm,
            **kwargs
        )
        color_mesh_mirror = axis.pcolormesh(
            u.twopi - self.theta_mesh[_slice],
            self.r_mesh[_slice] / unit_value,
            mesh[_slice],
            shading = shading,
            cmap = colormap,
            norm = norm,
            **kwargs
        )  # another colormesh, mirroring the first mesh onto pi to 2pi

        return color_mesh, color_mesh_mirror

    def attach_probability_current_to_axis(self, axis, plot_limit = None, distance_unit: u.Unit = 'bohr_radius'):
        raise NotImplementedError

    def plot_mesh(
            self,
            mesh,
            name: str = '',
            title: Optional[str] = None,
            overlay_probability_current = False,
            probability_current_time_step = 0,
            plot_limit = None,
            distance_unit = 'nm',
            color_map = plt.get_cmap('inferno'),
            **kwargs):
        plt.close()  # close any old figures

        plt.set_cmap(color_map)

        unit_value, unit_name = u.get_unit_value_and_latex_from_unit(distance_unit)

        fig = si.vis.get_figure('full')
        fig.set_tight_layout(True)
        axis = plt.subplot(111, projection = 'polar')
        axis.set_theta_zero_location('N')
        axis.set_theta_direction('clockwise')

        color_mesh, color_mesh_mirror = self.attach_mesh_to_axis(axis, mesh, plot_limit = plot_limit, distance_unit = distance_unit)
        # if overlay_probability_current:
        #     quiv = self.attach_probability_current_to_axis(axis, plot_limit = plot_limit, distance_unit = distance_unit)

        if title is not None:
            title = axis.set_title(title, fontsize = 15)
            title.set_x(.03)  # move title to the upper left corner
            title.set_y(.97)

        # make a colorbar
        cbar_axis = fig.add_axes([1.01, .1, .04, .8])  # add a new axis for the cbar so that the old axis can stay square
        cbar = plt.colorbar(mappable = color_mesh, cax = cbar_axis)
        cbar.ax.tick_params(labelsize = 10)

        axis.grid(True, color = si.vis.CMAP_TO_OPPOSITE[color_map], **si.vis.COLORMESH_GRID_KWARGS)  # change grid color to make it show up against the colormesh
        angle_labels = [f'{s}\u00b0' for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
        axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)

        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)  # increase size of tick labels
        axis.tick_params(axis = 'y', which = 'major', colors = si.vis.COLOR_OPPOSITE_INFERNO, pad = 3)  # make r ticks a color that shows up against the colormesh
        axis.tick_params(axis = 'both', which = 'both', length = 0)

        axis.set_rlabel_position(80)

        max_yticks = 5
        yloc = plt.MaxNLocator(max_yticks, symmetric = False, prune = 'both')
        axis.yaxis.set_major_locator(yloc)

        fig.canvas.draw()  # must draw early to modify the axis text

        tick_labels = axis.get_yticklabels()
        for t in tick_labels:
            t.set_text(t.get_text() + rf'${unit_name}$')
        axis.set_yticklabels(tick_labels)

        axis.set_rmax((self.r_max - (self.delta_r / 2)) / unit_value)

        si.vis.save_current_figure(name = f'{self.spec.name}_{name}', **kwargs)

        plt.close()


class SphericalHarmonicMesh(QuantumMesh):
    mesh_storage_method = ('l', 'r')

    def __init__(self, simulation: 'sims.MeshSimulation'):
        super().__init__(simulation)

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
    def g_factor(self):
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

    def z_dipole_moment_inner_product(self, a: StateOrGMesh = None, b: StateOrGMesh = None):
        operator = self._get_interaction_hamiltonian_matrix_operators_without_field_LEN()
        b = self.wrap_vector(operator.dot(self.flatten_mesh(self.state_to_mesh(b), 'l')), 'l')
        return -self.inner_product(a = a, b = b)

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
        # # def phase(l, k):
        # #     return np.exp(1j * states.coulomb_phase_shift(l, k))
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
        # def phase(k):
        #     return np.exp(1j * states.coulomb_phase_shift(l_mesh, k))
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

    def _get_kinetic_energy_matrix_operators_HAM(self) -> SparseMatrixOperator:
        r_prefactor = -(u.hbar ** 2) / (2 * u.electron_mass_reduced * (self.delta_r ** 2))

        r_diagonal = r_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        r_offdiagonal = r_prefactor * np.ones(self.mesh_points - 1, dtype = np.complex128)

        effective_potential_mesh = ((u.hbar ** 2) / (2 * u.electron_mass_reduced)) * self.l_mesh * (self.l_mesh + 1) / (self.r_mesh ** 2)
        r_diagonal += self.flatten_mesh(effective_potential_mesh, 'r')

        r_kinetic = sparse.diags([r_offdiagonal, r_diagonal, r_offdiagonal], offsets = (-1, 0, 1))

        return r_kinetic

    def alpha(self, j) -> float:
        x = (j ** 2) + (2 * j)
        return (x + 1) / (x + 0.75)

    def beta(self, j) -> float:
        x = 2 * (j ** 2) + (2 * j)
        return (x + 1) / (x + 0.5)

    def gamma(self, j) -> float:
        """For radial probability current."""
        return 1 / ((j ** 2) - 0.25)

    def _get_kinetic_energy_matrix_operator_single_l(self, l: int) -> SparseMatrixOperator:
        r_prefactor = -(u.hbar ** 2) / (2 * u.electron_mass_reduced * (self.delta_r ** 2))
        effective_potential = ((u.hbar ** 2) / (2 * u.electron_mass_reduced)) * l * (l + 1) / (self.r ** 2)

        r_beta = self.beta(np.array(range(len(self.r)), dtype = np.complex128))
        if l == 0 and self.spec.hydrogen_zero_angular_momentum_correction:
            dr = self.delta_r / u.bohr_radius
            r_beta[0] += dr * (1 + dr) / 8
        r_diagonal = (-2 * r_prefactor * r_beta) + effective_potential
        r_offdiagonal = r_prefactor * self.alpha(np.array(range(len(self.r) - 1), dtype = np.complex128))

        return sparse.diags([r_offdiagonal, r_diagonal, r_offdiagonal], offsets = (-1, 0, 1))

    def _get_internal_hamiltonian_matrix_operator_single_l(self, l: int) -> SparseMatrixOperator:
        r_kinetic = self._get_kinetic_energy_matrix_operator_single_l(l)
        potential = self.spec.internal_potential(r = self.r, test_charge = self.spec.test_charge)

        r_kinetic.data[1] += potential

        return r_kinetic

    def _get_kinetic_energy_matrix_operators_LAG(self, include_effective_potential: bool = True) -> SparseMatrixOperator:
        """Get the radial kinetic energy matrix operator."""
        r_prefactor = -(u.hbar ** 2) / (2 * u.electron_mass_reduced * (self.delta_r ** 2))

        r_diagonal = np.zeros(self.mesh_points, dtype = np.complex128)
        r_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for r_index in range(self.mesh_points):
            j = r_index % self.spec.r_points
            r_diagonal[r_index] = self.beta(j)
        if self.spec.hydrogen_zero_angular_momentum_correction:
            dr = self.delta_r / u.bohr_radius
            r_diagonal[0] += dr * (1 + dr) / 8  # modify beta_j for l = 0   (see notes)

        for r_index in range(self.mesh_points - 1):
            if (r_index + 1) % self.spec.r_points != 0:
                j = (r_index % self.spec.r_points)
                r_offdiagonal[r_index] = self.alpha(j)
        r_diagonal *= -2 * r_prefactor
        r_offdiagonal *= r_prefactor

        if include_effective_potential:
            effective_potential_mesh = ((u.hbar ** 2) / (2 * u.electron_mass_reduced)) * self.l_mesh * (self.l_mesh + 1) / (self.r_mesh ** 2)
            r_diagonal += self.flatten_mesh(effective_potential_mesh, 'r')

        r_kinetic = sparse.diags([r_offdiagonal, r_diagonal, r_offdiagonal], offsets = (-1, 0, 1))

        return r_kinetic

    @si.utils.memoize
    def get_internal_hamiltonian_matrix_operators(self) -> SparseMatrixOperator:
        r_kinetic = self.get_kinetic_energy_matrix_operators().copy()

        potential_mesh = self.spec.internal_potential(r = self.r_mesh, test_charge = self.spec.test_charge)

        r_kinetic.data[1] += self.flatten_mesh(potential_mesh, 'r')

        return r_kinetic

    @si.utils.memoize
    def _get_interaction_hamiltonian_matrix_operators_without_field_LEN(self) -> SparseMatrixOperator:
        l_prefactor = -self.spec.test_charge * self.flatten_mesh(self.r_mesh, 'l')[:-1]

        l_diagonal = np.zeros(self.mesh_points, dtype = np.complex128)
        l_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for l_index in range(self.mesh_points - 1):
            if (l_index + 1) % self.spec.l_bound != 0:
                l = (l_index % self.spec.l_bound)
                l_offdiagonal[l_index] = c_l(l)
        l_offdiagonal *= l_prefactor

        return sparse.diags([l_offdiagonal, l_diagonal, l_offdiagonal], offsets = (-1, 0, 1))

    def _get_interaction_hamiltonian_matrix_operators_LEN(self) -> SparseMatrixOperator:
        """Get the angular momentum interaction term calculated from the Lagrangian evolution equations in the length gauge."""
        return self._get_interaction_hamiltonian_matrix_operators_without_field_LEN() * self.spec.electric_potential.get_electric_field_amplitude(self.sim.time + (self.spec.time_step / 2))

    @si.utils.memoize
    def _get_interaction_hamiltonian_matrix_operators_without_field_VEL(self) -> Tuple[SparseMatrixOperator, SparseMatrixOperator]:
        h1_prefactor = 1j * u.hbar * (self.spec.test_charge / self.spec.test_mass) / self.flatten_mesh(self.r_mesh, 'l')[:-1]

        h1_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for l_index in range(self.mesh_points - 1):
            if (l_index + 1) % self.spec.l_bound != 0:
                l = (l_index % self.spec.l_bound)
                h1_offdiagonal[l_index] = c_l(l) * (l + 1)
        h1_offdiagonal *= h1_prefactor

        h1 = sparse.diags((-h1_offdiagonal, h1_offdiagonal), offsets = (-1, 1))

        h2_prefactor = 1j * u.hbar * (self.spec.test_charge / self.spec.test_mass) / (2 * self.delta_r)

        alpha_vec = self.alpha(np.array(range(len(self.r) - 1), dtype = np.complex128))
        alpha_block = sparse.diags((-alpha_vec, alpha_vec), offsets = (-1, 1))

        c_vec = c_l(np.array(range(len(self.l) - 1), dtype = np.complex128))
        c_block = sparse.diags((c_vec, c_vec), offsets = (-1, 1))

        h2 = h2_prefactor * sparse.kron(c_block, alpha_block, format = 'dia')

        return h1, h2

    def _get_interaction_hamiltonian_matrix_operators_VEL(self) -> Tuple[SparseMatrixOperator, SparseMatrixOperator]:
        vector_potential_amp = self.spec.electric_potential.get_vector_potential_amplitude_numeric(self.sim.times_to_current)
        return (x * vector_potential_amp for x in self._get_interaction_hamiltonian_matrix_operators_without_field_VEL())

    def get_numeric_eigenstate_basis(self, max_energy: float, max_angular_momentum: int) -> Dict[states.QuantumState, states.NumericSphericalHarmonicState]:
        analytic_to_numeric = {}

        for l in range(max_angular_momentum + 1):
            h = self._get_internal_hamiltonian_matrix_operator_single_l(l = l)

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
                    bound = False
                else:
                    n_guess = round(np.sqrt(u.rydberg / np.abs(eigenvalue)))
                    if n_guess == 0:
                        n_guess = 1
                    analytic_state = states.HydrogenBoundState(n = n_guess, l = l)
                    bound = True

                numeric_state = states.NumericSphericalHarmonicState(eigenvector, l, 0, eigenvalue, analytic_state, bound = bound)

                analytic_to_numeric[analytic_state] = numeric_state

            logger.debug(f'Generated numerical eigenbasis for l = {l}, energy <= {u.uround(max_energy, u.eV)} eV')

        logger.debug(f'Generated numerical eigenbasis for l <= {max_angular_momentum}, energy <= {u.uround(max_energy, u.eV)} eV. Found {len(analytic_to_numeric)} states.')

        return analytic_to_numeric

    def tg_mesh(self, use_abs_g: bool = False):
        if use_abs_g:
            g = np.abs(self.g)
        else:
            g = self.g

        hamiltonian_r = self.get_kinetic_energy_matrix_operators()

        return self.wrap_vector(hamiltonian_r.dot(self.flatten_mesh(g, 'r')), 'r')

    def hg_mesh(self, include_interaction: bool = False):
        hamiltonian_r = self.get_internal_hamiltonian_matrix_operators()

        hg = self.wrap_vector(hamiltonian_r.dot(self.flatten_mesh(self.g, 'r')), 'r')

        if include_interaction:
            hamiltonian_l = self.get_interaction_hamiltonian_matrix_operators()
            wrapping_direction = 'l' if self.spec.evolution_gauge == 'LEN' else 'r'

            hg += self.wrap_vector(hamiltonian_l.dot(self.flatten_mesh(self.g, wrapping_direction)), wrapping_direction)

        return hg

    def energy_expectation_value(self, include_interaction: bool = False):
        return np.real(self.inner_product(b = self.hg_mesh(include_interaction = include_interaction))) / self.norm()

    @si.utils.memoize
    def _get_probability_current_matrix_operators(self):
        raise NotImplementedError

    def get_probability_current_vector_field(self):
        raise NotImplementedError

    @si.utils.memoize
    def _get_radial_probability_current_operator__spatial(self) -> SparseMatrixOperator:
        r_prefactor = u.hbar / (2 * self.spec.test_mass * (self.delta_r ** 3))  # / extra 2 from taking Im later

        r_offdiagonal = np.zeros((self.spec.r_points * self.spec.theta_points) - 1, dtype = np.complex128)

        for r_index in range((self.spec.r_points * self.spec.theta_points) - 1):
            if (r_index + 1) % self.spec.r_points != 0:
                j = (r_index % self.spec.r_points) + 1
                r_offdiagonal[r_index] = self.gamma(j)

        r_offdiagonal *= r_prefactor

        r_current_operator = sparse.diags([-r_offdiagonal, r_offdiagonal], offsets = [-1, 1])

        return r_current_operator

    def get_radial_probability_current_density_mesh__spatial(self) -> SparseMatrixOperator:
        r_current_operator = self._get_radial_probability_current_operator__spatial()

        g_spatial = self.space_g_calc
        g_spatial_shape = g_spatial.shape

        g_vector_r = g_spatial.flatten('F')
        gradient_vector_r = r_current_operator.dot(g_vector_r)
        gradient_mesh_r = np.reshape(gradient_vector_r, g_spatial_shape, 'F')
        current_mesh_r = np.imag(np.conj(g_spatial) * gradient_mesh_r)

        return current_mesh_r

    def _make_split_operator_evolution_operators(self, interaction_hamiltonian_matrix_operators, tau: float):
        return getattr(self, f'_make_split_operator_evolution_operators_{self.spec.evolution_gauge}')(interaction_hamiltonian_matrix_operators, tau)

    def _make_split_operator_evolution_operators_LEN(self, interaction_hamiltonians_matrix_operators, tau: float):
        """Calculate split operator evolution matrices for the interaction term in the length gauge."""
        a = tau * interaction_hamiltonians_matrix_operators.data[0][:-1]

        a_even, a_odd = a[::2], a[1::2]

        len_a = len(a)

        even_diag = np.zeros(len_a + 1, dtype = np.complex128)
        even_offdiag = np.zeros(len_a, dtype = np.complex128)
        odd_diag = np.zeros(len_a + 1, dtype = np.complex128)
        odd_offdiag = np.zeros(len_a, dtype = np.complex128)

        if len(self.r) % 2 != 0 and len(self.l) % 2 != 0:
            even_diag[:-1] = np.cos(a_even).repeat(2)
            even_diag[-1] = 1

            even_offdiag[::2] = -1j * np.sin(a_even)

            odd_diag[0] = 1
            odd_diag[1:] = np.cos(a_odd).repeat(2)

            odd_offdiag[1::2] = -1j * np.sin(a_odd)
        else:
            even_diag[:] = np.cos(a_even).repeat(2)

            even_offdiag[::2] = -1j * np.sin(a_even)

            odd_diag[0] = odd_diag[-1] = 1
            odd_diag[1:-1] = np.cos(a_odd).repeat(2)

            odd_offdiag[1::2] = -1j * np.sin(a_odd)

        even = sparse.diags((even_offdiag, even_diag, even_offdiag), offsets = (-1, 0, 1))
        odd = sparse.diags((odd_offdiag, odd_diag, odd_offdiag), offsets = (-1, 0, 1))

        return (
            DotOperator(even, wrapping_direction = 'l'),
            DotOperator(odd, wrapping_direction = 'l'),
        )

    def _make_split_operators_VEL_h1(self, h1, tau: float):
        a = (tau * (-1j)) * h1.data[-1][1:]

        a_even, a_odd = a[::2], a[1::2]

        even_diag = np.zeros(len(a) + 1, dtype = np.complex128)
        even_offdiag = np.zeros(len(a), dtype = np.complex128)
        odd_diag = np.zeros(len(a) + 1, dtype = np.complex128)
        odd_offdiag = np.zeros(len(a), dtype = np.complex128)

        if len(self.r) % 2 != 0 and len(self.l) % 2 != 0:
            even_diag[:-1] = np.cos(a_even).repeat(2)
            even_diag[-1] = 1

            even_offdiag[::2] = np.sin(a_even)

            odd_diag[0] = 1
            odd_diag[1:] = np.cos(a_odd).repeat(2)

            odd_offdiag[1::2] = np.sin(a_odd)
        else:
            even_diag[:] = np.cos(a_even).repeat(2)

            even_offdiag[::2] = np.sin(a_even)

            odd_diag[0] = odd_diag[-1] = 1
            odd_diag[1:-1] = np.cos(a_odd).repeat(2)

            odd_offdiag[1::2] = np.sin(a_odd)

        even = sparse.diags([-even_offdiag, even_diag, even_offdiag], offsets = [-1, 0, 1])
        odd = sparse.diags([-odd_offdiag, odd_diag, odd_offdiag], offsets = [-1, 0, 1])

        return (
            DotOperator(even, wrapping_direction = 'l'),
            DotOperator(odd, wrapping_direction = 'l'),
        )

    def _make_split_operators_VEL_h2(self, h2, tau: float):
        len_r = len(self.r)

        a = h2.data[-1][len_r + 1:] * tau * (-1j)

        alpha_slices_even_l = []
        alpha_slices_odd_l = []
        for l in self.l:  # want last l but not last r, since unwrapped in r
            a_slice = a[l * len_r: ((l + 1) * len_r) - 1]
            if l % 2 == 0:
                alpha_slices_even_l.append(a_slice)
            else:
                alpha_slices_odd_l.append(a_slice)

        even_even_diag = []
        even_even_offdiag = []
        even_odd_diag = []
        even_odd_offdiag = []
        for alpha_slice in alpha_slices_even_l:  # FOR EACH l
            even_slice = alpha_slice[::2]
            odd_slice = alpha_slice[1::2]

            if len(even_slice) > 0:
                even_sines = np.zeros(len_r, dtype = np.complex128)
                if len_r % 2 == 0:
                    new_even_even_diag = np.tile(np.cos(even_slice).repeat(2), 2)
                    even_sines[::2] = np.sin(even_slice)
                else:
                    tile = np.ones(len_r, dtype = np.complex128)
                    tile[:-1] = np.cos(even_slice).repeat(2)
                    tile[-1] = 1
                    new_even_even_diag = np.tile(tile, 2)
                    even_sines[:-1:2] = np.sin(even_slice)
                    even_sines[-1] = 0

                even_even_diag.append(new_even_even_diag)
                even_even_offdiag.append(even_sines)
                even_even_offdiag.append(-even_sines)
            else:
                even_even_diag.append(np.ones(len_r))
                even_even_offdiag.append(np.zeros(len_r))

            if len(odd_slice) > 0:
                new_even_odd_diag = np.ones(len_r, dtype = np.complex128)

                if len_r % 2 == 0:
                    new_even_odd_diag[1:-1] = np.cos(odd_slice).repeat(2)
                else:
                    new_even_odd_diag[1::] = np.cos(odd_slice).repeat(2)

                new_even_odd_diag = np.tile(new_even_odd_diag, 2)

                even_odd_diag.append(new_even_odd_diag)

                odd_sines = np.zeros(len_r, dtype = np.complex128)
                odd_sines[1:-1:2] = np.sin(odd_slice)
                even_odd_offdiag.append(odd_sines)
                even_odd_offdiag.append(-odd_sines)
        if self.l[-1] % 2 == 0:
            even_odd_diag.append(np.ones(len_r))
            even_odd_offdiag.append(np.zeros(len_r))

        even_even_diag = np.hstack(even_even_diag)
        even_even_offdiag = np.hstack(even_even_offdiag)[:-1]  # last element is bogus

        even_odd_diag = np.hstack(even_odd_diag)
        even_odd_offdiag = np.hstack(even_odd_offdiag)[:-1]  # last element is bogus

        odd_even_diag = [np.ones(len_r)]
        odd_even_offdiag = [np.zeros(len_r)]
        odd_odd_diag = [np.ones(len_r)]
        odd_odd_offdiag = [np.zeros(len_r)]

        for alpha_slice in alpha_slices_odd_l:
            even_slice = alpha_slice[::2]
            odd_slice = alpha_slice[1::2]

            if len(even_slice) > 0:
                even_sines = np.zeros(len_r, dtype = np.complex128)
                if len_r % 2 == 0:
                    new_odd_even_diag = np.tile(np.cos(even_slice).repeat(2), 2)
                    even_sines[::2] = np.sin(even_slice)
                else:
                    tile = np.ones(len_r, dtype = np.complex128)
                    tile[:-1] = np.cos(even_slice).repeat(2)
                    # tile[-1] = 1
                    new_odd_even_diag = np.tile(tile, 2)
                    even_sines[:-1:2] = np.sin(even_slice)
                    # even_sines[-1] = 0

                odd_even_diag.append(new_odd_even_diag)
                odd_even_offdiag.append(even_sines)
                odd_even_offdiag.append(-even_sines)
            else:
                odd_even_diag.append(np.ones(len_r))
                odd_even_offdiag.append(np.zeros(len_r))

            if len(odd_slice) > 0:
                new_odd_odd_diag = np.ones(len_r, dtype = np.complex128)

                if len_r % 2 == 0:
                    new_odd_odd_diag[1:-1] = np.cos(odd_slice).repeat(2)
                else:
                    new_odd_odd_diag[1::] = np.cos(odd_slice).repeat(2)

                new_odd_odd_diag = np.tile(new_odd_odd_diag, 2)

                odd_odd_diag.append(new_odd_odd_diag)

                odd_sines = np.zeros(len_r, dtype = np.complex128)
                odd_sines[1:-1:2] = np.sin(odd_slice)
                odd_odd_offdiag.append(odd_sines)
                odd_odd_offdiag.append(-odd_sines)
        if self.l[-1] % 2 != 0:
            odd_odd_diag.append(np.ones(len_r))
            odd_odd_offdiag.append(np.zeros(len_r))

        odd_even_diag = np.hstack(odd_even_diag)
        odd_even_offdiag = np.hstack(odd_even_offdiag)[:-1]  # last element is bogus

        odd_odd_diag = np.hstack(odd_odd_diag)
        odd_odd_offdiag = np.hstack(odd_odd_offdiag)[:-1]  # last element is bogus

        even_even_matrix = sparse.diags((-even_even_offdiag, even_even_diag, even_even_offdiag), offsets = (-1, 0, 1))
        even_odd_matrix = sparse.diags((-even_odd_offdiag, even_odd_diag, even_odd_offdiag), offsets = (-1, 0, 1))
        odd_even_matrix = sparse.diags((-odd_even_offdiag, odd_even_diag, odd_even_offdiag), offsets = (-1, 0, 1))
        odd_odd_matrix = sparse.diags((-odd_odd_offdiag, odd_odd_diag, odd_odd_offdiag), offsets = (-1, 0, 1))

        operators = (
            SimilarityOperator(even_even_matrix, wrapping_direction = 'r', parity = 'even'),
            SimilarityOperator(even_odd_matrix, wrapping_direction = 'r', parity = 'even'),  # parity is based off FIRST splitting
            SimilarityOperator(odd_even_matrix, wrapping_direction = 'r', parity = 'odd'),
            SimilarityOperator(odd_odd_matrix, wrapping_direction = 'r', parity = 'odd'),
        )

        return operators

    def _make_split_operator_evolution_operators_VEL(self, interaction_hamiltonians_matrix_operators, tau: float):
        """Calculate split operator evolution matrices for the interaction term in the velocity gauge."""
        h1, h2 = interaction_hamiltonians_matrix_operators

        h1_operators = self._make_split_operators_VEL_h1(h1, tau)
        h2_operators = self._make_split_operators_VEL_h2(h2, tau)

        return [*h1_operators, *h2_operators]


    def _apply_length_gauge_transformation(self, vector_potential_amplitude: float, g: GMesh):
        bessel_mesh = special.spherical_jn(self.l_mesh, self.spec.test_charge * vector_potential_amplitude * self.r_mesh / u.hbar)

        g_transformed = np.zeros(np.shape(g), dtype = np.complex128)
        for l_result in self.l:
            for l_outer in self.l:  # l'
                prefactor = np.sqrt(4 * u.pi * ((2 * l_outer) + 1)) * ((1j) ** (l_outer % 4)) * bessel_mesh[l_outer, :]
                for l_inner in self.l:  # l
                    print(l_result, l_outer, l_inner)
                    g_transformed[l_result, :] += g[l_inner, :] * prefactor * core.triple_y_integral(l_outer, 0, l_result, 0, l_inner, 0)

        return g_transformed

    def gauge_transformation(self, *, g: Optional[GMesh] = None, leaving_gauge: Optional[str] = None):
        if g is None:
            g = self.g
        if leaving_gauge is None:
            leaving_gauge = self.spec.evolution_gauge

        vamp = self.spec.electric_potential.get_vector_potential_amplitude_numeric_cumulative(self.sim.times_to_current)
        integral = integ.simps(y = vamp ** 2,
                               x = self.sim.times_to_current)

        dipole_to_velocity = np.exp(1j * integral * (self.spec.test_charge ** 2) / (2 * self.spec.test_mass * u.hbar))

        if leaving_gauge == 'LEN':
            return self._apply_length_gauge_transformation(-vamp[-1], dipole_to_velocity * g)
        elif leaving_gauge == 'VEL':
            return dipole_to_velocity * self._apply_length_gauge_transformation(vamp[-1], g)

    @si.utils.memoize
    def get_mesh_slicer(self, distance_from_center: Optional[float] = None):
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if distance_from_center is None:
            mesh_slicer = (slice(None, None, 1), slice(None, None, 1))
        else:
            r_lim_points = int(distance_from_center / self.delta_r)
            mesh_slicer = (slice(None, None, 1), slice(0, int(r_lim_points + 1), 1))

        return mesh_slicer

    @si.utils.memoize
    def get_mesh_slicer_spatial(self, distance_from_center: Optional[float] = None):
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

    def reconstruct_spatial_mesh__plot(self, mesh: WavefunctionMesh):
        """Reconstruct the spatial (r, theta) representation of a mesh from the (l, r) representation."""
        # l: l, angular momentum index
        # r: r, radial position index
        # t: theta, polar angle index
        return np.einsum('lr,lt->rt', mesh, self._sph_harm_l_theta_plot_mesh)

    def reconstruct_spatial_mesh__calc(self, mesh: WavefunctionMesh):
        """Reconstruct the spatial (r, theta) representation of a mesh from the (l, r) representation."""
        # l: l, angular momentum index
        # r: r, radial position index
        # t: theta, polar angle index
        return np.einsum('lr,lt->rt', mesh, self._sph_harm_l_theta_calc_mesh)

    @property
    @si.utils.watcher(lambda s: s.sim.time)
    def space_g(self) -> GMesh:
        return self.reconstruct_spatial_mesh__plot(self.g)

    @property
    @si.utils.watcher(lambda s: s.sim.time)
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

    def attach_mesh_to_axis(
            self,
            axis: plt.Axes,
            mesh: WavefunctionMesh,
            distance_unit: str = 'bohr_radius',
            colormap = plt.get_cmap('inferno'),
            norm = si.vis.AbsoluteRenormalize(),
            shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
            plot_limit: Optional[float] = None,
            slicer: str = 'get_mesh_slicer_spatial',
            **kwargs):
        unit_value, _ = u.get_unit_value_and_latex_from_unit(distance_unit)

        _slice = getattr(self, slicer)(plot_limit)

        color_mesh = axis.pcolormesh(
            self.theta_plot_mesh[_slice],
            self.r_theta_mesh[_slice] / unit_value,
            mesh[_slice],
            shading = shading,
            cmap = colormap,
            norm = norm,
            **kwargs
        )

        return color_mesh

    def plot_mesh(
            self,
            mesh: WavefunctionMesh,
            name: str = '',
            title: Optional[str] = None,
            distance_unit: str = 'bohr_radius',
            colormap = vis.COLORMAP_WAVEFUNCTION,
            norm = si.vis.AbsoluteRenormalize(),
            shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
            plot_limit: Optional[float] = None,
            slicer: str = 'get_mesh_slicer_spatial',
            aspect_ratio: float = 1,
            show_colorbar: bool = True,
            show_title: bool = True,
            show_axes: bool = True,
            title_size: float = 20,
            tick_label_size: float = 10,
            grid_kwargs: Optional[dict] = None,
            # overlay_probability_current = False,
            # probability_current_time_step = 0,
            **kwargs):
        if grid_kwargs is None:
            grid_kwargs = {}

        with si.vis.FigureManager(name = f'{self.spec.name}__{name}', aspect_ratio = aspect_ratio, **kwargs) as figman:
            fig = figman.fig

            fig.set_tight_layout(True)
            axis = plt.subplot(111, projection = 'polar')
            axis.set_theta_zero_location('N')
            axis.set_theta_direction('clockwise')

            unit_value, unit_latex = u.get_unit_value_and_latex_from_unit(distance_unit)

            color_mesh = self.attach_mesh_to_axis(
                axis, mesh,
                distance_unit = distance_unit,
                colormap = colormap,
                norm = norm,
                shading = shading,
                plot_limit = plot_limit,
                slicer = slicer
            )
            # if overlay_probability_current:
            #     quiv = self.attach_probability_current_to_axis(axis, plot_limit = plot_limit, distance_unit = distance_unit)

            if title is not None and title != '' and show_axes and show_title:
                title = axis.set_title(title, fontsize = title_size)
                title.set_x(.03)  # move title to the upper left corner
                title.set_y(.97)

            # make a colorbar
            if show_colorbar and show_axes:
                cbar_axis = fig.add_axes([1.01, .1, .04, .8])  # add a new axis for the cbar so that the old axis can stay square
                cbar = plt.colorbar(mappable = color_mesh, cax = cbar_axis)
                cbar.ax.tick_params(labelsize = tick_label_size)

            axis.grid(True, color = si.vis.CMAP_TO_OPPOSITE[colormap.name], **{**si.vis.COLORMESH_GRID_KWARGS, **grid_kwargs})  # change grid color to make it show up against the colormesh
            angle_labels = [f'{s}\u00b0' for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
            axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.09, labels = angle_labels)

            axis.tick_params(axis = 'both', which = 'major', labelsize = tick_label_size)  # increase size of tick labels
            axis.tick_params(axis = 'y', which = 'major', colors = si.vis.CMAP_TO_OPPOSITE[colormap.name], pad = 3)  # make r ticks a color that shows up against the colormesh
            # axis.tick_params(axis = 'both', which = 'both', length = 0)

            axis.set_rlabel_position(80)

            max_yticks = 5
            yloc = plt.MaxNLocator(max_yticks, symmetric = False, prune = 'both')
            axis.yaxis.set_major_locator(yloc)

            fig.canvas.draw()  # must draw early to modify the axis text

            tick_labels = axis.get_yticklabels()
            for t in tick_labels:
                t.set_text(t.get_text() + rf'${unit_latex}$')
            axis.set_yticklabels(tick_labels)

            if plot_limit is not None and plot_limit < self.r_max:
                axis.set_rmax((plot_limit - (self.delta_r / 2)) / unit_value)
            else:
                axis.set_rmax((self.r_max - (self.delta_r / 2)) / unit_value)

            if not show_axes:
                axis.axis('off')

    def attach_g_to_axis(
            self,
            axis: plt.Axes,
            colormap = plt.get_cmap('richardson'),
            norm = None,
            **kwargs):
        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.space_g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        return self.attach_mesh_to_axis(
            axis,
            self.space_g,
            colormap = colormap,
            norm = norm,
            **kwargs
        )

    def plot_g(
            self,
            name_postfix: str = '',
            colormap = plt.get_cmap('richardson'),
            norm = None,
            **kwargs):
        title = r'$g$'
        name = 'g' + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.space_g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        self.plot_mesh(
            self.space_g,
            name = name,
            title = title,
            colormap = colormap,
            norm = norm,
            show_colorbar = False,
            **kwargs
        )

    def attach_psi_to_axis(
            self,
            axis: plt.Axes,
            colormap = plt.get_cmap('richardson'),
            norm = None,
            **kwargs):
        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.space_psi) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        return self.attach_mesh_to_axis(
            axis,
            self.space_psi,
            colormap = colormap,
            norm = norm,
            **kwargs
        )

    def plot_psi(
            self,
            name_postfix: str = '',
            colormap = plt.get_cmap('richardson'),
            norm = None,
            **kwargs):
        title = r'$\Psi$'
        name = 'psi' + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.space_psi) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        self.plot_mesh(
            self.space_psi,
            name = name,
            title = title,
            colormap = colormap,
            norm = norm,
            show_colorbar = False,
            **kwargs
        )

    def update_g_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.space_g, **kwargs)

    def update_psi_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.space_psi, **kwargs)

    # I have no idea what this method does, sinec it doesn't use mesh...
    def attach_mesh_repr_to_axis(
            self,
            axis: plt.Axes,
            mesh,
            distance_unit: str = 'bohr_radius',
            colormap = plt.get_cmap('inferno'),
            norm = si.vis.AbsoluteRenormalize(),
            shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
            plot_limit: Optional[float] = None,
            slicer: str = 'get_mesh_slicer',
            **kwargs):
        unit_value, _ = u.get_unit_value_and_latex_from_unit(distance_unit)

        _slice = getattr(self, slicer)(plot_limit)

        color_mesh = axis.pcolormesh(
            self.l_mesh[_slice],
            self.r_mesh[_slice] / unit_value,
            mesh[_slice],
            shading = shading,
            cmap = colormap,
            norm = norm,
            **kwargs
        )

        return color_mesh

    def plot_mesh_repr(
            self,
            mesh,
            name: str = '',
            title: Optional[str] = None,
            distance_unit: str = 'bohr_radius',
            colormap = vis.COLORMAP_WAVEFUNCTION,
            norm = si.vis.AbsoluteRenormalize(),
            shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
            plot_limit: Optional[float] = None,
            slicer: str = 'get_mesh_slicer',
            aspect_ratio: float = si.vis.GOLDEN_RATIO,
            show_colorbar: bool = True,
            show_title: bool = True,
            show_axes: bool = True,
            title_y_adjust: float = 1.1,
            title_size: float = 12,
            axis_label_size: float = 12,
            tick_label_size: float = 10,
            grid_kwargs: Optional[dict] = None,
            **kwargs):
        if grid_kwargs is None:
            grid_kwargs = {}
        with si.vis.FigureManager(name = f'{self.spec.name}__{name}', aspect_ratio = aspect_ratio, **kwargs) as figman:
            fig = figman.fig

            fig.set_tight_layout(True)
            axis = plt.subplot(111)

            unit_value, unit_latex = u.get_unit_value_and_latex_from_unit(distance_unit)

            color_mesh = self.attach_mesh_repr_to_axis(
                axis, mesh,
                distance_unit = distance_unit,
                colormap = colormap,
                norm = norm,
                shading = shading,
                plot_limit = plot_limit,
                slicer = slicer
            )

            axis.set_xlabel(r'$\ell$', fontsize = axis_label_size)
            axis.set_ylabel(rf'$r$ (${unit_latex}$)', fontsize = axis_label_size)
            if title is not None and title != '' and show_axes and show_title:
                title = axis.set_title(title, fontsize = title_size)
                title.set_y(title_y_adjust)  # move title up a bit

            # make a colorbar
            if show_colorbar and show_axes:
                cbar = fig.colorbar(mappable = color_mesh, ax = axis)
                cbar.ax.tick_params(labelsize = tick_label_size)

            axis.grid(True, color = si.vis.CMAP_TO_OPPOSITE[colormap.name], **{**si.vis.COLORMESH_GRID_KWARGS, **grid_kwargs})  # change grid color to make it show up against the colormesh

            axis.tick_params(labelright = True, labeltop = True)  # ticks on all sides
            axis.tick_params(axis = 'both', which = 'major', labelsize = tick_label_size)  # increase size of tick labels
            # axis.tick_params(axis = 'both', which = 'both', length = 0)

            y_ticks = axis.yaxis.get_major_ticks()
            y_ticks[0].label1.set_visible(False)
            y_ticks[0].label2.set_visible(False)
            y_ticks[-1].label1.set_visible(False)
            y_ticks[-1].label2.set_visible(False)

            axis.axis('tight')

            if not show_axes:
                axis.axis('off')

    def plot_g_repr(
            self,
            name_postfix: str = '',
            title: Optional[str] = None,
            colormap = plt.get_cmap('richardson'),
            norm = None,
            **kwargs):
        if title is None:
            title = r'$g$'
        name = 'g_repr' + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        self.plot_mesh_repr(
            self.g,
            name = name,
            title = title,
            colormap = colormap,
            norm = norm,
            show_colorbar = False,
            **kwargs
        )

    def attach_g_repr_to_axis(
            self,
            axis: plt.Axes,
            colormap = plt.get_cmap('richardson'),
            norm = None,
            **kwargs):

        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        return self.attach_mesh_repr_to_axis(
            axis,
            self.g,
            colormap = colormap,
            norm = norm,
            **kwargs
        )

    def plot_electron_momentum_spectrum(
            self,
            r_type: u.Unit = 'wavenumber',
            r_scale: u.Unit = 'per_nm',
            r_lower_lim: float = u.twopi * .01 * u.per_nm,
            r_upper_lim: float = u.twopi * 10 * u.per_nm,
            r_points: int = 100,
            theta_points: int = 360,
            g: GMesh = None,
            **kwargs):
        if r_type not in ('wavenumber', 'energy', 'momentum'):
            raise ValueError("Invalid argument to plot_electron_spectrum: r_type must be either 'wavenumber', 'energy', or 'momentum'")

        thetas = np.linspace(0, u.twopi, theta_points)
        r = np.linspace(r_lower_lim, r_upper_lim, r_points)

        if r_type == 'wavenumber':
            wavenumbers = r
        elif r_type == 'energy':
            wavenumbers = core.electron_wavenumber_from_energy(r)
        elif r_type == 'momentum':
            wavenumbers = r / u.hbar

        if g is None:
            g = self.g

        theta_mesh, wavenumber_mesh, inner_product_mesh = self.inner_product_with_plane_waves(thetas, wavenumbers, g = g)

        if r_type == 'wavenumber':
            r_mesh = wavenumber_mesh
        elif r_type == 'energy':
            r_mesh = core.electron_energy_from_wavenumber(wavenumber_mesh)
        elif r_type == 'momentum':
            r_mesh = wavenumber_mesh * u.hbar

        return self.plot_electron_momentum_spectrum_from_meshes(
            theta_mesh, r_mesh, inner_product_mesh,
            r_type,
            r_scale,
            **kwargs
        )

    def plot_electron_momentum_spectrum_from_meshes(
            self,
            theta_mesh,
            r_mesh,
            inner_product_mesh,
            r_type: str,
            r_scale: float,
            log: bool = False,
            shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
            **kwargs):
        """
        Generate a polar plot of the wavefunction decomposed into plane waves.

        The radial dimension can be displayed in wavenumbers, energy, or momentum. The angle is the angle of the plane wave in the z-x plane (because m=0, the decomposition is symmetric in the x-y plane).

        :param r_type: type of unit for the radial axis ('wavenumber', 'energy', or 'momentum')
        :param r_scale: unit specification for the radial dimension
        :param r_lower_lim: lower limit for the radial dimension
        :param r_upper_lim: upper limit for the radial dimension
        :param r_points: number of points for the radial dimension
        :param theta_points: number of points for the angular dimension
        :param log: True to displayed logged data, False otherwise (default: False)
        :param kwargs: kwargs are passed to compy.utils.FigureManager
        :return: the FigureManager generated during plot creation
        """
        if r_type not in ('wavenumber', 'energy', 'momentum'):
            raise ValueError("Invalid argument to plot_electron_spectrum: r_type must be either 'wavenumber', 'energy', or 'momentum'")

        r_unit_value, r_unit_name = u.get_unit_value_and_latex_from_unit(r_scale)

        plot_kwargs = {**dict(aspect_ratio = 1), **kwargs}

        r_mesh = np.real(r_mesh)
        overlap_mesh = np.abs(inner_product_mesh) ** 2

        with si.vis.FigureManager(self.sim.name + '__electron_spectrum', **plot_kwargs) as figman:
            fig = figman.fig

            fig.set_tight_layout(True)

            axis = plt.subplot(111, projection = 'polar')
            axis.set_theta_zero_location('N')
            axis.set_theta_direction('clockwise')

            figman.name += f'__{r_type}'

            norm = None
            if log:
                norm = matplotlib.colors.LogNorm(vmin = np.nanmin(overlap_mesh), vmax = np.nanmax(overlap_mesh))
                figman.name += '__log'

            color_mesh = axis.pcolormesh(
                theta_mesh,
                r_mesh / r_unit_value,
                overlap_mesh,
                shading = shading,
                norm = norm,
                cmap = 'viridis',
            )

            # make a colorbar
            cbar_axis = fig.add_axes([1.01, .1, .04, .8])  # add a new axis for the cbar so that the old axis can stay square
            cbar = plt.colorbar(mappable = color_mesh, cax = cbar_axis)
            cbar.ax.tick_params(labelsize = 10)

            axis.grid(True, color = si.vis.COLOR_OPPOSITE_VIRIDIS, **si.vis.COLORMESH_GRID_KWARGS)  # change grid color to make it show up against the colormesh
            angle_labels = [f'{s}\u00b0' for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
            axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)

            axis.tick_params(axis = 'both', which = 'major', labelsize = 8)  # increase size of tick labels
            axis.tick_params(axis = 'y', which = 'major', colors = si.vis.COLOR_OPPOSITE_VIRIDIS, pad = 3)  # make r ticks a color that shows up against the colormesh
            axis.tick_params(axis = 'both', which = 'both', length = 0)

            axis.set_rlabel_position(80)

            max_yticks = 5
            yloc = plt.MaxNLocator(max_yticks, symmetric = False, prune = 'both')
            axis.yaxis.set_major_locator(yloc)

            fig.canvas.draw()  # must draw early to modify the axis text

            tick_labels = axis.get_yticklabels()
            for t in tick_labels:
                t.set_text(t.get_text() + rf'${r_unit_name}$')
            axis.set_yticklabels(tick_labels)

            axis.set_rmax(np.nanmax(r_mesh) / r_unit_value)

        return figman
