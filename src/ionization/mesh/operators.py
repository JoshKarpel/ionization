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

from .. import cy
from . import meshes


def add_to_diagonal_sparse_matrix_diagonal(dia_matrix: 'meshes.SparseMatrixOperator', value = 1) -> sparse.dia_matrix:
    s = dia_matrix.copy()
    s.setdiag(s.diagonal() + value)
    return s


def add_to_diagonal_sparse_matrix_diagonal_inplace(dia_matrix: 'meshes.SparseMatrixOperator', value = 1) -> sparse.dia_matrix:
    dia_matrix.data[1] += value
    return dia_matrix


class MeshOperator(abc.ABC):
    def __init__(self, matrix: 'meshes.SparseMatrixOperator', *, wrapping_direction: Optional['meshes.WrappingDirection']):
        self.matrix = matrix
        self.wrapping_direction = wrapping_direction

    def __repr__(self):
        matrix_repr = repr(self.matrix).replace('\n', '')
        return f"{self.__class__.__name__}(operator = {matrix_repr}, wrapping_direction = '{self.wrapping_direction}')"

    def apply(self, mesh: 'meshes.QuantumMesh', g: 'meshes.GVector', current_wrapping_direction):
        if current_wrapping_direction != self.wrapping_direction:
            g = mesh.flatten_mesh(mesh.wrap_vector(g, current_wrapping_direction), self.wrapping_direction)

        result = self._apply(g)

        return result, self.wrapping_direction

    @abc.abstractmethod
    def _apply(self, g: 'meshes.GVector'):
        raise NotImplementedError


class DotOperator(MeshOperator):
    def _apply(self, g: 'meshes.GVector') -> 'meshes.GVector':
        return self.matrix.dot(g)


class TDMAOperator(MeshOperator):
    def _apply(self, g: 'meshes.GVector') -> 'meshes.GVector':
        return cy.tdma(self.matrix, g)


class Parity(si.utils.StrEnum):
    EVEN = 'even'
    ODD = 'odd'


class SimilarityOperator(DotOperator):
    def __init__(self, matrix: 'meshes.SparseMatrixOperator', *, wrapping_direction: Optional['meshes.WrappingDirection'], parity: Parity):
        super().__init__(matrix, wrapping_direction = wrapping_direction)

        self.parity = parity
        self.transform = getattr(self, f'u_{self.parity}_g')

    def __repr__(self):
        op_repr = repr(self.matrix).replace('\n', '')
        return f"{self.__class__.__name__}({op_repr}, wrapping_direction = {self.wrapping_direction}, parity = {self.parity})"

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


def apply_operators(mesh, g: 'meshes.GMesh', operators: Iterable[MeshOperator]):
    """Operators should be entered in operation (the order they would act on something on their right)"""
    current_wrapping_direction = None

    for operator in operators:
        g, current_wrapping_direction = operator.apply(mesh, g, current_wrapping_direction)

    return mesh.wrap_vector(g, current_wrapping_direction)


class Operators(abc.ABC):
    @abc.abstractmethod
    def kinetic_energy(self, mesh: 'meshes.QuantumMesh') -> Tuple[MeshOperator, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def internal_hamiltonian(self, mesh: 'meshes.QuantumMesh') -> Tuple[MeshOperator, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def interaction_hamiltonian(self, mesh: 'meshes.QuantumMesh') -> Tuple[MeshOperator, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def total_hamiltonian(self, mesh: 'meshes.QuantumMesh') -> Tuple[MeshOperator, ...]:
        raise NotImplementedError

    def info(self):
        return si.Info(header = f'{self.__class__.__name__}')


class CylindricalSliceLengthGaugeOperators(Operators):
    def kinetic_energy(self, mesh) -> Tuple[MeshOperator, ...]:
        z_prefactor = -(u.hbar ** 2) / (2 * mesh.spec.test_mass * (mesh.delta_z ** 2))
        rho_prefactor = -(u.hbar ** 2) / (2 * mesh.spec.test_mass * (mesh.delta_rho ** 2))

        z_diagonal = z_prefactor * (-2) * np.ones(mesh.mesh_points, dtype = np.complex128)
        z_offdiagonal = z_prefactor * np.array([1 if (z_index + 1) % mesh.spec.z_points != 0 else 0 for z_index in range(mesh.mesh_points - 1)], dtype = np.complex128)

        def c(j):
            return j / np.sqrt((j ** 2) - 0.25)

        rho_diagonal = rho_prefactor * (-2) * np.ones(mesh.mesh_points, dtype = np.complex128)
        rho_offdiagonal = np.zeros(mesh.mesh_points - 1, dtype = np.complex128)
        for rho_index in range(mesh.mesh_points - 1):
            if (rho_index + 1) % mesh.spec.rho_points != 0:
                j = (rho_index % mesh.spec.rho_points) + 1  # get j for the upper diagonal
                rho_offdiagonal[rho_index] = c(j)
        rho_offdiagonal *= rho_prefactor

        z_kinetic = sparse.diags([z_offdiagonal, z_diagonal, z_offdiagonal], offsets = (-1, 0, 1))
        rho_kinetic = sparse.diags([rho_offdiagonal, rho_diagonal, rho_offdiagonal], offsets = (-1, 0, 1))

        return (
            DotOperator(rho_kinetic, wrapping_direction = meshes.WrappingDirection.RHO),
            DotOperator(z_kinetic, wrapping_direction = meshes.WrappingDirection.Z),
        )

    @si.utils.memoize
    def internal_hamiltonian(self, mesh) -> Tuple[MeshOperator, ...]:
        kinetic_operators = self.kinetic_energy(mesh)
        potential_mesh = mesh.spec.internal_potential(r = mesh.r_mesh, test_charge = mesh.spec.test_charge)

        pre = 1 / len(kinetic_operators)

        return tuple(
            DotOperator(
                add_to_diagonal_sparse_matrix_diagonal(
                    k.matrix,
                    value = pre * mesh.flatten_mesh(potential_mesh, k.wrapping_direction)
                ),
                wrapping_direction = k.wrapping_direction,
            )
            for k in kinetic_operators
        )

    def interaction_hamiltonian(self, mesh) -> Tuple[MeshOperator, ...]:
        electric_potential_energy_mesh = mesh.spec.electric_potential(t = mesh.sim.time, distance_along_polarization = mesh.z_mesh, test_charge = mesh.spec.test_charge)

        interaction_hamiltonian_z = sparse.diags(mesh.flatten_mesh(electric_potential_energy_mesh, meshes.WrappingDirection.Z))
        interaction_hamiltonian_rho = sparse.diags(mesh.flatten_mesh(electric_potential_energy_mesh, meshes.WrappingDirection.RHO))

        return (
            DotOperator(interaction_hamiltonian_rho, wrapping_direction = meshes.WrappingDirection.RHO),
            DotOperator(interaction_hamiltonian_z, wrapping_direction = meshes.WrappingDirection.Z),
        )

    def total_hamiltonian(self, mesh) -> Tuple[MeshOperator, ...]:
        internal_operators = self.internal_hamiltonian(mesh)
        interaction_operators = self.interaction_hamiltonian(mesh)

        direction_to_internal_operators = {oper.wrapping_direction: oper for oper in internal_operators}
        direction_to_interaction_operators = {oper.wrapping_direction: oper for oper in interaction_operators}

        pre = 1 / len(direction_to_internal_operators)
        total_operators = []
        for direction, internal_operator in direction_to_internal_operators.items():
            total_operators.append(
                DotOperator(
                    add_to_diagonal_sparse_matrix_diagonal(internal_operator.matrix, value = pre * direction_to_interaction_operators[direction].matrix.diagonal()),
                    wrapping_direction = direction,
                )
            )

        return tuple(total_operators)


class SphericalSliceLengthGaugeOperators(Operators):
    def kinetic_energy(self, mesh) -> Tuple[MeshOperator, ...]:
        r_prefactor = -(u.hbar ** 2) / (2 * u.electron_mass_reduced * (mesh.delta_r ** 2))
        theta_prefactor = -(u.hbar ** 2) / (2 * u.electron_mass_reduced * ((mesh.delta_r * mesh.delta_theta) ** 2))

        r_diagonal = r_prefactor * (-2) * np.ones(mesh.mesh_points, dtype = np.complex128)
        r_offdiagonal = r_prefactor * np.array([1 if (z_index + 1) % mesh.spec.r_points != 0 else 0 for z_index in range(mesh.mesh_points - 1)], dtype = np.complex128)

        @si.utils.memoize
        def theta_j_prefactor(x):
            return 1 / (x + 0.5) ** 2

        @si.utils.memoize
        def sink(x):
            return np.sin(x * mesh.delta_theta)

        @si.utils.memoize
        def sqrt_sink_ratio(x_num, x_den):
            return np.sqrt(sink(x_num) / sink(x_den))

        @si.utils.memoize
        def cotank(x):
            return 1 / np.tan(x * mesh.delta_theta)

        theta_diagonal = (-2) * np.ones(mesh.mesh_points, dtype = np.complex128)
        for theta_index in range(mesh.mesh_points):
            j = theta_index // mesh.spec.theta_points
            theta_diagonal[theta_index] *= theta_j_prefactor(j)
        theta_diagonal *= theta_prefactor

        theta_upper_diagonal = np.zeros(mesh.mesh_points - 1, dtype = np.complex128)
        theta_lower_diagonal = np.zeros(mesh.mesh_points - 1, dtype = np.complex128)
        for theta_index in range(mesh.mesh_points - 1):
            if (theta_index + 1) % mesh.spec.theta_points != 0:
                j = theta_index // mesh.spec.theta_points
                k = theta_index % mesh.spec.theta_points
                k_p = k + 1  # add 1 because the entry for the lower diagonal is really for the next point (k -> k + 1), not this one
                theta_upper_diagonal[theta_index] = theta_j_prefactor(j) * (1 + (mesh.delta_theta / 2) * cotank(k + 0.5)) * sqrt_sink_ratio(k + 0.5, k + 1.5)
                theta_lower_diagonal[theta_index] = theta_j_prefactor(j) * (1 - (mesh.delta_theta / 2) * cotank(k_p + 0.5)) * sqrt_sink_ratio(k_p + 0.5, k_p - 0.5)
        theta_upper_diagonal *= theta_prefactor
        theta_lower_diagonal *= theta_prefactor

        r_kinetic = sparse.diags([r_offdiagonal, r_diagonal, r_offdiagonal], offsets = (-1, 0, 1))
        theta_kinetic = sparse.diags([theta_lower_diagonal, theta_diagonal, theta_upper_diagonal], offsets = (-1, 0, 1))

        return (
            DotOperator(theta_kinetic, wrapping_direction = meshes.WrappingDirection.THETA),
            DotOperator(r_kinetic, wrapping_direction = meshes.WrappingDirection.R),
        )

    @si.utils.memoize
    def internal_hamiltonian(self, mesh) -> Tuple[MeshOperator, ...]:
        kinetic_operators = self.kinetic_energy(mesh)
        potential_mesh = mesh.spec.internal_potential(r = mesh.r_mesh, test_charge = mesh.spec.test_charge)

        pre = 1 / len(kinetic_operators)

        return tuple(
            DotOperator(
                add_to_diagonal_sparse_matrix_diagonal(
                    k.matrix,
                    value = pre * mesh.flatten_mesh(potential_mesh, k.wrapping_direction)
                ),
                wrapping_direction = k.wrapping_direction,
            )
            for k in kinetic_operators
        )

    def interaction_hamiltonian(self, mesh) -> Tuple[MeshOperator, ...]:
        electric_potential_energy_mesh = mesh.spec.electric_potential(t = mesh.sim.time, distance_along_polarization = mesh.z_mesh, test_charge = mesh.spec.test_charge)

        interaction_hamiltonian_r = sparse.diags(mesh.flatten_mesh(electric_potential_energy_mesh, meshes.WrappingDirection.R))
        interaction_hamiltonian_theta = sparse.diags(mesh.flatten_mesh(electric_potential_energy_mesh, meshes.WrappingDirection.THETA))

        return (
            DotOperator(interaction_hamiltonian_theta, wrapping_direction = meshes.WrappingDirection.THETA),
            DotOperator(interaction_hamiltonian_r, wrapping_direction = meshes.WrappingDirection.R),
        )

    def total_hamiltonian(self, mesh) -> Tuple[MeshOperator, ...]:
        internal_operators = self.internal_hamiltonian(mesh)
        interaction_operators = self.interaction_hamiltonian(mesh)

        direction_to_internal_operators = {oper.wrapping_direction: oper for oper in internal_operators}
        direction_to_interaction_operators = {oper.wrapping_direction: oper for oper in interaction_operators}

        pre = 1 / len(direction_to_internal_operators)
        total_operators = []
        for direction, internal_operator in direction_to_internal_operators.items():
            total_operators.append(
                DotOperator(
                    add_to_diagonal_sparse_matrix_diagonal(internal_operator.matrix, value = pre * direction_to_interaction_operators[direction].matrix.diagonal()),
                    wrapping_direction = direction,
                )
            )

        return tuple(total_operators)


class SphericalHarmonicLengthGaugeOperators(Operators):
    def __init__(self, hydrogen_zero_angular_momentum_correction = True):
        self.hydrogen_zero_angular_momentum_correction = hydrogen_zero_angular_momentum_correction

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Hydrogen Zero Angular Momentum Correction', self.hydrogen_zero_angular_momentum_correction)

    def alpha(self, j) -> float:
        x = (j ** 2) + (2 * j)
        return (x + 1) / (x + 0.75)

    def beta(self, j) -> float:
        x = 2 * (j ** 2) + (2 * j)
        return (x + 1) / (x + 0.5)

    def gamma(self, j) -> float:
        """For radial probability current."""
        return 1 / ((j ** 2) - 0.25)

    def c_l(self, l) -> float:
        """a particular set of 3j coefficients for SphericalHarmonicMesh"""
        return (l + 1) / np.sqrt(((2 * l) + 1) * ((2 * l) + 3))

    # THESE ARE FOR THE LAGRANGIAN EVOLUTION EQNS ONLY SO FAR
    def kinetic_energy(self, mesh) -> Tuple[MeshOperator, ...]:
        r_prefactor = -(u.hbar ** 2) / (2 * u.electron_mass_reduced * (mesh.delta_r ** 2))

        r_diagonal = np.zeros(mesh.mesh_points, dtype = np.complex128)
        r_offdiagonal = np.zeros(mesh.mesh_points - 1, dtype = np.complex128)
        for r_index in range(mesh.mesh_points):
            j = r_index % mesh.spec.r_points
            r_diagonal[r_index] = self.beta(j)
        if self.hydrogen_zero_angular_momentum_correction:
            dr = mesh.delta_r / u.bohr_radius
            r_diagonal[0] += dr * (1 + dr) / 8  # modify beta_j for l = 0   (see notes)

        for r_index in range(mesh.mesh_points - 1):
            if (r_index + 1) % mesh.spec.r_points != 0:
                j = (r_index % mesh.spec.r_points)
                r_offdiagonal[r_index] = self.alpha(j)
        r_diagonal *= -2 * r_prefactor
        r_offdiagonal *= r_prefactor

        effective_potential_mesh = ((u.hbar ** 2) / (2 * u.electron_mass_reduced)) * mesh.l_mesh * (mesh.l_mesh + 1) / (mesh.r_mesh ** 2)
        r_diagonal += mesh.flatten_mesh(effective_potential_mesh, meshes.WrappingDirection.R)

        r_kinetic = sparse.diags([r_offdiagonal, r_diagonal, r_offdiagonal], offsets = (-1, 0, 1))

        return DotOperator(r_kinetic, wrapping_direction = meshes.WrappingDirection.R),

    @si.utils.memoize
    def internal_hamiltonian(self, mesh) -> Tuple[MeshOperator, ...]:
        kinetic_operators = self.kinetic_energy(mesh)
        potential_mesh = mesh.spec.internal_potential(r = mesh.r_mesh, test_charge = mesh.spec.test_charge)

        pre = 1 / len(kinetic_operators)

        return tuple(
            DotOperator(
                add_to_diagonal_sparse_matrix_diagonal(
                    k.matrix,
                    value = pre * mesh.flatten_mesh(potential_mesh, k.wrapping_direction)
                ),
                wrapping_direction = k.wrapping_direction,
            )
            for k in kinetic_operators
        )

    @si.utils.memoize
    def interaction_hamiltonian_matrices_without_field(self, mesh) -> 'meshes.SparseMatrixOperator':
        l_prefactor = -mesh.spec.test_charge * mesh.flatten_mesh(mesh.r_mesh, 'l')[:-1]

        l_diagonal = np.zeros(mesh.mesh_points, dtype = np.complex128)
        l_offdiagonal = np.zeros(mesh.mesh_points - 1, dtype = np.complex128)
        for l_index in range(mesh.mesh_points - 1):
            if (l_index + 1) % mesh.spec.l_bound != 0:
                l = (l_index % mesh.spec.l_bound)
                l_offdiagonal[l_index] = self.c_l(l)
        l_offdiagonal *= l_prefactor

        l_interaction = sparse.diags([l_offdiagonal, l_diagonal, l_offdiagonal], offsets = (-1, 0, 1))

        return l_interaction

    def interaction_hamiltonian(self, mesh) -> Tuple[MeshOperator, ...]:
        l_interaction = self.interaction_hamiltonian_matrices_without_field(mesh)

        efield = mesh.spec.electric_potential.get_electric_field_amplitude(mesh.sim.time + (mesh.spec.time_step / 2))  # TODO: decouple this
        return DotOperator(efield * l_interaction, wrapping_direction = meshes.WrappingDirection.L),

    # TIGHT DEPENDENCE ON IMPLEMENTATION RIGHT NOW
    def total_hamiltonian(self, mesh) -> Tuple[MeshOperator, ...]:
        r_hams = self.internal_hamiltonian(mesh)
        l_hams = self.interaction_hamiltonian(mesh)

        return (*l_hams, *r_hams)

    def split_interaction_operators(self, mesh, interaction_operators, tau: complex) -> Tuple[MeshOperator, ...]:
        interaction_operator, = interaction_operators  # only one operator in length gauge
        a = tau * interaction_operator.matrix.data[0][:-1]

        a_even, a_odd = a[::2], a[1::2]

        len_a = len(a)

        even_diag = np.zeros(len_a + 1, dtype = np.complex128)
        even_offdiag = np.zeros(len_a, dtype = np.complex128)
        odd_diag = np.zeros(len_a + 1, dtype = np.complex128)
        odd_offdiag = np.zeros(len_a, dtype = np.complex128)

        if len(mesh.r) % 2 != 0 and len(mesh.l) % 2 != 0:
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
            DotOperator(even, wrapping_direction = meshes.WrappingDirection.L),
            DotOperator(odd, wrapping_direction = meshes.WrappingDirection.L),
        )


class SphericalHarmonicVelocityGaugeOperators(SphericalHarmonicLengthGaugeOperators):
    def __init__(self, hydrogen_zero_angular_momentum_correction = True):
        super().__init__(hydrogen_zero_angular_momentum_correction = hydrogen_zero_angular_momentum_correction)

    @si.utils.memoize
    def interaction_hamiltonian_matrices_without_field(self, mesh) -> Tuple['meshes.SparseMatrixOperator', ...]:
        h1_prefactor = 1j * u.hbar * (mesh.spec.test_charge / mesh.spec.test_mass) / mesh.flatten_mesh(mesh.r_mesh, 'l')[:-1]

        h1_offdiagonal = np.zeros(mesh.mesh_points - 1, dtype = np.complex128)
        for l_index in range(mesh.mesh_points - 1):
            if (l_index + 1) % mesh.spec.l_bound != 0:
                l = (l_index % mesh.spec.l_bound)
                h1_offdiagonal[l_index] = self.c_l(l) * (l + 1)
        h1_offdiagonal *= h1_prefactor

        h1 = sparse.diags((-h1_offdiagonal, h1_offdiagonal), offsets = (-1, 1))

        h2_prefactor = 1j * u.hbar * (mesh.spec.test_charge / mesh.spec.test_mass) / (2 * mesh.delta_r)

        alpha_vec = self.alpha(np.array(range(len(mesh.r) - 1), dtype = np.complex128))
        alpha_block = sparse.diags((-alpha_vec, alpha_vec), offsets = (-1, 1))

        c_vec = self.c_l(np.array(range(len(mesh.l) - 1), dtype = np.complex128))
        c_block = sparse.diags((c_vec, c_vec), offsets = (-1, 1))

        h2 = h2_prefactor * sparse.kron(c_block, alpha_block, format = 'dia')

        return h1, h2

    def interaction_hamiltonian(self, mesh) -> Tuple['meshes.SparseMatrixOperator', ...]:
        h1, h2 = self.interaction_hamiltonian_matrices_without_field(mesh)
        vector_potential_amp = mesh.spec.electric_potential.get_vector_potential_amplitude_numeric(mesh.sim.times_to_current)  # decouple

        return vector_potential_amp * h1, vector_potential_amp * h2  # TODO: NOT MESH OPERATORS YET! SHOULD THEY BE?

    def split_interaction_operators(self, mesh, interaction_operators, tau: complex) -> Tuple[MeshOperator, ...]:
        h1, h2 = self.interaction_hamiltonian(mesh)

        h1_operators = self.split_h1(mesh, h1, tau)
        h2_operators = self.split_h2(mesh, h2, tau)

        return (*h1_operators, *h2_operators)

    def split_h1(self, mesh, h1, tau: complex) -> Tuple[MeshOperator, ...]:
        a = (tau * (-1j)) * h1.data[-1][1:]

        a_even, a_odd = a[::2], a[1::2]

        even_diag = np.zeros(len(a) + 1, dtype = np.complex128)
        even_offdiag = np.zeros(len(a), dtype = np.complex128)
        odd_diag = np.zeros(len(a) + 1, dtype = np.complex128)
        odd_offdiag = np.zeros(len(a), dtype = np.complex128)

        if len(mesh.r) % 2 != 0 and len(mesh.l) % 2 != 0:
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
            DotOperator(even, wrapping_direction = meshes.WrappingDirection.L),
            DotOperator(odd, wrapping_direction = meshes.WrappingDirection.L),
        )

    def split_h2(self, mesh, h2, tau: complex) -> Tuple[MeshOperator, ...]:
        len_r = len(mesh.r)

        a = h2.data[-1][len_r + 1:] * tau * (-1j)

        alpha_slices_even_l = []
        alpha_slices_odd_l = []
        for l in mesh.l:  # want last l but not last r, since unwrapped in r
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
        if mesh.l[-1] % 2 == 0:
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
                    new_odd_even_diag = np.tile(tile, 2)
                    even_sines[:-1:2] = np.sin(even_slice)

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
        if mesh.l[-1] % 2 != 0:
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
            SimilarityOperator(even_even_matrix, wrapping_direction = meshes.WrappingDirection.R, parity = Parity.EVEN),
            SimilarityOperator(even_odd_matrix, wrapping_direction = meshes.WrappingDirection.R, parity = Parity.EVEN),  # parity is based off FIRST splitting
            SimilarityOperator(odd_even_matrix, wrapping_direction = meshes.WrappingDirection.R, parity = Parity.ODD),
            SimilarityOperator(odd_odd_matrix, wrapping_direction = meshes.WrappingDirection.R, parity = Parity.ODD),
        )

        return operators
