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


class SimilarityOperator(DotOperator):
    def __init__(self, matrix: 'meshes.SparseMatrixOperator', *, wrapping_direction: Optional['meshes.WrappingDirection'], parity: str):
        super().__init__(matrix, wrapping_direction = wrapping_direction)

        self.parity = parity
        self.transform = getattr(self, f'u_{self.parity}_g')

    def __repr__(self):
        op_repr = repr(self.matrix).replace('\n', '')
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


def apply_operators(mesh, g: 'meshes.GMesh', operators: Iterable[MeshOperator]):
    """Operators should be entered in operation (the order they would act on something on their right)"""
    current_wrapping_direction = None

    for operator in operators:
        g, current_wrapping_direction = operator.apply(mesh, g, current_wrapping_direction)

    return mesh.wrap_vector(g, current_wrapping_direction)


class Operators(abc.ABC):
    @abc.abstractmethod
    def kinetic_energy(self, mesh):
        raise NotImplementedError


class CylindricalSliceLengthGaugeOperators(Operators):
    def kinetic_energy(self, mesh) -> Iterable[MeshOperator]:
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
            DotOperator(z_kinetic, wrapping_direction = meshes.WrappingDirection.Z),
            DotOperator(rho_kinetic, wrapping_direction = meshes.WrappingDirection.RHO),
        )

    @si.utils.memoize
    def internal_hamiltonian(self, mesh):
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

    def interaction_hamiltonian(self, mesh):
        electric_potential_energy_mesh = mesh.spec.electric_potential(t = mesh.sim.time, distance_along_polarization = mesh.z_mesh, test_charge = mesh.spec.test_charge)

        interaction_hamiltonian_z = sparse.diags(mesh.flatten_mesh(electric_potential_energy_mesh, meshes.WrappingDirection.Z))
        interaction_hamiltonian_rho = sparse.diags(mesh.flatten_mesh(electric_potential_energy_mesh, meshes.WrappingDirection.RHO))

        return (
            DotOperator(interaction_hamiltonian_z, wrapping_direction = meshes.WrappingDirection.Z),
            DotOperator(interaction_hamiltonian_rho, wrapping_direction = meshes.WrappingDirection.RHO),
        )

    def total_hamiltonian(self, mesh):
        internal_operators = self.internal_hamiltonian(mesh)
        interaction_operators = self.interaction_hamiltonian(mesh)

        direction_to_internal_operators = {oper.wrapping_direction: oper for oper in internal_operators}
        direction_to_interaction_operators = {oper.wrapping_direction: oper for oper in interaction_operators}

        pre = 1 / len(direction_to_internal_operators)
        total_operators = []
        for direction, internal_operator in sorted(direction_to_internal_operators.items()):
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

    def alpha(self, j) -> float:
        x = (j ** 2) + (2 * j)
        return (x + 1) / (x + 0.75)

    def beta(self, j) -> float:
        x = 2 * (j ** 2) + (2 * j)
        return (x + 1) / (x + 0.5)

    def gamma(self, j) -> float:
        """For radial probability current."""
        return 1 / ((j ** 2) - 0.25)

    # THESE ARE FOR THE LAGRANGIAN EVOLUTION EQNS ONLY SO FAR
    def kinetic_energy(self, mesh) -> Iterable[MeshOperator]:
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
    def internal_hamiltonian(self, mesh):
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
    def interaction_hamiltonian_matrix_without_field(self, mesh):
        l_prefactor = -mesh.spec.test_charge * mesh.flatten_mesh(mesh.r_mesh, 'l')[:-1]

        def c_l(l) -> float:
            """a particular set of 3j coefficients for SphericalHarmonicMesh"""
            return (l + 1) / np.sqrt(((2 * l) + 1) * ((2 * l) + 3))

        l_diagonal = np.zeros(mesh.mesh_points, dtype = np.complex128)
        l_offdiagonal = np.zeros(mesh.mesh_points - 1, dtype = np.complex128)
        for l_index in range(mesh.mesh_points - 1):
            if (l_index + 1) % mesh.spec.l_bound != 0:
                l = (l_index % mesh.spec.l_bound)
                l_offdiagonal[l_index] = c_l(l)
        l_offdiagonal *= l_prefactor

        l_interaction = sparse.diags([l_offdiagonal, l_diagonal, l_offdiagonal], offsets = (-1, 0, 1))

        return l_interaction

    def interaction_hamiltonian(self, mesh):
        l_interaction = self.interaction_hamiltonian_matrix_without_field(mesh)

        efield = mesh.spec.electric_potential.get_electric_field_amplitude(mesh.sim.time + (mesh.spec.time_step / 2))
        return DotOperator(efield * l_interaction, wrapping_direction = meshes.WrappingDirection.L),

    # TIGHT DEPENDENCE ON IMPLEMENTATION RIGHT NOW
    def total_hamiltonian(self, mesh):
        r_ham, = self.internal_hamiltonian(mesh)
        l_ham, = self.interaction_hamiltonian(mesh)

        return l_ham, r_ham
