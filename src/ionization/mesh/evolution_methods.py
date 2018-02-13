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
from . import sims, meshes


def add_to_diagonal_sparse_matrix_diagonal(dia_matrix: 'meshes.SparseMatrixOperator', value = 1) -> sparse.dia_matrix:
    s = dia_matrix.copy()
    s.setdiag(s.diagonal() + value)
    return s


def add_to_diagonal_sparse_matrix_diagonal_inplace(dia_matrix: 'meshes.SparseMatrixOperator', value = 1) -> sparse.dia_matrix:
    dia_matrix.data[1] += value
    return dia_matrix


class MeshOperator(abc.ABC):
    def __init__(self, operator: 'meshes.SparseMatrixOperator', *, wrapping_direction: Optional['meshes.WrappingDirection']):
        self.operator = operator
        self.wrapping_direction = wrapping_direction

    def __repr__(self):
        return f"{self.__class__.__name__}(operator = {repr(self.operator)}, wrapping_direction = '{self.wrapping_direction}')"

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
        return self.operator.dot(g)


class TDMAOperator(MeshOperator):
    def _apply(self, g: 'meshes.GVector') -> 'meshes.GVector':
        return cy.tdma(self.operator, g)


class SimilarityOperator(DotOperator):
    def __init__(self, operator: 'meshes.SparseMatrixOperator', *, wrapping_direction: Optional['meshes.WrappingDirection'], parity: str):
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


def apply_operators(mesh, g: 'meshes.GMesh', *operators: MeshOperator):
    """Operators should be entered in operation (the order they would act on something on their right)"""
    current_wrapping_direction = None

    for operator in operators:
        g, current_wrapping_direction = operator.apply(mesh, g, current_wrapping_direction)

    return mesh.wrap_vector(g, current_wrapping_direction)


class EvolutionMethod(abc.ABC):
    @abc.abstractmethod
    def evolve(self, mesh: 'meshes.QuantumMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
        raise NotImplementedError


class LineCrankNicolson(EvolutionMethod):
    def evolve(self, mesh: 'meshes.LineMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
        tau = time_step / (2 * u.hbar)

        interaction_operator = mesh.get_interaction_hamiltonian_matrix_operators()

        hamiltonian_x = mesh.get_internal_hamiltonian_matrix_operators()
        hamiltonian = 1j * tau * sparse.dia_matrix(hamiltonian_x + interaction_operator)

        ham_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian, 1)
        ham_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian, 1)

        operators = [
            DotOperator(ham_explicit, wrapping_direction = None),
            TDMAOperator(ham_implicit, wrapping_direction = None),
        ]

        return apply_operators(mesh, g, *operators)


class LineSplitOperator(EvolutionMethod):
    def evolve(self, mesh: 'meshes.LineMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
        tau = time_step / (2 * u.hbar)

        hamiltonian_x = mesh.get_internal_hamiltonian_matrix_operators()

        ham_x_explicit = add_to_diagonal_sparse_matrix_diagonal(-1j * tau * hamiltonian_x, 1)
        ham_x_implicit = add_to_diagonal_sparse_matrix_diagonal(1j * tau * hamiltonian_x, 1)

        split_operators = mesh._make_split_operator_evolution_operators(mesh.get_interaction_hamiltonian_matrix_operators(), tau)

        operators = [
            *split_operators,
            DotOperator(ham_x_explicit, wrapping_direction = None),
            TDMAOperator(ham_x_implicit, wrapping_direction = None),
            *reversed(split_operators),
        ]

        return apply_operators(mesh, g, *operators)


class LineSpectral(EvolutionMethod):
    def evolve(self, mesh: 'meshes.LineMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
        g = self._evolve_potential(mesh, g, time_step / 2)
        g = self._evolve_free(mesh, g, time_step)  # splitting order chosen for computational efficiency (only one FFT per time step)
        g = self._evolve_potential(mesh, g, time_step / 2)

        return g

    def _evolve_potential(self, mesh: 'meshes.LineMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
        pot = mesh.spec.internal_potential(t = mesh.sim.time, r = mesh.x_mesh, distance = mesh.x_mesh, test_charge = mesh.spec.test_charge)
        pot += mesh.spec.electric_potential(t = mesh.sim.time, r = mesh.x_mesh, distance = mesh.x_mesh, distance_along_polarization = mesh.x_mesh, test_charge = mesh.spec.test_charge)

        return g * np.exp(-1j * time_step * pot / u.hbar)

    def _evolve_free(self, mesh: 'meshes.LineMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
        return mesh.ifft(mesh.fft(g) * np.exp(mesh.free_evolution_prefactor * time_step) * mesh.wavenumber_mask)


class CylindricalSliceCrankNicolson(EvolutionMethod):
    def evolve(self, mesh: 'meshes.CylindricalSliceMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
        tau = time_step / (2 * u.hbar)

        hamiltonian_z, hamiltonian_rho = mesh.get_internal_hamiltonian_matrix_operators()
        interaction_hamiltonian_z, interaction_hamiltonian_rho = mesh.get_interaction_hamiltonian_matrix_operators()

        hamiltonian_z = 1j * tau * add_to_diagonal_sparse_matrix_diagonal(hamiltonian_z, value = 0.5 * interaction_hamiltonian_z.diagonal())
        hamiltonian_rho = 1j * tau * add_to_diagonal_sparse_matrix_diagonal(hamiltonian_rho, value = 0.5 * interaction_hamiltonian_rho.diagonal())

        hamiltonian_rho_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_rho, value = 1)
        hamiltonian_z_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian_z, value = 1)
        hamiltonian_z_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_z, value = 1)
        hamiltonian_rho_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian_rho, value = 1)

        operators = [
            DotOperator(hamiltonian_rho_explicit, wrapping_direction = meshes.WrappingDirection.RHO),
            TDMAOperator(hamiltonian_z_implicit, wrapping_direction = meshes.WrappingDirection.Z),
            DotOperator(hamiltonian_z_explicit, wrapping_direction = meshes.WrappingDirection.Z),
            TDMAOperator(hamiltonian_rho_implicit, wrapping_direction = meshes.WrappingDirection.RHO),
        ]

        return apply_operators(mesh, g, *operators)


class WarpedCylindricalSliceCrankNicolson(EvolutionMethod):
    pass


class SphericalSliceCrankNicolson(EvolutionMethod):
    def evolve(self, mesh: 'meshes.SphericalSliceMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
        tau = time_step / (2 * u.hbar)

        hamiltonian_r, hamiltonian_theta = mesh.get_internal_hamiltonian_matrix_operators()
        interaction_hamiltonian_r, interaction_hamiltonian_theta = mesh.get_interaction_hamiltonian_matrix_operators()

        hamiltonian_r = 1j * tau * add_to_diagonal_sparse_matrix_diagonal(hamiltonian_r, value = 0.5 * interaction_hamiltonian_r.diagonal())
        hamiltonian_theta = 1j * tau * add_to_diagonal_sparse_matrix_diagonal(hamiltonian_theta, value = 0.5 * interaction_hamiltonian_theta.diagonal())

        hamiltonian_theta_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_theta, value = 1)
        hamiltonian_r_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian_r, value = 1)
        hamiltonian_r_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_r, value = 1)
        hamiltonian_theta_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian_theta, value = 1)

        operators = [
            DotOperator(hamiltonian_theta_explicit, wrapping_direction = meshes.WrappingDirection.THETA),
            TDMAOperator(hamiltonian_r_implicit, wrapping_direction = meshes.WrappingDirection.R),
            DotOperator(hamiltonian_r_explicit, wrapping_direction = meshes.WrappingDirection.R),
            TDMAOperator(hamiltonian_theta_implicit, wrapping_direction = meshes.WrappingDirection.THETA),
        ]

        return apply_operators(mesh, g, *operators)


class SphericalHarmonicCrankNicolson(EvolutionMethod):
    def evolve(self, mesh: 'meshes.SphericalHarmonicMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
        tau = 1j * time_step / (2 * u.hbar)

        hamiltonian_r = tau * mesh.get_internal_hamiltonian_matrix_operators()
        hamiltonian_l = tau * mesh.get_interaction_hamiltonian_matrix_operators()

        hamiltonian_l_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_l, 1)
        hamiltonian_r_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian_r, 1)
        hamiltonian_r_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_r, 1)
        hamiltonian_l_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian_l, 1)

        operators = [
            DotOperator(hamiltonian_l_explicit, wrapping_direction = meshes.WrappingDirection.L),
            TDMAOperator(hamiltonian_r_implicit, wrapping_direction = meshes.WrappingDirection.R),
            DotOperator(hamiltonian_r_explicit, wrapping_direction = meshes.WrappingDirection.R),
            TDMAOperator(hamiltonian_l_implicit, wrapping_direction = meshes.WrappingDirection.L),
        ]

        return apply_operators(mesh, g, *operators)


class SphericalHarmonicSplitOperator(EvolutionMethod):
    def evolve(self, mesh: 'meshes.SphericalHarmonicMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
        tau = time_step / (2 * u.hbar)

        hamiltonian_r = (1j * tau) * mesh.get_internal_hamiltonian_matrix_operators()

        hamiltonian_r_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_r, 1)
        hamiltonian_r_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian_r, 1)

        split_operators = mesh._make_split_operator_evolution_operators(mesh.get_interaction_hamiltonian_matrix_operators(), tau)

        operators = (
            *split_operators,
            DotOperator(hamiltonian_r_explicit, wrapping_direction = meshes.WrappingDirection.R),
            TDMAOperator(hamiltonian_r_implicit, wrapping_direction = meshes.WrappingDirection.R),
            *reversed(split_operators),
        )

        return apply_operators(mesh, g, *operators)
