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

from . import meshes, operators


class EvolutionMethod(abc.ABC):
    def evolve(self, mesh: 'meshes.QuantumMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
        evolution_operators = self.get_evolution_operators(mesh, time_step)
        return operators.apply_operators(mesh, g, evolution_operators)

    @abc.abstractmethod
    def get_evolution_operators(self, mesh: 'meshes.QuantumMesh', time_step: complex) -> Iterable[operators.MeshOperator]:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def info(self) -> si.Info:
        info = si.Info(header = self.__class__.__name__)

        return info


class LineCrankNicolson(EvolutionMethod):
    def get_evolution_operators(self, mesh: 'meshes.LineMesh', time_step: complex) -> Iterable[operators.MeshOperator]:
        tau = time_step / (2 * u.hbar)

        interaction_operator = mesh.get_interaction_hamiltonian_matrix_operators()

        hamiltonian_x = mesh.get_internal_hamiltonian_matrix_operators()
        hamiltonian = 1j * tau * sparse.dia_matrix(hamiltonian_x + interaction_operator)

        ham_explicit = operators.add_to_diagonal_sparse_matrix_diagonal(-hamiltonian, 1)
        ham_implicit = operators.add_to_diagonal_sparse_matrix_diagonal(hamiltonian, 1)

        evolution_operators = [
            operators.DotOperator(ham_explicit, wrapping_direction = None),
            operators.TDMAOperator(ham_implicit, wrapping_direction = None),
        ]

        return evolution_operators


class LineSplitOperator(EvolutionMethod):
    def get_evolution_operators(self, mesh: 'meshes.LineMesh', time_step: complex) -> Iterable[operators.MeshOperator]:
        tau = time_step / (2 * u.hbar)

        hamiltonian_x = mesh.get_internal_hamiltonian_matrix_operators()

        ham_x_explicit = operators.add_to_diagonal_sparse_matrix_diagonal(-1j * tau * hamiltonian_x, 1)
        ham_x_implicit = operators.add_to_diagonal_sparse_matrix_diagonal(1j * tau * hamiltonian_x, 1)

        split_operators = mesh._make_split_operator_evolution_operators(mesh.get_interaction_hamiltonian_matrix_operators(), tau)

        evolution_operators = [
            *split_operators,
            operators.DotOperator(ham_x_explicit, wrapping_direction = None),
            operators.TDMAOperator(ham_x_implicit, wrapping_direction = None),
            *reversed(split_operators),
        ]

        return evolution_operators


# class LineSpectral(EvolutionMethod):
#     def evolve(self, mesh: 'meshes.LineMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
#         g = self._evolve_potential(mesh, g, time_step / 2)
#         g = self._evolve_free(mesh, g, time_step)  # splitting order chosen for computational efficiency (only one FFT per time step)
#         g = self._evolve_potential(mesh, g, time_step / 2)
#
#         return g
#
#     def _evolve_potential(self, mesh: 'meshes.LineMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
#         pot = mesh.spec.internal_potential(t = mesh.sim.time, r = mesh.x_mesh, distance = mesh.x_mesh, test_charge = mesh.spec.test_charge)
#         pot += mesh.spec.electric_potential(t = mesh.sim.time, r = mesh.x_mesh, distance = mesh.x_mesh, distance_along_polarization = mesh.x_mesh, test_charge = mesh.spec.test_charge)
#
#         return g * np.exp(-1j * time_step * pot / u.hbar)
#
#     def _evolve_free(self, mesh: 'meshes.LineMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
#         return mesh.ifft(mesh.fft(g) * np.exp(mesh.free_evolution_prefactor * time_step) * mesh.wavenumber_mask)


class AlternatingDirectionImplicitCrankNicolson(EvolutionMethod):
    def get_evolution_operators(self, mesh: 'meshes.QuantumMesh', time_step: complex) -> Iterable[operators.MeshOperator]:
        tau = time_step / (2 * u.hbar)

        ham_opers = mesh.operators.total_hamiltonian(mesh)

        evolution_operators = []
        explicit = True
        for oper in itertools.chain(ham_opers, reversed(ham_opers)):
            if explicit:
                e_oper = operators.DotOperator(
                    operators.add_to_diagonal_sparse_matrix_diagonal(-1j * tau * oper.matrix, value = 1),
                    wrapping_direction = oper.wrapping_direction,
                )
            else:
                e_oper = operators.TDMAOperator(
                    operators.add_to_diagonal_sparse_matrix_diagonal(1j * tau * oper.matrix, value = 1),
                    wrapping_direction = oper.wrapping_direction,
                )

            evolution_operators.append(e_oper)
            explicit = not explicit

        return tuple(evolution_operators)


class SphericalHarmonicSplitOperator(EvolutionMethod):
    def get_evolution_operators(self, mesh: 'meshes.SphericalHarmonicMesh', time_step: complex) -> 'meshes.GMesh':
        tau = time_step / (2 * u.hbar)

        hamiltonian_r = (1j * tau) * mesh.get_internal_hamiltonian_matrix_operators()

        hamiltonian_r_explicit = operators.add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_r, 1)
        hamiltonian_r_implicit = operators.add_to_diagonal_sparse_matrix_diagonal(hamiltonian_r, 1)

        split_operators = mesh._make_split_operator_evolution_operators(mesh.get_interaction_hamiltonian_matrix_operators(), tau)

        evolution_operators = (
            *split_operators,
            operators.DotOperator(hamiltonian_r_explicit, wrapping_direction = meshes.WrappingDirection.R),
            operators.TDMAOperator(hamiltonian_r_implicit, wrapping_direction = meshes.WrappingDirection.R),
            *reversed(split_operators),
        )

        return evolution_operators
