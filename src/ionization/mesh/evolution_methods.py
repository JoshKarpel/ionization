import itertools
import logging
from typing import Union, Optional, Iterable, NewType, Tuple, Dict
import abc

import simulacra as si
import simulacra.units as u

from . import meshes, mesh_operators

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EvolutionMethod(abc.ABC):
    def evolve(self, mesh: 'meshes.QuantumMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
        evolution_operators = self.get_evolution_operators(mesh, time_step)
        return mesh_operators.apply_operators(mesh, g, evolution_operators)

    @abc.abstractmethod
    def get_evolution_operators(self, mesh: 'meshes.QuantumMesh', time_step: complex) -> Iterable[mesh_operators.MeshOperator]:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def info(self) -> si.Info:
        info = si.Info(header = self.__class__.__name__)

        return info


class LineSplitOperator(EvolutionMethod):
    def get_evolution_operators(self, mesh: 'meshes.SphericalHarmonicMesh', time_step: complex) -> 'meshes.GMesh':
        tau = time_step / (2 * u.hbar)

        x_oper, = mesh.operators.internal_hamiltonian(mesh)
        interaction_operators = mesh.operators.interaction_hamiltonian(mesh)

        evolution_operators = [
            mesh_operators.DotOperator(mesh_operators.add_to_diagonal_sparse_matrix_diagonal(-1j * tau * x_oper.matrix, value = 1), wrapping_direction = x_oper.wrapping_direction),
            mesh_operators.TDMAOperator(mesh_operators.add_to_diagonal_sparse_matrix_diagonal(1j * tau * x_oper.matrix, value = 1), wrapping_direction = x_oper.wrapping_direction),
        ]

        split_operators = mesh.operators.split_interaction_operators(mesh, interaction_operators, tau)

        evolution_operators = (
            *split_operators,
            *evolution_operators,
            *reversed(split_operators),
        )

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
    def get_evolution_operators(self, mesh: 'meshes.QuantumMesh', time_step: complex) -> Iterable[mesh_operators.MeshOperator]:
        tau = time_step / (2 * u.hbar)

        ham_opers = mesh.operators.total_hamiltonian(mesh)[0].operators  # TODO: ack

        evolution_operators = []
        explicit = True
        for oper in itertools.chain(ham_opers, reversed(ham_opers)):
            if explicit:
                e_oper = mesh_operators.DotOperator(
                    mesh_operators.add_to_diagonal_sparse_matrix_diagonal(-1j * tau * oper.matrix, value = 1),
                    wrapping_direction = oper.wrapping_direction,
                )
            else:
                e_oper = mesh_operators.TDMAOperator(
                    mesh_operators.add_to_diagonal_sparse_matrix_diagonal(1j * tau * oper.matrix, value = 1),
                    wrapping_direction = oper.wrapping_direction,
                )

            evolution_operators.append(e_oper)
            explicit = not explicit

        return tuple(evolution_operators)


class SphericalHarmonicSplitOperator(EvolutionMethod):
    def get_evolution_operators(self, mesh: 'meshes.SphericalHarmonicMesh', time_step: complex) -> 'meshes.GMesh':
        tau = time_step / (2 * u.hbar)

        r_oper, = mesh.operators.internal_hamiltonian(mesh)
        interaction_operators = mesh.operators.interaction_hamiltonian(mesh)

        evolution_operators = [
            mesh_operators.DotOperator(mesh_operators.add_to_diagonal_sparse_matrix_diagonal(-1j * tau * r_oper.matrix, value = 1), wrapping_direction = r_oper.wrapping_direction),
            mesh_operators.TDMAOperator(mesh_operators.add_to_diagonal_sparse_matrix_diagonal(1j * tau * r_oper.matrix, value = 1), wrapping_direction = r_oper.wrapping_direction),
        ]

        split_operators = mesh.operators.split_interaction_operators(mesh, interaction_operators, tau)

        evolution_operators = (
            *split_operators,
            *evolution_operators,
            *reversed(split_operators),
        )

        return evolution_operators
