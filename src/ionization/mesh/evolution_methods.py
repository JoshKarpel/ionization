import itertools
import logging
from typing import Iterable
import abc

import simulacra as si
import simulacra.units as u

from . import meshes, mesh_operators

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EvolutionMethod(abc.ABC):
    """An abstract class that represents an evolution method for a wavefunction on a mesh."""

    def evolve(self, mesh: 'meshes.QuantumMesh', g: 'meshes.GMesh', time_step: complex) -> 'meshes.GMesh':
        """Evolve ``g`` in time by ``time_step``."""
        evolution_operators = self.get_evolution_operators(mesh, time_step)
        return mesh_operators.apply_operators_sequentially(mesh, g, evolution_operators)

    @abc.abstractmethod
    def get_evolution_operators(self, mesh: 'meshes.QuantumMesh', time_step: complex) -> Iterable[mesh_operators.MeshOperator]:
        """
        Return a sequence of operators that collectively evolve a ``g`` mesh in time by ``time_step``.

        This is an abstract method that should be overridden in concrete subclasses to implement the desired evolution algorithm.
        """
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def info(self) -> si.Info:
        info = si.Info(header = self.__class__.__name__)

        return info


class AlternatingDirectionImplicit(EvolutionMethod):
    """This is the two-dimensional Crank-Nicolson-style Alternating Direction Implicit method for solving PDEs."""

    def get_evolution_operators(self, mesh: 'meshes.QuantumMesh', time_step: complex) -> Iterable[mesh_operators.MeshOperator]:
        tau = time_step / (2 * u.hbar)

        ham_opers = mesh.operators.total_hamiltonian(mesh).operators

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


class SplitInteractionOperator(EvolutionMethod):
    """
    This is the split-operator method.

    This implementation only works when the field-free Hamiltonian is given by a single matrix operator.
    Whichever operator in the internal Hamiltonian is the first in the sum will be used.
    That means that, for the moment, this method only works with :class:`LineMesh` and :class:`SphericalHarmonicMesh` (with ``r`` as the non-interacting dimension).
    """

    def get_evolution_operators(self, mesh: 'meshes.SphericalHarmonicMesh', time_step: complex) -> 'meshes.GMesh':
        tau = time_step / (2 * u.hbar)

        cn_oper, = mesh.operators.internal_hamiltonian(mesh).operators  # this is the operator we do Crank-Nicolson ADI with

        evolution_operators = [
            mesh_operators.DotOperator(mesh_operators.add_to_diagonal_sparse_matrix_diagonal((-1j * tau) * cn_oper.matrix, value = 1), wrapping_direction = cn_oper.wrapping_direction),
            mesh_operators.TDMAOperator(mesh_operators.add_to_diagonal_sparse_matrix_diagonal((1j * tau) * cn_oper.matrix, value = 1), wrapping_direction = cn_oper.wrapping_direction),
        ]

        split_operators = mesh.operators.split_interaction_operators(mesh, mesh.operators.interaction_hamiltonian(mesh), tau)

        evolution_operators = (
            *split_operators,
            *evolution_operators,
            *reversed(split_operators),
        )

        return evolution_operators
