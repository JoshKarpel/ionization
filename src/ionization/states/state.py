import logging
import abc
import collections
from copy import deepcopy
from typing import NewType, Tuple

import numpy as np
import simulacra as si

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Eigenvalues(si.utils.StrEnum):
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'


class Binding(si.utils.StrEnum):
    BOUND = 'bound'
    FREE = 'free'


class Derivation(si.utils.StrEnum):
    ANALYTIC = 'analytic'
    NUMERIC = 'numeric'


ProbabilityAmplitude = NewType('ProbabilityAmplitude', complex)
Probability = NewType('Probability', float)


class QuantumState(si.summables.Summand, abc.ABC):
    """
    A class that represents a quantum state, with an amplitude and some basic multiplication/addition rules.
    Can be summed to form a Superposition.

    Currently, there is no support for product states of multiple particles.
    """

    eigenvalues = None
    binding = None
    derivation = None

    def __init__(self, amplitude: ProbabilityAmplitude = 1):
        """

        Parameters
        ----------
        amplitude
            The probability amplitude of this state.
        """
        super().__init__()
        self.amplitude = amplitude
        self.summation_class = Superposition

    @property
    def numeric(self) -> bool:
        return self.derivation == Derivation.NUMERIC

    @property
    def analytic(self) -> bool:
        return self.derivation == Derivation.ANALYTIC

    @property
    def bound(self) -> bool:
        return self.binding == Binding.BOUND

    @property
    def free(self) -> bool:
        return self.binding == Binding.FREE

    @property
    def discrete_eigenvalues(self) -> bool:
        return self.eigenvalues == Eigenvalues.DISCRETE

    @property
    def continuous_eigenvalues(self) -> bool:
        return self.eigenvalues == Eigenvalues.CONTINUOUS

    @property
    def norm(self) -> Probability:
        return np.abs(self.amplitude) ** 2

    def normalized(self) -> 'QuantumState':
        return self / np.sqrt(self.norm)

    def __mul__(self, other: ProbabilityAmplitude) -> 'QuantumState':
        new = deepcopy(self)
        new.amplitude *= other
        return new

    def __rmul__(self, other: ProbabilityAmplitude) -> 'QuantumState':
        return self * other

    def __truediv__(self, other: ProbabilityAmplitude) -> 'QuantumState':
        return self * (1 / other)

    @property
    @abc.abstractmethod
    def tuple(self) -> Tuple:
        """This property should return a tuple of unique information about the state, which will be used to hash it and perform comparison operations."""
        raise NotImplementedError

    def __hash__(self):
        return hash((self.__class__.__name__,) + self.tuple)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.tuple == other.tuple

    def __lt__(self, other):
        return isinstance(other, self.__class__) and self.tuple < other.tuple

    def __gt__(self, other):
        return isinstance(other, self.__class__) and self.tuple > other.tuple

    def __le__(self, other):
        return isinstance(other, self.__class__) and self.tuple <= other.tuple

    def __ge__(self, other):
        return isinstance(other, self.__class__) and self.tuple >= other.tuple

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.ket

    @property
    @abc.abstractmethod
    def ket(self) -> str:
        raise NotImplementedError

    @property
    def bra(self) -> str:
        """Gets the bra representation of the HydrogenBoundState"""
        return self.ket.replace('|', '<').replace('>', '|')

    @property
    def latex(self) -> si.units.TeXString:
        """Return a string in TeX notation that should be placed inside bras or kets in output."""
        raise NotImplementedError

    @property
    def latex_ket(self) -> si.units.TeXString:
        raise NotImplementedError

    @property
    def latex_bra(self) -> si.units.TeXString:
        return self.latex_ket.replace(r'\left|', r'\left\langle').replace(r'\right\rangle', r'\right|')

    def info(self) -> si.Info:
        info = si.Info(header = f'{self.__class__.__name__}(amplitude = {np.around(self.amplitude, 3)}, norm = {np.around(self.norm, 3)})')

        return info


class Superposition(si.summables.Sum, QuantumState):
    """A class that represents a superposition of bound states."""

    container_name = 'states'

    def __init__(self, *states: QuantumState):
        """
        Construct a discrete superposition of states.

        :param states: any number of QuantumStates
        """
        states_to_amplitudes = collections.defaultdict(float)

        for state in states:
            states_to_amplitudes[state] += state.amplitude

        combined_states = []
        for state, amplitude in states_to_amplitudes.items():
            s = deepcopy(state)
            s.amplitude = amplitude
            combined_states.append(deepcopy(s))

        combined_states = tuple(combined_states)

        norm = sum(s.norm for s in combined_states)
        super().__init__(amplitude = np.sqrt(norm))

        self.states = combined_states

    @property
    def tuple(self):
        return sum((s.tuple for s in self), tuple())

    @property
    def ket(self) -> str:
        return ' + '.join(s.ket for s in self)

    @property
    def latex(self) -> si.units.TeXString:
        return ' + '.join(s.latex for s in self)

    @property
    def latex_ket(self) -> si.units.TeXString:
        return ' + '.join(s.latex_ket for s in self)

    def normalized(self) -> 'Superposition':
        return Superposition(*tuple(s / np.sqrt(self.norm) for s in self))

    def info(self) -> si.Info:
        info = super().info()

        for s in self:
            info.add_info(s.info())

        return info
