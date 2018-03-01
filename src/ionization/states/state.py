import logging
from copy import deepcopy

import numpy as np
import simulacra as si

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QuantumState(si.summables.Summand):
    """A class that represents a quantum state, with an amplitude and some basic multiplication/addition rules. Can be summed to form a Superposition."""

    bound = None
    discrete_eigenvalues = None
    analytic = None

    def __init__(self, amplitude = 1):
        """
        Construct a QuantumState with a given amplitude.

        QuantumStates should not be instantiated directly (they have no useful properties).

        :param amplitude: the probability amplitude of the state
        """
        super().__init__()
        self.amplitude = amplitude
        self.summation_class = Superposition

    @property
    def numeric(self):
        return not self.analytic

    @property
    def free(self):
        return not self.bound

    @property
    def continuous_eigenvalues(self):
        return not self.discrete_eigenvalues

    @property
    def norm(self):
        return np.abs(self.amplitude) ** 2

    def __mul__(self, other):
        new = deepcopy(self)
        new.amplitude *= other
        return new

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1 / other)

    @property
    def tuple(self):
        """This property should return a tuple of unique information about the state, which will be used to hash itertools or perform comparison operations."""
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

    @property
    def latex(self):
        """Return a string in TeX notation that should be placed inside bras or kets in output."""
        return r'\psi'

    def __call__(self, *args, **kwargs):
        return 0


class Superposition(si.summables.Sum, QuantumState):
    """A class that represents a superposition of bound states."""

    container_name = 'states'

    def __init__(self, *states):
        """
        Construct a discrete superposition of states.

        :param states: any number of QuantumStates
        """
        super().__init__(amplitude = 1)
        norm = np.sqrt(sum(s.norm for s in states))
        self.states = list(s / norm for s in states)  # note that the states are implicitly copied here

    @property
    def tuple(self):
        return sum((s.tuple for s in self.states), tuple())
