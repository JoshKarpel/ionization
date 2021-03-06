import logging
import abc
import collections
from copy import deepcopy
from typing import NewType, Tuple, Optional, Union

import numpy as np
import simulacra as si

from .. import summables

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

QuantumNumber = NewType("QuantumNumber", int)
ProbabilityAmplitude = NewType("ProbabilityAmplitude", Union[float, complex])
Probability = NewType("Probability", float)


class Eigenvalues(si.utils.StrEnum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class Binding(si.utils.StrEnum):
    BOUND = "bound"
    FREE = "free"


class Derivation(si.utils.StrEnum):
    ANALYTIC = "analytic"
    NUMERIC = "numeric"
    VARIATIONAL = "variational"


def fmt_amplitude(amplitude: ProbabilityAmplitude) -> str:
    if amplitude == 1:
        return ""
    else:
        return f"{amplitude:.3f}"


def fmt_amplitude_for_tex(amplitude: ProbabilityAmplitude) -> str:
    out = fmt_amplitude(amplitude)

    if out == "":
        out += r" \, "

    return out


class QuantumState(summables.Summand, abc.ABC):
    """
    A class that represents a quantum state, with an amplitude and some basic multiplication/addition rules.
    Can be summed to form a Superposition.

    Attributes
    ----------
    numeric: :class:`bool`
        ``True`` if the state was constructed numerically.
    analytic: :class:`bool`
        ``True`` if the state was constructed analytically.
    variational: :class:`bool`
        ``True`` if the state was constructed variationally.
    bound: :class:`bool`
        ``True`` if the state is a bound (i.e., not free) state.
    free: :class:`bool`
        ``True`` if the state is a free (i.e., not bound) state.
    discrete_eigenvalues: :class:`bool`
        ``True`` if the state has fully discrete eigenvalues.
    continuous_eigenvalues: :class:`bool`
        ``True`` if the state has any continuous eigenvalues.

    amplitude: ProbabilityAmplitude
        The probability amplitude of the state (i.e., expansion coefficient in this basis).
    norm: Probability
        The norm of the state.

    ket: :class:`str`
        An ASCII representation of the state as a ket.
    bra: :class:`str`
        An ASCII representation of the state as a bra.
    tex: :class:`str`
        A TeX-formatted string of the state.
    tex_ket: :class:`str`
        A TeX-formatted string of the state as a ket.
    tex_bra: :class:`str`
        A TeX-formatted string of the state as a bra.
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
    def variational(self) -> bool:
        return self.derivation == Derivation.VARIATIONAL

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

    def normalized(self) -> "QuantumState":
        """Return a normalized version of the state."""
        return self / np.sqrt(self.norm)

    def __mul__(self, other: ProbabilityAmplitude) -> "QuantumState":
        new = deepcopy(self)
        new.amplitude *= other
        return new

    def __rmul__(self, other: ProbabilityAmplitude) -> "QuantumState":
        return self * other

    def __truediv__(self, other: ProbabilityAmplitude) -> "QuantumState":
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

    @abc.abstractmethod
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
        return self.ket.replace("|", "<").replace(">", "|")

    @property
    def tex(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tex_ket(self) -> str:
        raise NotImplementedError

    @property
    def tex_bra(self) -> str:
        return self.tex_ket.replace(r"\left|", r"\left\langle").replace(
            r"\right\rangle", r"\right|"
        )

    def info(self) -> si.Info:
        info = si.Info(
            header=f"{self.__class__.__name__}(amplitude = {np.around(self.amplitude, 3)}, norm = {np.around(self.norm, 3)})"
        )

        return info


class Superposition(summables.Sum, QuantumState):
    """
    A class that represents a discrete superposition of states.

    Although a :class:`Superposition` of states with continuous eigenvalues can be created, such states are non-physical and may have unexpected properties.
    """

    container_name = "states"

    def __init__(self, *states: QuantumState):
        """
        Parameters
        ----------
        states
             Any number of :class:`QuantumState`
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
        super().__init__(amplitude=np.sqrt(norm))

        self.states = combined_states

    @property
    def tuple(self):
        return sum((s.tuple for s in self), tuple())

    @property
    def ket(self) -> str:
        return " + ".join(s.ket for s in self)

    @property
    def tex(self) -> str:
        return " + ".join(s.tex for s in self)

    @property
    def tex_ket(self) -> str:
        return " + ".join(s.tex_ket for s in self)

    def normalized(self) -> "Superposition":
        """Return a normalized version of the state."""
        return Superposition(*tuple(s / np.sqrt(self.norm) for s in self))

    def info(self) -> si.Info:
        info = super().info()

        for s in self:
            info.add_info(s.info())

        return info


def fmt_inner_product_for_tex(a: QuantumState, b: QuantumState, op: str = "") -> str:
    """

    Parameters
    ----------
    a
    b
    op

    Returns
    -------

    """
    return fr"\left\langle {a.tex} \right| {op} \left| {b.tex} \right\rangle"
