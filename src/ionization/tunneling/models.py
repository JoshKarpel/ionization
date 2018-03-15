import abc
import warnings
import functools

import numpy as np
import scipy.special as special
import scipy.integrate as integ

import simulacra as si
import simulacra.units as u

from .. import utils


class TunnelingModel(abc.ABC):
    """
    A class that implements a tunneling model based on the instantaneous amplitude of the electric field.

    NB: tunneling rates are often given in terms of the probability to remain in the bound state.
    The tunneling models here operate on the wavefunction itself, so generally they are half as large as reported in literature.
    """

    def __init__(self, upper_amplitude_cutoff: float = np.Inf):
        """
        Parameters
        ----------
        upper_amplitude_cutoff
            A model-independent upper amplitude cutoff in the tunneling rate.
        """
        self.upper_amplitude_cutoff = upper_amplitude_cutoff

    def tunneling_rate(self, electric_field_amplitude: float, ionization_potential: float) -> float:
        """
        Calculate the tunneling rate for this model from the instantaneous electric field amplitude and the bound state's ionization potential.

        Should generally not be overridden in subclasses.

        Parameters
        ----------
        electric_field_amplitude
            The instantaneous electric field amplitude.
        ionization_potential
            The ionization potential of the bound state being ionized.

        Returns
        -------
        tunneling_rate
            The tunneling rate at this instant.
        """
        rate = self._tunneling_rate(electric_field_amplitude = electric_field_amplitude, ionization_potential = ionization_potential)
        rate = np.where(
            np.abs(electric_field_amplitude) <= self.upper_amplitude_cutoff,
            rate,
            0,
        ).squeeze()

        return rate

    @abc.abstractmethod
    def _tunneling_rate(self, electric_field_amplitude, ionization_potential):
        """This method should be overridden in subclasses to implement the actual tunneling model."""
        raise NotImplementedError

    def info(self) -> si.Info:
        info = si.Info(header = self.__class__.__name__)

        info.add_field('Upper Electric Field Amplitude Cutoff', utils.fmt_quantity(self.upper_amplitude_cutoff, utils.ELECTRIC_FIELD_UNITS))

        return info


class NoTunneling(TunnelingModel):
    """A tunneling model with zero tunneling rate for any electric field amplitude."""

    def _tunneling_rate(self, electric_field_amplitude, ionization_potential):
        return 0


# the following rate calculations are from
# https://doi.org/10.1103/PhysRevA.59.569

class LandauRate(TunnelingModel):
    """
    A tunneling rate based on calculating the tunneling rate through a barrier in parabolic coordinates.

    Taken from https://doi.org/10.1103/PhysRevA.59.569 based on a derivation in Landau, L. D., & Lifshitz, E. M. (1977). Quantum Mechanics: Non-Relativistic Theory (3rd ed.). Pergamon Press.
    """

    def _tunneling_rate(self, electric_field_amplitude, ionization_potential):
        scaled_amplitude = np.abs(electric_field_amplitude) / u.atomic_electric_field
        scaled_potential = np.abs(ionization_potential) / u.hartree

        f = scaled_amplitude / ((2 * scaled_potential) ** 1.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # ignore div by zero errors

            rate = -.5 * (4 * (2 * scaled_potential) / f) * np.exp(-2 / (3 * f)) / u.atomic_time,

        rate = np.where(
            np.isclose(scaled_amplitude, 0),
            0,
            rate,
        )

        return rate


class KeldyshRate(TunnelingModel):
    """
    A tunneling rate based on calculating the tunneling rate through a barrier formed by an oscillating field.
    """

    def _tunneling_rate(self, electric_field_amplitude, ionization_potential):
        scaled_amplitude = np.abs(electric_field_amplitude) / u.atomic_electric_field
        scaled_potential = np.abs(ionization_potential) / u.hartree

        f = scaled_amplitude / ((2 * scaled_potential) ** 1.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # ignore div by zero errors

            pre = (np.sqrt(6 * u.pi) / (2 ** 1.25))
            rate = -.5 * pre * scaled_potential * np.sqrt(f) * np.exp(-2 / (3 * f)) / u.atomic_time

        rate = np.where(
            np.isclose(scaled_amplitude, 0),
            0,
            rate,
        )

        return rate


class PosthumusRate(TunnelingModel):
    """
    This is actually a BSI model, based on a semi-classical picture of the electron oscillating back and forth near a "hole" in the potential barrier.

    Taken from https://doi.org/10.1103/PhysRevA.59.569

    NB: current includes an amplitude cutoff which is only appropriate for hydrogen.
    """

    def __init__(self, atomic_number = 1, **kwargs):
        """
        Parameters
        ----------
        atomic_number
            The atomic number of the atom the electron is bound to (i.e., the number of protons, :math:`Z`).
        kwargs
            Any additional keyword arguments are passed to the :class:`TunnelingModel` constructor.
        """
        super().__init__(**kwargs)

        self.atomic_number = atomic_number

    def _tunneling_rate(self, electric_field_amplitude, ionization_potential):
        scaled_amplitude = np.abs(electric_field_amplitude) / u.atomic_electric_field
        scaled_potential = np.abs(ionization_potential) / u.hartree

        T = u.pi * self.atomic_number / (scaled_potential * np.sqrt(2 * scaled_potential))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # ignore div by zero errors

            rate = -.5 * (1 - ((scaled_potential ** 2) / (4 * self.atomic_number * scaled_amplitude))) / (2 * T) / u.atomic_time

        rate = np.where(
            np.less_equal(scaled_amplitude, u.atomic_electric_field / 16),
            0,
            rate,
        )

        return rate


class MulserRate(TunnelingModel):
    """
    A tunneling rate based on approximating the tunneling barrier as a parabola.

    Taken from https://doi.org/10.1103/PhysRevA.59.569
    """

    def _tunneling_rate(self, electric_field_amplitude, ionization_potential):
        scaled_amplitude = np.abs(electric_field_amplitude) / u.atomic_electric_field
        scaled_potential = np.abs(ionization_potential) / u.hartree

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # ignore div by zero errors

            C = -u.pi * ((2 * scaled_potential) ** .125) * (2 * scaled_potential) / (np.sqrt(2) * (scaled_amplitude ** .75))
            alpha = 4 * np.sqrt(scaled_amplitude) / ((2 * scaled_potential) ** .75)
            beta = (3 + alpha) * C / 4
            A = np.exp(-(7 - (3 * alpha)) * C / 4)

            rate = -.5 * (scaled_potential / np.abs(beta)) * np.log((A + np.exp(np.abs(beta))) / (A + 1)) / u.atomic_time

        rate = np.where(
            np.isclose(scaled_amplitude, 0),
            0,
            rate,
        )

        return rate


class ADKRate(TunnelingModel):
    """
    An extension of the Keldysh model to complex atoms.
    ADK stands for the derivers: Ammosov, Delone, and Krainov.

    Taken from https://doi.org/10.1103/PhysRevA.59.569

    NB: This rate is already averaged over one laser cycle, so it's not technically appropriate to use it in a simulation.
    """

    def __init__(self, n_star = 1, l = 0, m = 0, **kwargs):
        """
        Parameters
        ----------
        n_star
            Effective quantum number
        l
            Angular momentum quantum number
        m
            Magnetic quantum number
        kwargs
            Any additional keyword arguments are passed to the :class:`TunnelingModel` constructor.
        """
        super().__init__(**kwargs)

        self.n_star = n_star
        self.l = l
        self.m = m

    def _tunneling_rate(self, electric_field_amplitude, ionization_potential):
        scaled_amplitude = np.abs(electric_field_amplitude) / u.atomic_electric_field
        scaled_potential = np.abs(ionization_potential) / u.hartree

        f = scaled_amplitude / ((2 * scaled_potential) ** 1.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # ignore div by zero errors

            f_lm = ((2 * self.l) + 1) * special.factorial(self.l + np.abs(self.m)) / ((2 ** np.abs(self.m)) * special.factorial(np.abs(self.m)) * special.factorial(self.l - np.abs(self.m)))
            C_nl = ((2 * u.e / self.n_star) ** self.n_star) / np.sqrt(u.twopi * self.n_star)
            pre = (C_nl ** 2) * f_lm

            rate = -.5 * pre * scaled_potential * np.sqrt(3 * f / u.pi) * ((2 / f) ** (2 * self.n_star - np.abs(self.m) - 1)) * np.exp(-2 / (3 * f)) / u.atomic_time

        rate = np.where(
            np.isclose(scaled_amplitude, 0),
            0,
            rate,
        )

        return rate


class ADKExtendedToBSIRate(ADKRate):
    """
    An extension of the ADI theory to the barrier-suppression regime.

    Taken from https://doi.org/10.1103/PhysRevA.59.569
    """

    def __init__(self, n_star = 1, **kwargs):
        """
        Parameters
        ----------
        n_star
            Effective quantum number
        kwargs
            Any additional keyword arguments are passed to the :class:`TunnelingModel` constructor.
        """
        super().__init__(**kwargs)

        self.n_star = n_star

        raise NotImplementedError("Haven't quite figured out the right expression yet")

    def _tunneling_rate(self, electric_field_amplitude, ionization_potential):
        scaled_amplitude = np.abs(electric_field_amplitude) / u.atomic_electric_field
        scaled_potential = np.abs(ionization_potential) / u.hartree

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # ignore div by zero errors

            pre = 4 * np.sqrt(3) / (u.pi * self.n_star)
            factor_1 = scaled_amplitude / ((2 * scaled_amplitude) ** (1 / 3))
            factor_2 = (4 * u.e * (scaled_potential ** 1.5) / (scaled_amplitude * self.n_star)) ** (2 * self.n_star)

            def integrand(x, amplitude = 0):
                ai, aip, bi, bip = special.airy((x ** 2) + ((2 * scaled_potential) / ((2 * amplitude) ** 1.5)))
                return (x * ai) ** 2

            try:
                integral, *_ = integ.quad(functools.partial(integrand, amplitude = scaled_amplitude), 0, np.inf)
            except TypeError:
                integral = np.array([integ.quad(functools.partial(integrand, amplitude = a), 0, np.inf)[0] for a in scaled_amplitude])

            rate = -.5 * pre * factor_1 * factor_2 * integral / u.atomic_time

        rate = np.where(
            np.isclose(scaled_amplitude, 0),
            0,
            rate,
        )

        return rate


TUNNELING_MODEL_TYPES = [
    NoTunneling,
    LandauRate,
    KeldyshRate,
    PosthumusRate,
    MulserRate,
    ADKRate,
]
