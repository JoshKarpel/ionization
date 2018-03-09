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
    IMPORTANT: tunneling rates are often given in terms of the probability to remain in the bound state.
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

    def tunneling_rate(self, electric_field_amplitude, ionization_potential):
        rate = self._tunneling_rate(electric_field_amplitude = electric_field_amplitude, ionization_potential = ionization_potential)
        rate = np.where(
            np.abs(electric_field_amplitude) <= self.upper_amplitude_cutoff,
            rate,
            0,
        ).squeeze()

        return rate

    @abc.abstractmethod
    def _tunneling_rate(self, electric_field_amplitude, ionization_potential):
        raise NotImplementedError

    def info(self) -> si.Info:
        info = si.Info(header = self.__class__.__name__)

        info.add_field('Upper Electric Field Amplitude Cutoff', utils.fmt_quantity(self.upper_amplitude_cutoff, utils.ELECTRIC_FIELD_UNITS))

        return info


class NoTunneling(TunnelingModel):
    def __init__(self):
        super().__init__()

    def _tunneling_rate(self, electric_field_amplitude, ionization_potential):
        return 0


# the following rate calculations are from
# https://doi.org/10.1103/PhysRevA.59.569

class LandauRate(TunnelingModel):
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
    This is actually a BSI model.

    NB: current includes an amplitude cutoff which is only appropriate for hydrogen.
    """

    def __init__(self, atomic_number = 1, **kwargs):
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
    This rate is average over one laser cycle.
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
    def __init__(self, n_star = 1, **kwargs):
        """
        Parameters
        ----------
        n_star
            Effective quantum number
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
