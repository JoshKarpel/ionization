import abc
import warnings
import functools

import numpy as np
import scipy.special as special
import scipy.integrate as integ

import simulacra as si
import simulacra.units as u


class TunnelingModel(abc.ABC):
    def tunneling_rate(self, sim, electric_potential, time):
        amplitude = electric_potential.get_electric_field_amplitude(time)

        rate = self.tunneling_rate_from_amplitude(electric_field_amplitude = amplitude, ionization_potential = sim.spec.ionization_potential)

        return rate

    @abc.abstractmethod
    def tunneling_rate_from_amplitude(self, electric_field_amplitude, ionization_potential):
        raise NotImplementedError

    def info(self) -> si.Info:
        info = si.Info(header = self.__class__.__name__)

        return info


# the following rate calculations are from
# https://doi.org/10.1103/PhysRevA.59.569

class LandauRate(TunnelingModel):
    def tunneling_rate_from_amplitude(self, electric_field_amplitude, ionization_potential):
        scaled_amplitude = np.abs(electric_field_amplitude) / u.atomic_electric_field
        scaled_potential = np.abs(ionization_potential) / u.hartree

        f = scaled_amplitude / ((2 * scaled_potential) ** 1.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # ignore div by zero errors

            rate = -(4 / f) * np.exp(-2 / (3 * f)) / u.atomic_time,

        rate = np.where(
            np.isclose(scaled_amplitude, 0),
            0,
            rate,
        )

        return rate


class KeldyshRate(TunnelingModel):
    def tunneling_rate_from_amplitude(self, electric_field_amplitude, ionization_potential):
        scaled_amplitude = np.abs(electric_field_amplitude) / u.atomic_electric_field
        scaled_potential = np.abs(ionization_potential) / u.hartree

        f = scaled_amplitude / ((2 * scaled_potential) ** 1.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # ignore div by zero errors

            pre = (np.sqrt(6 * u.pi) / (2 ** 1.25))
            rate = pre * scaled_potential * np.sqrt(f) * np.exp(-2 / (3 * f)) / u.atomic_time

        rate = np.where(
            np.isclose(scaled_amplitude, 0),
            0,
            rate,
        )

        return rate


class PosthumusRate(TunnelingModel):
    def __init__(self, atomic_number = 1):
        super().__init__()

        self.atomic_number = atomic_number

        warnings.warn(f"{self.__class__.__name__} uses a lower cutoff in the electric field amplitude that I haven't full figured out yet")

    def tunneling_rate_from_amplitude(self, electric_field_amplitude, ionization_potential):
        scaled_amplitude = np.abs(electric_field_amplitude) / u.atomic_electric_field
        scaled_potential = np.abs(ionization_potential) / u.hartree

        T = u.pi * self.atomic_number / (scaled_potential * np.sqrt(2 * scaled_potential))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # ignore div by zero errors

            rate = (1 - ((scaled_potential ** 2) / (4 * self.atomic_number * scaled_amplitude))) / (2 * T) / u.atomic_time

        rate = np.where(
            np.less_equal(scaled_amplitude, 0.0625),
            0,
            rate,
        )

        return rate


class MulserRate(TunnelingModel):
    def tunneling_rate_from_amplitude(self, electric_field_amplitude, ionization_potential):
        scaled_amplitude = np.abs(electric_field_amplitude) / u.atomic_electric_field
        scaled_potential = np.abs(ionization_potential) / u.hartree

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # ignore div by zero errors

            C = -u.pi * ((2 * scaled_potential) ** .125) * (2 * scaled_potential) / (np.sqrt(2) * (scaled_amplitude ** .75))
            alpha = 4 * np.sqrt(scaled_amplitude) / ((2 * scaled_potential) ** .75)
            beta = (3 + alpha) * C / 4
            A = np.exp(-(7 - (3 * alpha)) * C / 4)

            rate = (scaled_potential / np.abs(beta)) * np.log((A + np.exp(np.abs(beta))) / (A + 1)) / u.atomic_time

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

    def __init__(self, n_star = 1, l = 0, m = 0):
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
        super().__init__()

        self.n_star = n_star
        self.l = l
        self.m = m

    def tunneling_rate_from_amplitude(self, electric_field_amplitude, ionization_potential):
        scaled_amplitude = np.abs(electric_field_amplitude) / u.atomic_electric_field
        scaled_potential = np.abs(ionization_potential) / u.hartree

        f = scaled_amplitude / ((2 * scaled_potential) ** 1.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # ignore div by zero errors

            f_lm = ((2 * self.l) + 1) * special.factorial(self.l + np.abs(self.m)) / ((2 ** np.abs(self.m)) * special.factorial(np.abs(self.m)) * special.factorial(self.l - np.abs(self.m)))
            C_nl = ((2 * u.e / self.n_star) ** self.n_star) / np.sqrt(u.twopi * self.n_star)
            pre = (C_nl ** 2) * f_lm

            rate = pre * scaled_potential * np.sqrt(3 * f / u.pi) * ((2 / f) ** (2 * self.n_star - np.abs(self.m) - 1)) * np.exp(-2 / (3 * f)) / u.atomic_time

        rate = np.where(
            np.isclose(scaled_amplitude, 0),
            0,
            rate,
        )

        return rate


class ADKExtendedToBSIRate(ADKRate):
    def __init__(self, n_star = 1):
        """
        Parameters
        ----------
        n_star
            Effective quantum number
        """
        super().__init__()

        self.n_star = n_star

        raise NotImplementedError("Haven't quite figured out the right expression yet")

    def tunneling_rate_from_amplitude(self, electric_field_amplitude, ionization_potential):
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

            rate = pre * factor_1 * factor_2 * integral / u.atomic_time

        rate = np.where(
            np.isclose(scaled_amplitude, 0),
            0,
            rate,
        )

        return rate
