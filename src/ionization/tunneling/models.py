import abc
import warnings

import numpy as np

import simulacra as si
import simulacra.units as u


class TunnelingModel(abc.ABC):
    @abc.abstractmethod
    def tunneling_rate(self, sim, electric_potential, time):
        raise NotImplementedError

    def info(self) -> si.Info:
        info = si.Info(header = self.__class__.__name__)

        return info


class LandauRate(TunnelingModel):
    def tunneling_rate(self, sim, electric_potential, time):
        amplitude = electric_potential.get_electric_field_amplitude(time)

        rate = self.tunneling_rate_from_amplitude(electric_field_amplitude = amplitude, ionization_potential = sim.spec.ionization_potential)

        return rate

    def tunneling_rate_from_amplitude(self, electric_field_amplitude, ionization_potential):
        amplitude_scaled = np.abs(electric_field_amplitude) / u.atomic_electric_field
        potential_scaled = np.abs(ionization_potential) / u.hartree

        f = amplitude_scaled / ((2 * potential_scaled) ** 1.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # ignore div by zero errors
            rate = -(4 / f) * np.exp(-2 / (3 * f)) / u.atomic_time,

        rate = np.where(
            np.isclose(amplitude_scaled, 0),
            0,
            rate,
        )

        return rate
