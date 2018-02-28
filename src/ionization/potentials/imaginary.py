import numpy as np

import simulacra.units as u

from .. import utils
from . import potential


class ImaginaryGaussianRing(potential.PotentialEnergy):
    def __init__(self, center = 20 * u.bohr_radius, width = 2 * u.bohr_radius, decay_time = 100 * u.asec):
        """
        Construct a RadialImaginary potential. The potential is shaped like a Gaussian wrapped around a ring and has an imaginary amplitude.

        A positive/negative amplitude yields an imaginary potential that causes decay/amplification.

        Parameters
        ----------
        center
            The radial coordinate of the center of the potential.
        width
            The width (FWHM) of the Gaussian.
        decay_time
            The decay time (1/e time) of a wavefunction packet at the peak of the imaginary potential.
        """
        self.center = center
        self.width = width
        self.decay_time = decay_time
        self.decay_rate = 1 / decay_time

        self.prefactor = -1j * self.decay_rate * u.hbar

        super().__init__()

    def __repr__(self):
        return f'{self.__class__.__name__}(center = {self.center}, width = {self.width}, decay_time = {self.decay_time})'

    def __str__(self):
        return f'{self.__class__.__name__}(center = {u.uround(self.center, u.bohr_radius)} a_0, width = {u.uround(self.width, u.bohr_radius)} a_0, decay_time = {u.uround(self.decay_time, u.asec)} as)'

    def info(self):
        info = super().info()

        info.add_field('Center', utils.fmt_quantity(self.center, utils.LENGTH_UNITS))
        info.add_field('Width', utils.fmt_quantity(self.width, utils.LENGTH_UNITS))
        info.add_field('Decay Time', utils.fmt_quantity(self.decay_time, utils.TIME_UNITS))

        return info

    def __call__(self, *, r, **kwargs):
        rel = r - self.center
        return self.prefactor * np.exp(-((rel / self.width) ** 2))
