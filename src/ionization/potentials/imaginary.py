import numpy as np

import simulacra.units as u

from . import potentials


class ImaginaryGaussianRing(potentials.PotentialEnergy):
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
        return '{}(center = {} a_0, width = {} a_0, decay time = {} as)'.format(self.__class__.__name__,
                                                                                u.uround(self.center, u.bohr_radius, 3),
                                                                                u.uround(self.width, u.bohr_radius, 3),
                                                                                u.uround(self.decay_time, u.asec))

    def __call__(self, *, r, **kwargs):
        rel = r - self.center
        return self.prefactor * np.exp(-((rel / self.width) ** 2))
