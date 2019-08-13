import numpy as np

import simulacra as si
import simulacra.units as u

from .. import utils, exceptions
from . import potential


class ImaginaryGaussianRing(potential.PotentialEnergy):
    def __init__(
        self,
        center: float = 20 * u.bohr_radius,
        width: float = 2 * u.bohr_radius,
        decay_time: float = 100 * u.asec,
    ):
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
        if center < 0:
            raise exceptions.InvalidPotentialParameter("center must be non-negative")
        if width <= 0:
            raise exceptions.InvalidPotentialParameter("width must be non-negative")

        self.center = center
        self.width = width

        if decay_time.imag != 0:
            raise exceptions.InvalidPotentialParameter("decay time must be real")
        if decay_time <= 0:
            raise exceptions.InvalidPotentialParameter("decay time must be positive")

        self.decay_time = decay_time
        self.decay_rate = 1 / decay_time

        self.prefactor = -1j * self.decay_rate * u.hbar

        super().__init__()

    def __call__(self, *, r, **kwargs):
        rel = r - self.center
        return self.prefactor * np.exp(-((rel / self.width) ** 2))

    def __repr__(self):
        return utils.fmt_fields(self, self.center, self.width, self.decay_time)

    def __str__(self):
        return utils.fmt_fields(
            self,
            (self.center, "bohr_radius"),
            (self.width, "bohr_radius"),
            (self.decay_time, "asec"),
        )

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Center", utils.fmt_quantity(self.center, utils.LENGTH_UNITS))
        info.add_field("Width", utils.fmt_quantity(self.width, utils.LENGTH_UNITS))
        info.add_field(
            "Decay Time", utils.fmt_quantity(self.decay_time, utils.TIME_UNITS)
        )

        return info
