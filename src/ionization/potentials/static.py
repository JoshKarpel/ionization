import numpy as np

import simulacra as si
import simulacra.units as u

from .. import utils
from . import potential


class CoulombPotential(potential.PotentialEnergy):
    """A class representing the electric potential energy caused by the Coulomb potential."""

    def __init__(self, charge = 1 * u.proton_charge):
        """
        Construct a Coulomb from a charge.

        :param charge: the charge of the particle providing the potential
        """
        super().__init__()

        self.charge = charge

    def __call__(self, *, r, test_charge, **kwargs):
        """
        Return the Coulomb potential energy evaluated at radial distance r for charge test_charge.

        Accepts only keyword arguments.

        :param r: the radial distance coordinate
        :param test_charge: the test charge
        :param kwargs: absorbs any other keyword arguments
        :return:
        """
        return u.coulomb_constant * self.charge * test_charge / r

    def __repr__(self):
        return utils.fmt_fields(
            self,
            'charge',
        )

    def __str__(self):
        return utils.fmt_fields(
            self,
            ('charge', 'proton_charge'),
        )

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Charge', utils.fmt_quantity(self.charge, utils.CHARGE_UNITS))

        return info


class SoftCoulombPotential(potential.PotentialEnergy):
    """A class representing the electric potential energy caused by the softened Coulomb potential."""

    def __init__(self, charge = 1 * u.proton_charge, softening_distance = .05 * u.bohr_radius):
        """
        Parameters
        ----------
        charge
            The charge of the source.
        softening_distance
            The distance over which the charge is "smeared" to soften the Coulomb singularity.
        """
        super().__init__()

        self.charge = charge
        self.softening_distance = softening_distance

    def __call__(self, *, r, test_charge, **kwargs):
        """
        Return the Coulomb potential energy evaluated at radial distance r for charge test_charge.

        Accepts only keyword arguments.

        Parameters
        ----------
        r
            The position of the test particle.
        test_charge
            The charge of the test particle.
        kwargs

        Returns
        -------
        """
        return u.coulomb_constant * self.charge * test_charge / np.sqrt((r ** 2) + (self.softening_distance ** 2))

    def __repr__(self):
        return utils.fmt_fields(
            self,
            'charge',
            'softening_distance',
        )

    def __str__(self):
        return utils.fmt_fields(
            self,
            ('charge', 'proton_charge'),
            ('softening_distance', 'bohr_radius'),
        )

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Charge', utils.fmt_quantity(self.charge, utils.LENGTH_UNITS))
        info.add_field('Softening Distance', utils.fmt_quantity(self.softening_distance, utils.LENGTH_UNITS))

        return info


class HarmonicOscillator(potential.PotentialEnergy):
    """A PotentialEnergy representing the potential energy of a harmonic oscillator."""

    def __init__(self, spring_constant = 4.20521 * u.N / u.m, center = 0 * u.nm, cutoff_distance = None):
        """Construct a HarmonicOscillator object with the given spring constant and center position."""
        self.spring_constant = spring_constant
        self.center = center

        self.cutoff_distance = cutoff_distance

        super().__init__()

    @classmethod
    def from_frequency_and_mass(cls, omega = 1.5192675e15 * u.Hz, mass = u.electron_mass, **kwargs):
        """Return a HarmonicOscillator constructed from the given angular frequency and mass."""
        return cls(spring_constant = mass * (omega ** 2), **kwargs)

    @classmethod
    def from_ground_state_energy_and_mass(cls, ground_state_energy = 0.5 * u.eV, mass = u.electron_mass, **kwargs):
        """
        Return a HarmonicOscillator constructed from the given ground state energy and mass.

        Note: the ground state energy is half of the energy spacing of the oscillator.
        """
        return cls.from_frequency_and_mass(omega = 2 * ground_state_energy / u.hbar, mass = mass, **kwargs)

    @classmethod
    def from_energy_spacing_and_mass(cls, energy_spacing = 1 * u.eV, mass = u.electron_mass, **kwargs):
        """
        Return a HarmonicOscillator constructed from the given state energy spacing and mass.

        Note: the ground state energy is half of the energy spacing of the oscillator.
        """
        return cls.from_frequency_and_mass(omega = energy_spacing / u.hbar, mass = mass, **kwargs)

    def __call__(self, *, r, **kwargs):
        """Return the HarmonicOscillator potential energy evaluated at position r."""
        centered_r = r - self.center

        inside = 0.5 * self.spring_constant * (centered_r ** 2)
        if self.cutoff_distance is not None:
            outside = 0.5 * self.spring_constant * (self.cutoff_distance ** 2)
            return np.where(np.less_equal(np.abs(centered_r), self.cutoff_distance), inside, outside)
        else:
            return inside

    def omega(self, mass):
        """Return the angular frequency for this potential for the given mass."""
        return np.sqrt(self.spring_constant / mass)

    def frequency(self, mass):
        """Return the cyclic frequency for this potential for the given mass."""
        return self.omega(mass) / u.twopi

    def __repr__(self):
        return utils.fmt_fields(
            self,
            'spring_constant',
            'center',
            'cutoff_distance',
        )

    def __str__(self):
        return utils.fmt_fields(
            self,
            ('spring_constant', 'N/m'),
            ('center', 'bohr_radius'),
            ('cutoff_distance', 'bohr_radius')
        )

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Spring Constant', utils.fmt_quantity(self.spring_constant, utils.FORCE_UNITS))
        info.add_field('Center', utils.fmt_quantity(self.center, utils.LENGTH_UNITS))
        if self.cutoff_distance is None:
            cutoff_val = 'None'
        else:
            cutoff_val = utils.fmt_quantity(self.cutoff_distance, utils.LENGTH_UNITS)
        info.add_field('Cutoff Distance', cutoff_val)

        return info


class FiniteSquareWell(potential.PotentialEnergy):
    def __init__(self, potential_depth = 1 * u.eV, width = 10 * u.nm, center = 0 * u.nm):
        self.potential_depth = np.abs(potential_depth)
        self.width = width
        self.center = center

        super().__init__()

    @property
    def left_edge(self):
        return self.center - (self.width / 2)

    @property
    def right_edge(self):
        return self.center + (self.width / 2)

    def __call__(self, *, r, **kwargs):
        cond = np.greater_equal(r, self.left_edge) * np.less_equal(r, self.right_edge)

        return -self.potential_depth * np.where(cond, 1, 0)

    def __repr__(self):
        return utils.fmt_fields(
            self,
            'potential_depth',
            'width',
            'center',
        )

    def __str__(self):
        return utils.fmt_fields(
            self,
            ('potential_depth', 'u.eV'),
            ('width', 'u.nm'),
            ('center', 'u.nm'),
        )

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Potential Depth', utils.fmt_quantity(self.potential_depth, utils.ENERGY_UNITS))
        info.add_field('Width', utils.fmt_quantity(self.width, utils.LENGTH_UNITS))
        info.add_field('Center', utils.fmt_quantity(self.center, utils.LENGTH_UNITS))

        return info


class GaussianPotential(potential.PotentialEnergy):
    def __init__(self, potential_depth = -1 * u.eV, width = 1 * u.bohr_radius, center = 0):
        super().__init__()

        self.potential_depth = potential_depth
        self.width = width
        self.center = center

    def fwhm(self):
        return 2 * np.sqrt(2 * np.log(2)) * self.width

    def __call__(self, *, r, **kwargs):
        centered_r = r - self.center
        return self.potential_depth * np.exp(-.5 * ((centered_r / self.width) ** 2))

    def __repr__(self):
        return utils.fmt_fields(
            self,
            'potential_depth',
            'width',
            'center',
        )

    def __str__(self):
        return utils.fmt_fields(
            self,
            ('potential_depth', 'u.eV'),
            ('width', 'u.nm'),
            ('center', 'u.nm'),
        )

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Potential Depth', utils.fmt_quantity(self.potential_depth, utils.ENERGY_UNITS))
        info.add_field('Width', utils.fmt_quantity(self.width, utils.LENGTH_UNITS))
        info.add_field('Center', utils.fmt_quantity(self.center, utils.LENGTH_UNITS))

        return info
