from typing import Optional

import numpy as np

import simulacra as si
import simulacra.units as u

from .. import utils
from . import potential


class CoulombPotential(potential.PotentialEnergy):
    """The electric potential energy caused by the Coulomb potential."""

    def __init__(self, charge: float = 1 * u.proton_charge):
        """
        Parameters
        ----------
        charge
            The charge creating the Coulomb potential.
        """
        super().__init__()

        self.charge = charge

    def __call__(self, *, r, test_charge, **kwargs):
        """

        Parameters
        ----------
        r
            Distance to the charge from the test particle.
        test_charge
            The charge of the test particle.
        kwargs
            Ignores additional keyword arguments.
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

    def __init__(
        self,
        charge: float = 1 * u.proton_charge,
        softening_distance: float = .05 * u.bohr_radius,
    ):
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
        Parameters
        ----------
        r
            Distance to the charge from the test particle.
        test_charge
            The charge of the test particle.
        kwargs
            Ignores additional keyword arguments.
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

    def __init__(
        self,
        spring_constant: float = 4.20521 * u.N / u.m,
        center: float = 0 * u.nm,
        cutoff_distance: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        spring_constant
            The spring constant of the harmonic oscillator.
        center
            The position of the center of the potential (the equilibrium position).
        cutoff_distance
            The distance at which the potential stops acting.
        """
        self.spring_constant = spring_constant
        self.center = center

        self.cutoff_distance = cutoff_distance

        super().__init__()

    @classmethod
    def from_frequency_and_mass(
        cls,
        omega: float = 1.5192675e15 * u.Hz,
        mass: float = u.electron_mass,
        **kwargs,
    ) -> 'HarmonicOscillator':
        """
        Parameters
        ----------
        omega
        mass
        kwargs
        """
        return cls(spring_constant = mass * (omega ** 2), **kwargs)

    @classmethod
    def from_ground_state_energy_and_mass(
        cls,
        ground_state_energy: float = 0.5 * u.eV,
        mass: float = u.electron_mass,
        **kwargs,
    ) -> 'HarmonicOscillator':
        """
        Return a :class:`HarmonicOscillator` constructed from the given ground state energy and mass.

        NB: the ground state energy is half of the energy spacing of the oscillator.

        Parameters
        ----------
        ground_state_energy
            The ground state energy of the quantum harmonic oscillator.
        mass
            The mass of the test particle that would have that ``ground_state_energy``.
        kwargs

        """
        return cls.from_frequency_and_mass(
            omega = 2 * ground_state_energy / u.hbar,
            mass = mass,
            **kwargs,
        )

    @classmethod
    def from_energy_spacing_and_mass(
        cls,
        energy_spacing: float = 1 * u.eV,
        mass: float = u.electron_mass,
        **kwargs,
    ) -> 'HarmonicOscillator':
        """
        Return a :class:`HarmonicOscillator` constructed from the given energy spacing and mass.

        NB: the ground state energy is half of the energy spacing of the oscillator.

        Parameters
        ----------
        energy_spacing
            The energy spacing of the quantum harmonic oscillator.
        mass
            The mass of the test particle that would have that ``energy_spacing``.
        kwargs

        """
        return cls.from_frequency_and_mass(
            omega = energy_spacing / u.hbar,
            mass = mass,
            **kwargs,
        )

    def __call__(self, *, r, **kwargs):
        """
        Parameters
        ----------
        r
            The distance to the center of the harmonic oscillator potential.
        kwargs
            Ignores additional keyword arguments.
        """
        centered_r = r - self.center

        inside = 0.5 * self.spring_constant * (centered_r ** 2)
        if self.cutoff_distance is not None:
            outside = 0.5 * self.spring_constant * (self.cutoff_distance ** 2)
            return np.where(np.less_equal(np.abs(centered_r), self.cutoff_distance), inside, outside)
        else:
            return inside

    def omega(self, mass: float) -> float:
        """Return the angular frequency for this potential for the given mass."""
        return np.sqrt(self.spring_constant / mass)

    def frequency(self, mass: float) -> float:
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
    """
    A finite square well potential.

    Attributes
    ----------
    left_edge
        The position of the left edge of the well.
    right_edge
        The position of the right edge of the well.
    """

    def __init__(
        self,
        potential_depth: float = 1 * u.eV,
        width: float = 10 * u.nm,
        center: float = 0 * u.nm,
    ):
        """

        Parameters
        ----------
        potential_depth
            The depth of the potential well.
        width
            The width the potential well.
        center
            The center position of the potential well.
        """
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
    """A Gaussian potential well."""

    def __init__(
        self,
        potential_extrema: float = -1 * u.eV,
        width: float = 1 * u.bohr_radius,
        center: float = 0,
    ):
        """

        Parameters
        ----------
        potential_extrema
            The value of the potential at ``r = 0``.
            Can be positive or negative!
        width
            The width of the potential.
        center
            The center position of the potential.
        """
        super().__init__()

        self.potential_extrema = potential_extrema
        self.width = width
        self.center = center

    def fwhm(self) -> float:
        """Return the full-width at half-max of the potential."""
        return 2 * np.sqrt(2 * np.log(2)) * self.width

    def __call__(self, *, r, **kwargs):
        centered_r = r - self.center
        return self.potential_extrema * np.exp(-.5 * ((centered_r / self.width) ** 2))

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

        info.add_field('Potential Depth', utils.fmt_quantity(self.potential_extrema, utils.ENERGY_UNITS))
        info.add_field('Width', utils.fmt_quantity(self.width, utils.LENGTH_UNITS))
        info.add_field('Center', utils.fmt_quantity(self.center, utils.LENGTH_UNITS))

        return info


class GaussianScatterer(potential.PotentialEnergy):
    """A Gaussian potential well."""

    def __init__(
        self,
        potential_extrema: float = 100 * u.eV,
        z_width: float = 1 * u.nm,
        x_width: float = 1 * u.nm,
        z_center: float = 0,
        x_center: float = 0,
    ):
        super().__init__()

        self.potential_extrema = potential_extrema
        self.z_width = z_width
        self.x_width = x_width
        self.z_center = z_center
        self.x_center = x_center

    def __call__(self, *, z, x, **kwargs):
        centered_z = z - self.z_center
        centered_x = x - self.x_center

        gaussian = self.potential_extrema
        gaussian *= np.exp(-.5 * ((centered_z / self.z_width) ** 2))
        gaussian *= np.exp(-.5 * ((centered_x / self.x_width) ** 2))

        return gaussian

    def info(self):
        info = super().info()

        info.add_field('Potential Extrema', utils.fmt_quantity(self.potential_extrema, utils.ENERGY_UNITS))

        info.add_field('Z Center', utils.fmt_quantity(self.z_center, utils.LENGTH_UNITS))
        info.add_field('X Center', utils.fmt_quantity(self.x_center, utils.LENGTH_UNITS))

        info.add_field('Z Width', utils.fmt_quantity(self.z_width, utils.LENGTH_UNITS))
        info.add_field('X Width', utils.fmt_quantity(self.x_width, utils.LENGTH_UNITS))

        return info


class LogisticScatterer(potential.PotentialEnergy):
    """A logistic potential barrier."""

    def __init__(
        self,
        potential_extrema: float = 100 * u.eV,
        z_width: float = .2 * u.nm,
        x_width: float = .2 * u.nm,
        z_extent = 1 * u.nm,
        x_extent = 1 * u.nm,
        z_center: float = 0,
        x_center: float = 0,
    ):
        super().__init__()

        self.potential_extrema = potential_extrema
        self.z_width = z_width
        self.x_width = x_width
        self.z_extent = z_extent
        self.x_extent = x_extent
        self.z_center = z_center
        self.x_center = x_center

    @classmethod
    def from_bounds(
        cls,
        potential_extrema: float = 100 * u.eV,
        z_min = -.5 * u.nm,
        z_max = .5 * u.nm,
        x_min = -.5 * u.nm,
        x_max = .5 * u.nm,
        z_width: float = .2 * u.nm,
        x_width: float = .2 * u.nm,
    ):
        z_extent = np.abs(z_max - z_min)
        x_extent = np.abs(x_max - x_min)

        z_center = (z_max + z_min) / 2
        x_center = (x_max - x_min) / 2

        return cls(
            potential_extrema = potential_extrema,
            x_width = x_width,
            z_width = z_width,
            z_extent = z_extent,
            x_extent = x_extent,
            z_center = z_center,
            x_center = x_center,
        )

    def __call__(self, *, z, x, **kwargs):
        centered_z = z - self.z_center
        centered_x = x - self.x_center

        pot = self.potential_extrema
        pot *= 1 / (1 + np.exp(-(centered_x + self.x_extent) / self.x_width)) - 1 / (1 + np.exp(-(centered_x - self.x_extent) / self.x_width))
        pot *= 1 / (1 + np.exp(-(centered_z + self.z_extent) / self.z_width)) - 1 / (1 + np.exp(-(centered_z - self.z_extent) / self.z_width))

        return pot

    def info(self):
        info = super().info()

        info.add_field('Potential Extrema', utils.fmt_quantity(self.potential_extrema, utils.ENERGY_UNITS))

        info.add_field('Z Center', utils.fmt_quantity(self.z_center, utils.LENGTH_UNITS))
        info.add_field('X Center', utils.fmt_quantity(self.x_center, utils.LENGTH_UNITS))

        info.add_field('Z Extent', utils.fmt_quantity(self.z_extent, utils.LENGTH_UNITS))
        info.add_field('X Extent', utils.fmt_quantity(self.x_extent, utils.LENGTH_UNITS))

        info.add_field('Z Width', utils.fmt_quantity(self.z_width, utils.LENGTH_UNITS))
        info.add_field('X Width', utils.fmt_quantity(self.x_width, utils.LENGTH_UNITS))

        return info
