import numpy as np

import simulacra as si
import simulacra.units as u

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

    def __str__(self):
        return si.utils.field_str(self, ('charge', 'proton_charge'))

    def __repr__(self):
        return si.utils.field_str(self, 'charge')

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

    def info(self):
        info = super().info()

        info.add_field('Charge', f'{u.uround(self.charge, u.proton_charge, 3)} e')

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

    def __str__(self):
        return si.utils.field_str(self, ('charge', 'proton_charge'), ('softening_distance', 'bohr_radius'))

    def __repr__(self):
        return si.utils.field_str(self, 'charge', 'softening_distance')

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

    def info(self):
        info = super().info()

        info.add_field('Charge', f'{u.uround(self.charge, u.proton_charge)} e')
        info.add_field('Softening Distance', f'{u.uround(self.softening_distance, u.bohr_radius)} a_0')

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

    def __call__(self, *, distance, **kwargs):
        """Return the HarmonicOscillator potential energy evaluated at position distance."""
        d = (distance - self.center)

        inside = 0.5 * self.spring_constant * (d ** 2)
        if self.cutoff_distance is not None:
            outside = 0.5 * self.spring_constant * (self.cutoff_distance ** 2)
            return np.where(np.less_equal(np.abs(d), self.cutoff_distance), inside, outside)
        else:
            return inside

    def omega(self, mass):
        """Return the angular frequency for this potential for the given mass."""
        return np.sqrt(self.spring_constant / mass)

    def frequency(self, mass):
        """Return the cyclic frequency for this potential for the given mass."""
        return self.omega(mass) / u.twopi

    def __str__(self):
        return '{}(spring_constant = {} u.N/u.m, center = {} u.nm)'.format(self.__class__.__name__, np.around(self.spring_constant, 3), u.uround(self.center, u.nm, 3))

    def __repr__(self):
        return '{}(spring_constant = {}, center = {})'.format(self.__class__.__name__, self.spring_constant, self.center)

    def info(self):
        info = super().info()

        info.add_field('Spring Constant', f'{u.uround(self.spring_constant, 1, 3)} u.N/u.m | {u.uround(self.spring_constant, u.atomic_force, 3)} a.u./Bohr Radius')
        info.add_field('Center', f'{u.uround(self.center, u.bohr_radius, 3)} a_0 | {u.uround(self.center, u.nm, 3)} u.nm')
        if self.cutoff_distance is not None:
            info.add_field('Cutoff Distance', f'{u.uround(self.cutoff_distance, u.bohr_radius, 3)} a_0 | {u.uround(self.cutoff_distance, u.nm, 3)} u.nm')

        return info


class FiniteSquareWell(potential.PotentialEnergy):
    def __init__(self, potential_depth = 1 * u.eV, width = 10 * u.nm, center = 0 * u.nm):
        self.potential_depth = np.abs(potential_depth)
        self.width = width
        self.center = center

        super().__init__()

    def __str__(self):
        return si.utils.field_str(self, ('potential_depth', 'u.eV'), ('width', 'u.nm'), ('center', 'u.nm'))

    def __repr__(self):
        return si.utils.field_str(self, 'potential_depth', 'width', 'center')

    @property
    def left_edge(self):
        return self.center - (self.width / 2)

    @property
    def right_edge(self):
        return self.center + (self.width / 2)

    def __call__(self, *, distance, **kwargs):
        cond = np.greater_equal(distance, self.left_edge) * np.less_equal(distance, self.right_edge)

        return -self.potential_depth * np.where(cond, 1, 0)

    def info(self):
        info = super().info()

        info.add_field('Potential Depth', f'{u.uround(self.potential_depth, u.eV)} u.eV')
        info.add_field('Width', f'{u.uround(self.width, u.bohr_radius, 3)} a_0 | {u.uround(self.width, u.nm, 3)} u.nm')
        info.add_field('Center', f'{u.uround(self.center, u.bohr_radius, 3)} a_0 | {u.uround(self.center, u.nm, 3)} u.nm')

        return info


class GaussianPotential(potential.PotentialEnergy):
    def __init__(self, potential_extrema = -1 * u.eV, width = 1 * u.bohr_radius, center = 0):
        super().__init__()

        self.potential_extrema = potential_extrema
        self.width = width
        self.center = center

    def fwhm(self):
        return 2 * np.sqrt(2 * np.log(2)) * self.width

    def __call__(self, *, distance, **kwargs):
        x = distance - self.center
        return self.potential_extrema * np.exp(-.5 * ((x / self.width) ** 2))

    def info(self):
        info = super().info()

        info.add_field('Potential Depth', f'{u.uround(self.potential_extrema, u.eV)} u.eV')
        info.add_field('Width', f'{u.uround(self.width, u.bohr_radius, 3)} a_0 | {u.uround(self.width, u.nm, 3)} u.nm')
        info.add_field('Center', f'{u.uround(self.center, u.bohr_radius, 3)} a_0 | {u.uround(self.center, u.nm, 3)} u.nm')

        return info


