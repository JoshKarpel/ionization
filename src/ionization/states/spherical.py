import logging
import functools

import mpmath
import numpy as np
import scipy as sp
import scipy.special as special

import simulacra as si
import simulacra.units as u

from .. import core, exceptions

from . import state

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class FreeSphericalWave(state.QuantumState):
    """A class that represents a free spherical wave."""

    bound = False
    discrete_eigenvalues = False
    analytic = True

    def __init__(self, energy = 1 * u.eV, l = 0, m = 0, amplitude = 1):
        """
        Construct a FreeState from its energy and angular momentum quantum numbers.

        :param energy: energy of the free state
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
        :param amplitude: the probability amplitude of the state
        """
        super().__init__(amplitude = amplitude)

        if any(int(x) != x for x in (l, m)):
            raise exceptions.IllegalQuantumState('l and m must be integers')

        self.energy = energy

        if l >= 0:
            self._l = l
        else:
            raise exceptions.IllegalQuantumState('l ({}) must be greater than or equal to zero'.format(l))

        if -l <= m <= l:
            self._m = m
        else:
            raise exceptions.IllegalQuantumState('m ({}) must be between -l and l ({} to {})'.format(m, -l, l))

    @classmethod
    def from_wavenumber(cls, k, l = 0, m = 0):
        """Construct a FreeState from its wavenumber and angular momentum quantum numbers."""
        energy = core.electron_energy_from_wavenumber(k)

        return cls(energy, l, m)

    @property
    def k(self):
        return core.electron_wavenumber_from_energy(self.energy)

    @property
    def l(self):
        return self._l

    @property
    def m(self):
        return self._m

    @property
    def tuple(self):
        return self.k, self.l, self.m, self.energy

    @property
    def spherical_harmonic(self):
        """Return the SphericalHarmonic for the state's angular momentum quantum numbers."""
        return si.math.SphericalHarmonic(l = self.l, m = self.m)

    def __str__(self):
        return self.ket

    def __repr__(self):
        return '{}(energy = {} eV, k = {} 1/nm, l = {}, m = {}, amplitude = {})'.format(self.__class__.__name__, u.uround(self.energy, u.eV, 3), u.uround(self.k, 1 / u.nm, 3), self.l, self.m, self.amplitude)

    @property
    def ket(self):
        return '{}|{} eV, {} 1/nm, {}, {}>'.format(np.around(self.amplitude, 3), u.uround(self.energy, u.eV, 3), u.uround(self.k, 1 / u.nm, 3), self.l, self.m)

    @property
    def bra(self):
        return '{}<{} eV, {} 1/nm, {}, {}|'.format(np.around(self.amplitude, 3), u.uround(self.energy, u.eV, 3), u.uround(self.k, 1 / u.nm, 3), self.l, self.m)

    @property
    def latex(self):
        """Return a LaTeX-formatted string for the FreeSphericalWave."""
        return r'\phi_{{{},{},{}}}'.format(u.uround(self.energy, u.eV, 3), self.l, self.m)

    def radial_function(self, r):
        return np.sqrt(2 * (self.k ** 2) / u.pi) * special.spherical_jn(self.l, self.k * r)

    def __call__(self, r, theta, phi):
        """
        Evaluate the free spherical wave wavefunction at a point, or vectorized over an array of points.

        :param r: radial coordinate
        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the wavefunction at (r, theta, phi)
        """
        return self.amplitude * self.radial_function(r) * self.spherical_harmonic(theta, phi)


class HydrogenBoundState(state.QuantumState):
    """A class that represents a hydrogenic bound state."""

    bound = True
    discrete_eigenvalues = True
    analytic = True

    def __init__(self, n = 1, l = 0, m = 0, amplitude = 1):
        """
        Construct a HydrogenBoundState from its three quantum numbers (n, l, m).

        :param n: principal quantum number
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
        """
        super().__init__(amplitude = amplitude)

        self.n = n
        self.l = l
        self.m = m

    @property
    def n(self):
        """Gets _n."""
        return self._n

    @n.setter
    def n(self, n):
        if 0 < n == int(n):
            self._n = int(n)
        else:
            raise exceptions.IllegalQuantumState('n ({}) must be an integer greater than zero'.format(n))

    @property
    def l(self):
        """Gets _l."""
        return self._l

    @l.setter
    def l(self, l):
        if int(l) == l and 0 <= l < self.n:
            self._l = int(l)
        else:
            raise exceptions.IllegalQuantumState('l ({}) must be greater than or equal to zero and less than n ({})'.format(l, self.n))

    @property
    def m(self):
        """Gets _m."""
        return self._m

    @m.setter
    def m(self, m):
        if int(m) == m and -self.l <= m <= self.l:
            self._m = int(m)
        else:
            raise exceptions.IllegalQuantumState('|m| (|{}|) must be less than or equal to l ({})'.format(m, self.l))

    @property
    def energy(self):
        return -u.rydberg / (self.n ** 2)

    @property
    def tuple(self):
        return self.n, self.l, self.m

    @property
    def spherical_harmonic(self):
        """Gets the SphericalHarmonic associated with the HydrogenBoundState's l and m."""
        return si.math.SphericalHarmonic(l = self.l, m = self.m)

    def __str__(self):
        """Returns the external string representation of the HydrogenBoundState."""
        return self.ket

    def __repr__(self):
        """Returns the internal string representation of the HydrogenBoundState."""
        return si.utils.field_str(self, 'n', 'l', 'm', 'amplitude')

    @property
    def ket(self):
        """Gets the ket representation of the HydrogenBoundState."""
        return '|{},{},{}>'.format(*self.tuple)

    @property
    def bra(self):
        """Gets the bra representation of the HydrogenBoundState"""
        return '<{},{},{}|'.format(*self.tuple)

    @property
    def latex(self):
        """Gets a LaTeX-formatted string for the HydrogenBoundState."""
        return r'\psi_{{{},{},{}}}'.format(*self.tuple)

    @staticmethod
    def sort_key(state):
        return state.n, state.l, state.m

    def radial_function(self, r):
        """Return the radial part of the wavefunction R(r) evaluated at r."""
        normalization = np.sqrt(((2 / (self.n * u.bohr_radius)) ** 3) * (sp.math.factorial(self.n - self.l - 1) / (2 * self.n * sp.math.factorial(self.n + self.l))))  # Griffith's normalization
        r_dep = np.exp(-r / (self.n * u.bohr_radius)) * ((2 * r / (self.n * u.bohr_radius)) ** self.l)
        lag_poly = special.eval_genlaguerre(self.n - self.l - 1, (2 * self.l) + 1, 2 * r / (self.n * u.bohr_radius))

        return self.amplitude * normalization * r_dep * lag_poly

    def __call__(self, r, theta, phi):
        """
        Evaluate the hydrogenic bound state wavefunction at a point, or vectorized over an array of points.

        :param r: radial coordinate
        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the wavefunction at (r, theta, phi)
        """
        return self.radial_function(r) * self.spherical_harmonic(theta, phi)


def coulomb_phase_shift(l, k):
    """

    :param l: angular momentum quantum number
    :param k: wavenumber
    :return:
    """
    gamma = 1j / (k * u.bohr_radius)
    return np.angle(special.gamma(1 + l + gamma))


class HydrogenCoulombState(state.QuantumState):
    """A class that represents a hydrogenic free state."""

    bound = False
    discrete_eigenvalues = False
    analytic = True

    def __init__(self, energy = 1 * u.eV, l = 0, m = 0, amplitude = 1):
        """
        Construct a HydrogenCoulombState from its energy and angular momentum quantum numbers.

        :param energy: energy of the free state
        :param l: orbital angular momentum quantum number
        :param m: quantum number for angular momentum z-component
        """
        super().__init__(amplitude = amplitude)

        if any(int(x) != x for x in (l, m)):
            raise exceptions.IllegalQuantumState('l and m must be integers')

        if energy < 0:
            raise exceptions.IllegalQuantumState('energy must be greater than zero')
        self.energy = energy

        if l < 0:
            raise exceptions.IllegalQuantumState('l ({}) must be greater than or equal to zero'.format(l))
        self.l = int(l)

        if not -l <= m <= l:
            raise exceptions.IllegalQuantumState('m ({}) must be between -l and l ({} to {})'.format(m, -l, l))
        self.m = int(m)

    @classmethod
    def from_wavenumber(cls, k, l = 0, m = 0):
        """Construct a HydrogenCoulombState from its wavenumber and angular momentum quantum numbers."""
        energy = core.electron_energy_from_wavenumber(k)

        return cls(energy, l, m)

    @property
    def k(self):
        return core.electron_wavenumber_from_energy(self.energy)

    @property
    def spherical_harmonic(self):
        """Return the SphericalHarmonic for the state's angular momentum quantum numbers."""
        return si.math.SphericalHarmonic(l = self.l, m = self.m)

    @property
    def tuple(self):
        return self.k, self.l, self.m, self.energy

    def __str__(self):
        return self.ket

    def __repr__(self):
        return '{}(energy = {} eV, k = {} 1/nm, l = {}, m = {}, amplitude = {})'.format(self.__class__.__name__, u.uround(self.energy, u.eV, 3), u.uround(self.k, 1 / u.nm, 3), self.l, self.m, self.amplitude)

    @property
    def ket(self):
        return '|{} eV, {} 1/nm, {}, {}>'.format(u.uround(self.energy, u.eV, 3), u.uround(self.k, 1 / u.nm, 3), self.l, self.m)

    @property
    def bra(self):
        return '<{} eV, {} 1/nm, {}, {}|'.format(u.uround(self.energy, u.eV, 3), u.uround(self.k, 1 / u.nm, 3), self.l, self.m)

    @property
    def latex(self):
        """Return a LaTeX-formatted string for the HydrogenCoulombState."""
        return r'\phi_{{{},{},{}}}'.format(u.uround(self.energy, u.eV, 3), self.l, self.m)

    def radial_function(self, r):
        x = r / u.bohr_radius
        epsilon = self.energy / u.rydberg
        unit_prefactor = np.sqrt(1 / (u.bohr_radius * u.rydberg))

        if epsilon > 0:
            kappa = 1j / np.sqrt(epsilon)

            a = self.l + 1 - kappa
            b = 2 * (self.l + 1)
            hgf = functools.partial(mpmath.hyp1f1, a, b)  # construct a partial function, with a and b filled in
            hgf = np.vectorize(hgf, otypes = [np.complex128])  # vectorize using numpy

            A = (kappa ** (-((2 * self.l) + 1))) * special.gamma(1 + self.l + kappa) / special.gamma(kappa - self.l)
            B = A / (1 - np.exp(-u.twopi / np.sqrt(epsilon)))
            s_prefactor = np.sqrt(B / 2)

            l_prefactor = (2 ** (self.l + 1)) / special.factorial((2 * self.l) + 1)

            prefactor = s_prefactor * l_prefactor * unit_prefactor

            return self.amplitude * prefactor * hgf(2 * x / kappa) * (x ** (self.l + 1)) * np.exp(-x / kappa) / r

        elif epsilon == 0:
            bessel_order = (2 * self.l) + 1
            prefactor = unit_prefactor
            bessel = functools.partial(special.jv, bessel_order)  # construct a partial function with the Bessel function order filled in

            return self.amplitude * prefactor * bessel(np.sqrt(8 * x)) * np.sqrt(x) / r

    def __call__(self, r, theta, phi):
        """
        Evaluate the hydrogenic Coulomb state wavefunction at a point, or vectorized over an array of points.

        :param r: radial coordinate
        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the wavefunction at (r, theta, phi)
        """
        return self.radial_function(r) * self.spherical_harmonic(theta, phi)


class NumericSphericalHarmonicState(state.QuantumState):
    discrete_eigenvalues = True
    analytic = False

    def __init__(self, g, l, m, energy, analytic_state, bound = True, amplitude = 1):
        super().__init__(amplitude = amplitude)

        self.g = g

        self.l = l
        self.m = m
        self.energy = energy

        self.analytic_state = analytic_state

        self.bound = bound

    def __str__(self):
        return str(self.analytic_state) + '_n'

    def __repr__(self):
        return repr(self.analytic_state) + '_n'

    @property
    def n(self):
        return self.analytic_state.n

    @property
    def k(self):
        return self.analytic_state.k

    @property
    def tuple(self):
        return self.analytic_state.tuple

    @property
    def ket(self):
        return self.analytic_state.ket

    @property
    def bra(self):
        return self.analytic_state.bra

    @property
    def latex(self):
        """Return a LaTeX-formatted string for the NumericSphericalHarmonicState."""
        return self.analytic_state.latex

    @property
    def spherical_harmonic(self):
        """Return the SphericalHarmonic for the state's angular momentum quantum numbers."""
        return si.math.SphericalHarmonic(l = self.l, m = self.m)

    def radial_function(self, r):
        return self.g

    def __call__(self, r, theta, phi):
        return self.radial_function(r) * self.spherical_harmonic(theta, phi)
