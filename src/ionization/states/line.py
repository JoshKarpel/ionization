import logging
import functools
import itertools
from copy import deepcopy

import mpmath
import numpy as np
import scipy as sp
import scipy.optimize as optimize
import scipy.special as special

import simulacra as si
import simulacra.units as u

from .. import exceptions

from . import state

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NumericOneDState(state.QuantumState):
    discrete_eigenvalues = True
    analytic = False

    def __init__(self, g, energy, analytic_state = None, bound = True, amplitude = 1):
        super().__init__(amplitude = amplitude)

        self.g = g

        self.energy = energy

        self.analytic_state = analytic_state

        self.bound = bound

    def __str__(self):
        return str(self.analytic_state) + '_n'

    def __repr__(self):
        return repr(self.analytic_state) + '_n'

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
        """Return a LaTeX-formatted string for the state."""
        return self.analytic_state.latex

    @property
    def n(self):
        return self.analytic_state.n

    def __call__(self, x):
        return self.g


class OneDFreeParticle(state.QuantumState):
    """A class representing a free particle in one dimension."""

    bound = False
    discrete_eigenvalues = True
    analytic = True

    def __init__(self, wavenumber = u.twopi / u.nm, mass = u.electron_mass, amplitude = 1, dimension_label = 'x'):
        """
        Construct a OneDFreeParticle from a wavenumber and mass.

        :param wavenumber: the wavenumber (2p / wavelength) of the particle
        :param mass: the mass of the particle
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is travelling in
        """
        self.wavenumber = wavenumber
        self.mass = mass
        self.dimension_label = dimension_label

        super().__init__(amplitude = amplitude)

    @classmethod
    def from_energy(cls, energy = 1.50412 * u.eV, k_sign = 1, mass = u.electron_mass, amplitude = 1, dimension_label = 'x'):
        """
        Construct a OneDFreeparticle from an energy and a mass. The sign of the desired k-vector must be included as well.

        :param energy: the energy of the particle
        :param k_sign: a prefactor that will be multiplied by the magnitude of the wavenumber determined from the given energy
        :param mass: the mass of the particle
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is travelling in
        :return: a OneDFreeParticle instance
        """
        return cls(k_sign * np.sqrt(2 * mass * energy) / u.hbar, mass, amplitude = amplitude, dimension_label = dimension_label)

    @property
    def energy(self):
        return ((u.hbar * self.wavenumber) ** 2) / (2 * self.mass)

    @property
    def momentum(self):
        return u.hbar * self.wavenumber

    @property
    def tuple(self):
        return self.wavenumber, self.mass, self.dimension_label

    def __str__(self):
        return r'|k = 2pi * {} 1/nm, E = {} eV>'.format(u.uround(self.wavenumber / u.twopi, u.per_nm), u.uround(self.energy, u.eV))

    def __repr__(self):
        return si.utils.field_str(self, 'wavenumber', 'energy', 'mass', 'amplitude')

    @property
    def latex(self):
        return r'k = {} 2\pi/{}, E = {} {}'.format(u.uround(self.wavenumber / u.twopi, u.per_nm), r'\mathrm{nm}', u.uround(self.energy, u.eV), r'\mathrm{eV}')

    def __call__(self, x):
        """
        Evaluate the free particle wavefunction at a point, or vectorized over an array of points.

        :param x: the distance coordinate along the direction of motion
        :return: the value(s) of the wavefunction at x
        """
        return np.exp(1j * self.wavenumber * x) / np.sqrt(u.twopi)


class QHOState(state.QuantumState):
    """A class representing a bound state of the quantum harmonic oscillator."""

    smallest_n = 0

    bound = True
    discrete_eigenvalues = True
    analytic = True

    def __init__(self, spring_constant, mass = u.electron_mass, n = 0, amplitude = 1, dimension_label = 'x'):
        """
        Construct a QHOState from a spring constant, mass, and energy index n.

        :param spring_constant: the spring constant for the quantum harmonic oscillator
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        """
        self.n = n
        self.spring_constant = spring_constant
        self.mass = mass
        self.dimension_label = dimension_label

        super().__init__(amplitude = amplitude)

    @classmethod
    def from_omega_and_mass(cls, omega, mass = u.electron_mass, n = 0, amplitude = 1, dimension_label = 'x'):
        """
        Construct a QHOState from an angular frequency, mass, and energy index n.

        :param omega: the fundamental angular frequency of the quantum harmonic oscillator
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        :return: a QHOState instance
        """
        return cls(spring_constant = mass * (omega ** 2), mass = mass, n = n, amplitude = amplitude, dimension_label = dimension_label)

    @classmethod
    def from_potential(cls, potential, mass, n = 0, amplitude = 1, dimension_label = 'x'):
        """
        Construct a QHOState from a HarmonicOscillator, mass, and energy index n.

        :param potential: a HarmonicOscillator instance
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        :return: a QHOState instance
        """
        return cls(spring_constant = potential.spring_constant, mass = mass, n = n, amplitude = amplitude, dimension_label = dimension_label)

    @property
    def omega(self):
        return np.sqrt(self.spring_constant / self.mass)

    @property
    def energy(self):
        return u.hbar * self.omega * (self.n + 0.5)

    @property
    def frequency(self):
        return self.omega / u.twopi

    @property
    def period(self):
        return 1 / self.frequency

    def __str__(self):
        return self.ket

    def __repr__(self):
        return '{}(n = {}, mass = {}, omega = {}, energy = {}, amplitude = {})'.format(self.__class__.__name__,
                                                                                       self.n,
                                                                                       self.mass,
                                                                                       self.omega,
                                                                                       self.energy,
                                                                                       self.amplitude)

    @property
    def ket(self):
        return '|{}>'.format(self.n)

    @property
    def bra(self):
        return '<{}|'.format(self.n)

    @property
    def latex(self):
        """Return a LaTeX-formatted string for the QHOState."""
        return r'{}'.format(self.n)

    @property
    def tuple(self):
        return self.n, self.mass, self.omega, self.dimension_label

    def __call__(self, x):
        """
        Evaluate the quantum harmonic oscillator bound state wavefunction at a point, or vectorized over an array of points.

        Warning: for large enough n (>= ~60) this will fail due to n! overflowing.

        :param x: the distance coordinate along the direction of confinement
        :return: the value(s) of the wavefunction at x
        """
        norm = ((self.mass * self.omega / (u.pi * u.hbar)) ** (1 / 4)) / (np.float64(2 ** (self.n / 2)) * np.sqrt(np.float64(sp.math.factorial(self.n))))
        exp = np.exp(-self.mass * self.omega * (x ** 2) / (2 * u.hbar))
        herm = special.hermite(self.n)(np.sqrt(self.mass * self.omega / u.hbar) * x)

        # TODO: Stirling's approximation for large enough n in the normalization factor

        return self.amplitude * (norm * exp * herm).astype(np.complex128)


class FiniteSquareWellState(state.QuantumState):
    """A class representing a bound state of a finite square well."""

    smallest_n = 1

    bound = True
    discrete_eigenvalues = True
    analytic = True

    def __init__(self, well_depth, well_width, mass, n = 1, well_center = 0, amplitude = 1, dimension_label = 'x'):
        """
        Construct a FiniteSquareWellState from the well properties, the particle mass, and an energy index.

        :param well_depth: the depth of the potential well
        :param well_width: the full width of the potential well
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param well_center: the center position of the well
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        """
        self.well_depth = well_depth
        self.well_width = well_width
        self.well_center = well_center
        self.mass = mass
        self.n = n
        self.dimension_label = dimension_label

        z_0 = (well_width / 2) * np.sqrt(2 * mass * well_depth) / u.hbar

        if n - 1 > z_0 // (u.pi / 2):
            raise exceptions.Illegalstate.QuantumState('There is no bound state with the given parameters')

        left_bound = (n - 1) * u.pi / 2
        right_bound = min(z_0, left_bound + (u.pi / 2))

        with np.errstate(divide = 'ignore'):  # ignore division by zero on the edges
            # determine the energy of the state by solving a transcendental equation
            if n % 2 != 0:  # n is odd
                z = optimize.brentq(lambda z: np.tan(z) - np.sqrt(((z_0 / z) ** 2) - 1), left_bound, right_bound)
                self.function_inside_well = np.cos
                self.symmetry = 'symmetric'
            else:  # n is even
                z = optimize.brentq(lambda z: (1 / np.tan(z)) + np.sqrt(((z_0 / z) ** 2) - 1), left_bound, right_bound)
                self.function_inside_well = np.sin
                self.symmetry = 'antisymmetric'

        self.wavenumber_inside_well = z / (well_width / 2)
        self.energy = (((u.hbar * self.wavenumber_inside_well) ** 2) / (2 * mass)) - well_depth
        self.wavenumber_outside_well = np.sqrt(-2 * mass * self.energy) / u.hbar

        self.normalization_factor_inside_well = 1 / np.sqrt((self.well_width / 2) + (1 / self.wavenumber_outside_well))
        self.normalization_factor_outside_well = np.exp(self.wavenumber_outside_well * (self.well_width / 2)) * self.function_inside_well(self.wavenumber_inside_well * (self.well_width / 2)) / np.sqrt((self.well_width / 2) + (1 / self.wavenumber_outside_well))

        super().__init__(amplitude = amplitude)

    @classmethod
    def from_potential(cls, potential, mass, n = 1, amplitude = 1, dimension_label = 'x'):
        """
        Construct a FiniteSquareWellState from a FiniteSquareWell potential, the particle mass, and an energy index.

        :param potential: a FiniteSquareWell potential.
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        :return: a FiniteSquareWellState instance
        """
        return cls(potential.potential_depth, potential.width, mass, n = n, well_center = potential.center, amplitude = amplitude, dimension_label = dimension_label)

    def __str__(self):
        return self.ket

    def __repr__(self):
        return '{}(n = {}, mass = {}, well_depth = {}, well_width = {}, energy = {}, amplitude = {})'.format(self.__class__.__name__,
                                                                                                             self.n,
                                                                                                             self.mass,
                                                                                                             self.well_depth,
                                                                                                             self.well_width,
                                                                                                             self.energy,
                                                                                                             self.amplitude)

    @property
    def ket(self):
        return '|{}>'.format(self.n)

    @property
    def bra(self):
        return '<{}|'.format(self.n)

    @property
    def latex(self):
        """Return a LaTeX-formatted string for the QHOState."""
        return r'{}'.format(self.n)

    @property
    def tuple(self):
        return self.well_depth, self.well_width, self.mass, self.n

    @classmethod
    def all_states_of_well_from_parameters(cls, well_depth, well_width, mass, well_center = 0, amplitude = 1):
        """
        Return a list containing all of the bound states of a well.

        The states are ordered in increasing energy.

        :param well_depth: the depth of the potential well
        :param well_width: the full width of the potential well
        :param mass: the mass of the particle
        :param well_center: the center position of the well
        :param amplitude: the probability amplitude of the states
        :return: a list of FiniteSquareWell instances
        """
        states = []
        for n in itertools.count(1):
            try:
                states.append(cls(well_depth, well_width, mass, n = n, well_center = well_center, amplitude = amplitude))
            except exceptions.Illegalstate.QuantumState:
                return states

    @classmethod
    def all_states_of_well_from_well(cls, finite_square_well_potential, mass, amplitude = 1):
        """
        Return a list containing all of the bound states of a well.

        :param finite_square_well_potential: a FiniteSquareWell
        :param mass: the mass of the particle
        :param amplitude: the probability amplitude of the states
        :return:
        """
        return cls.all_states_of_well_from_parameters(finite_square_well_potential.potential_depth,
                                                      finite_square_well_potential.width,
                                                      mass,
                                                      well_center = finite_square_well_potential.center,
                                                      amplitude = 1)

    @property
    def left_edge(self):
        """Return the position of the left edge of the well."""
        return self.well_center - (self.well_width / 2)

    @property
    def right_edge(self):
        """Return the position of the right edge of the well."""
        return self.well_center + (self.well_width / 2)

    def __call__(self, x):
        """
        Evaluate the finite square well bound state wavefunction at a point, or vectorized over an array of points.

        :param x: the distance coordinate along the direction of confinement
        :return: the value(s) of the wavefunction at x
        """
        cond = np.greater_equal(x, self.left_edge) * np.less_equal(x, self.right_edge)

        if self.symmetry == 'antisymmetric':
            sym = -1
        else:
            sym = 1

        psi = np.where(cond,
                       self.normalization_factor_inside_well * self.function_inside_well(self.wavenumber_inside_well * x),
                       self.normalization_factor_outside_well * np.exp(-self.wavenumber_outside_well * np.abs(x))).astype(np.complex128)

        symmetrization = np.where(np.less_equal(x, self.left_edge),
                                  sym, 1)

        return psi * symmetrization


class GaussianWellState(state.QuantumState):
    """A class representing a bound state of a finite square well."""

    smallest_n = 0

    bound = True
    discrete_eigenvalues = True
    analytic = True

    def __init__(self, well_depth, well_width, mass, n = 0, well_center = 0, amplitude = 1, dimension_label = 'x'):
        """
        Construct a GaussianWellState from the well properties, the particle mass, and an energy index.

        :param well_depth: the depth of the potential well
        :param well_width: the full width of the potential well
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param well_center: the center position of the well
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        """
        self.well_depth = np.abs(well_depth)
        self.well_width = well_width
        self.well_center = well_center
        self.mass = mass
        self.n = n
        self.dimension_label = dimension_label

        if n > 0:
            logger.warning('Analytic GaussianWellStates are not available for n > 0, and n = 0 is only approximate!')

        max_n = np.ceil(2 * np.sqrt(2 * mass * self.well_depth / (u.pi * (u.hbar ** 2))) * well_width) + 0.5
        if n > max_n:
            raise exceptions.Illegalstate.QuantumState('Bound state energy must be less than zero')

        self.width = optimize.newton(lambda w: ((w ** 4) / (((well_width ** 2) + (w ** 2)) ** 1.5)) - ((u.hbar ** 2) / (4 * mass * well_width * self.well_depth)), well_width)
        self.energy = -(well_width * self.well_depth / np.sqrt(((well_width ** 2) + (self.width ** 2))) + ((u.hbar ** 2) / (8 * mass * (self.width ** 2))))

        super().__init__(amplitude = amplitude)

    @classmethod
    def from_potential(cls, potential, mass, n = 0, amplitude = 1, dimension_label = 'x'):
        """
        Construct a FiniteSquareWellState from a FiniteSquareWell potential, the particle mass, and an energy index.

        :param potential: a FiniteSquareWell potential.
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        :return: a FiniteSquareWellState instance
        """
        return cls(potential.potential_extrema, potential.width, mass, n = n, well_center = potential.center, amplitude = amplitude, dimension_label = dimension_label)

    def __str__(self):
        return self.ket

    def __repr__(self):
        return '{}(n = {}, mass = {}, well_depth = {}, well_width = {}, energy = {}, amplitude = {})'.format(self.__class__.__name__,
                                                                                                             self.n,
                                                                                                             self.mass,
                                                                                                             self.well_depth,
                                                                                                             self.well_width,
                                                                                                             self.energy,
                                                                                                             self.amplitude)

    @property
    def ket(self):
        return '|{}>'.format(self.n)

    @property
    def bra(self):
        return '<{}|'.format(self.n)

    @property
    def latex(self):
        """Return a LaTeX-formatted string for the QHOState."""
        return r'{}'.format(self.n)

    @property
    def tuple(self):
        return self.well_depth, self.well_width, self.mass, self.n

    def __call__(self, x):
        """
        Evaluate the finite square well bound state wavefunction at a point, or vectorized over an array of points.

        :param x: the distance coordinate along the direction of confinement
        :return: the value(s) of the wavefunction at x
        """

        return np.exp(-.25 * (x / self.width) ** 2) / (np.sqrt(np.sqrt(u.twopi) * self.width))


class OneDSoftCoulombState(state.QuantumState):
    smallest_n = 1

    bound = True
    discrete_eigenvalues = True
    analytic = True

    def __init__(self, n = 1, amplitude = 1, dimension_label = 'x'):
        self.n = n

        super().__init__(amplitude = amplitude)

    @classmethod
    def from_potential(cls, potential, test_mass, n = 1, **kwargs):
        return cls(n = n)

    @property
    def tuple(self):
        return self.n,

    @property
    def ket(self):
        return '|{}>'.format(self.n)

    @property
    def bra(self):
        return '<{}|'.format(self.n)

    def __str__(self):
        return self.ket
