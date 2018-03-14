import logging
import itertools

import numpy as np
import scipy.optimize as optimize
import scipy.special as special
import scipy.misc as spmisc

import simulacra as si
import simulacra.units as u

from .. import mesh, potentials, utils, exceptions

from . import states

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OneDPlaneWave(states.QuantumState):
    """A class representing a free particle in one dimension."""

    eigenvalues = states.Eigenvalues.CONTINUOUS
    binding = states.Binding.FREE
    derivation = states.Derivation.ANALYTIC

    def __init__(
        self,
        wavenumber: float = u.twopi / u.nm,
        mass: float = u.electron_mass,
        amplitude: states.ProbabilityAmplitude = 1,
    ):
        """

        Parameters
        ----------
        wavenumber
        mass
        amplitude
        """
        self.wavenumber = wavenumber
        self.mass = mass

        super().__init__(amplitude = amplitude)

    @classmethod
    def from_energy(
        cls,
        energy: float = 1.50412 * u.eV,
        k_sign: int = 1,
        mass = u.electron_mass,
        amplitude: states.ProbabilityAmplitude = 1,
    ):
        if k_sign not in [1, -1]:
            raise exceptions.IllegalQuantumState
        return cls(k_sign * np.sqrt(2 * mass * energy) / u.hbar, mass, amplitude = amplitude)

    @property
    def energy(self):
        return ((u.hbar * self.wavenumber) ** 2) / (2 * self.mass)

    @property
    def momentum(self):
        return u.hbar * self.wavenumber

    @property
    def tuple(self):
        return self.wavenumber, self.mass

    def __call__(self, x):
        """
        Evaluate the free particle wavefunction at a point, or vectorized over an array of points.

        :param x: the distance coordinate along the direction of motion
        :return: the value(s) of the wavefunction at x
        """
        return np.exp(1j * self.wavenumber * x) / np.sqrt(u.twopi)

    def __repr__(self):
        return utils.fmt_fields(self, 'wavenumber', 'energy', 'mass', 'amplitude')

    @property
    def ket(self):
        return rf'{states.fmt_amplitude(self.amplitude)}|k = {u.uround(self.wavenumber, u.per_nm)} 1/nm, E = {u.uround(self.energy, u.eV)} eV>'

    @property
    def tex(self):
        return rf'{states.fmt_amplitude_for_tex(self.amplitude)}\phi_{{{u.uround(self.energy, u.eV)} \, \mathrm{{eV}}}}'

    @property
    def tex_ket(self):
        return rf'{states.fmt_amplitude_for_tex(self.amplitude)}\left| \phi_{{{u.uround(self.energy, u.eV)} \, \mathrm{{eV}}}} \right\rangle'

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Mass', utils.fmt_quantity(self.mass, utils.MASS_UNITS))
        info.add_field('Energy', utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))
        info.add_field('Wavenumber', utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS))

        return info


class QHOState(states.QuantumState):
    """A class representing a bound state of the quantum harmonic oscillator."""

    smallest_n = 0

    eigenvalues = states.Eigenvalues.DISCRETE
    binding = states.Binding.BOUND
    derivation = states.Derivation.ANALYTIC

    def __init__(
        self,
        spring_constant: float,
        mass: float = u.electron_mass,
        n: states.QuantumNumber = 0,
        amplitude: states.ProbabilityAmplitude = 1,
    ):
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

        super().__init__(amplitude = amplitude)

    @classmethod
    def from_omega_and_mass(
        cls,
        omega: float,
        mass: float = u.electron_mass,
        n: states.QuantumNumber = 0,
        amplitude: states.ProbabilityAmplitude = 1,
    ):
        """
        Construct a QHOState from an angular frequency, mass, and energy index n.

        :param omega: the fundamental angular frequency of the quantum harmonic oscillator
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        :return: a QHOState instance
        """
        return cls(spring_constant = mass * (omega ** 2), mass = mass, n = n, amplitude = amplitude)

    @classmethod
    def from_potential(
        cls,
        potential: potentials.HarmonicOscillator,
        mass: float,
        n: states.QuantumNumber = 0,
        amplitude: states.ProbabilityAmplitude = 1,
    ):
        """
        Construct a QHOState from a HarmonicOscillator, mass, and energy index n.

        :param potential: a HarmonicOscillator instance
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        :return: a QHOState instance
        """
        return cls(spring_constant = potential.spring_constant, mass = mass, n = n, amplitude = amplitude)

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

    @property
    def tuple(self):
        return self.n, self.mass, self.omega

    def __call__(self, x):
        """
        Evaluate the quantum harmonic oscillator bound state wavefunction at a point, or vectorized over an array of points.

        Warning: for large enough n (>= ~60) this will fail due to n! overflowing.

        :param x: the distance coordinate along the direction of confinement
        :return: the value(s) of the wavefunction at x
        """
        norm = ((self.mass * self.omega / (u.pi * u.hbar)) ** (1 / 4)) / (np.float64(2 ** (self.n / 2)) * np.sqrt(np.float64(spmisc.factorial(self.n))))
        exp = np.exp(-self.mass * self.omega * (x ** 2) / (2 * u.hbar))
        herm = special.hermite(self.n)(np.sqrt(self.mass * self.omega / u.hbar) * x)

        return self.amplitude * (norm * exp * herm).astype(np.complex128)

    def __repr__(self):
        return utils.fmt_fields(self, 'n', 'mass', 'energy', 'amplitude')

    @property
    def ket(self):
        return f'{states.fmt_amplitude(self.amplitude)}|{self.n}>'

    @property
    def tex(self):
        return rf'{states.fmt_amplitude_for_tex(self.amplitude)}\psi_{{{self.n}}}'

    @property
    def tex_ket(self):
        return rf'{states.fmt_amplitude_for_tex(self.amplitude)}\left| \psi_{{{self.n}}} \right\rangle'

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Mass', utils.fmt_quantity(self.mass, utils.MASS_UNITS))
        info.add_field('n', self.n)
        info.add_field('Energy', utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))
        info.add_field('Period', utils.fmt_quantity(self.period, utils.TIME_UNITS))

        return info


class FiniteSquareWellState(states.QuantumState):
    """A class representing a bound state of a finite square well."""

    smallest_n = 1

    eigenvalues = states.Eigenvalues.DISCRETE
    binding = states.Binding.BOUND
    derivation = states.Derivation.ANALYTIC

    def __init__(
        self,
        well_depth: float,
        well_width: float,
        mass: float,
        n: states.QuantumNumber = 1,
        well_center: float = 0,
        amplitude: states.ProbabilityAmplitude = 1,
    ):
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

        z_0 = (well_width / 2) * np.sqrt(2 * mass * well_depth) / u.hbar

        if n - 1 > z_0 // (u.pi / 2):
            raise exceptions.IllegalQuantumState('There is no bound state with the given parameters')

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
    def from_potential(
        cls,
        potential: potentials.FiniteSquareWell,
        mass: float,
        n: states.QuantumNumber = 1,
        amplitude: states.ProbabilityAmplitude = 1,
    ):
        """
        Construct a FiniteSquareWellState from a FiniteSquareWell potential, the particle mass, and an energy index.

        :param potential: a FiniteSquareWell potential.
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        :return: a FiniteSquareWellState instance
        """
        return cls(potential.potential_depth, potential.width, mass, n = n, well_center = potential.center, amplitude = amplitude)

    @classmethod
    def all_states_of_well_from_parameters(
        cls,
        well_depth: float,
        well_width: float,
        mass: float,
        well_center: float = 0,
        amplitude: states.ProbabilityAmplitude = 1,
    ):
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
    def all_states_of_well_from_well(
        cls,
        potential: potentials.FiniteSquareWell,
        mass: float,
        amplitude: states.ProbabilityAmplitude = 1,
    ):
        """
        Return a list containing all of the bound states of a well.

        :param potential: a FiniteSquareWell
        :param mass: the mass of the particle
        :param amplitude: the probability amplitude of the states
        :return:
        """
        return cls.all_states_of_well_from_parameters(
            potential.potential_depth,
            potential.width,
            mass,
            well_center = potential.center,
            amplitude = amplitude,
        )

    @property
    def left_edge(self):
        """Return the position of the left edge of the well."""
        return self.well_center - (self.well_width / 2)

    @property
    def right_edge(self):
        """Return the position of the right edge of the well."""
        return self.well_center + (self.well_width / 2)

    @property
    def tuple(self):
        return self.well_depth, self.well_width, self.mass, self.n

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

        psi = np.where(
            cond,
            self.normalization_factor_inside_well * self.function_inside_well(self.wavenumber_inside_well * x),
            self.normalization_factor_outside_well * np.exp(-self.wavenumber_outside_well * np.abs(x))
        ).astype(np.complex128)

        symmetrization = np.where(
            np.less_equal(x, self.left_edge),
            sym,
            1,
        )

        return psi * symmetrization

    def __repr__(self):
        return utils.fmt_fields(self, 'n', 'mass', 'well_depth', 'well_width', 'energy', 'amplitude')

    @property
    def ket(self):
        return f'{states.fmt_amplitude(self.amplitude)}|{self.n}>'

    @property
    def tex(self):
        return rf'{states.fmt_amplitude_for_tex(self.amplitude)}\psi_{{{self.n}}}'

    @property
    def tex_ket(self):
        return rf'{states.fmt_amplitude_for_tex(self.amplitude)}\left| \psi_{{{self.n}}} \right\rangle'

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Mass', utils.fmt_quantity(self.mass, utils.MASS_UNITS))
        info.add_field('n', self.n)
        info.add_field('Energy', utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))
        info.add_field('Well Width', utils.fmt_quantity(self.well_width, utils.LENGTH_UNITS))
        info.add_field('Well Depth', utils.fmt_quantity(self.well_depth, utils.ENERGY_UNITS))

        return info


class GaussianWellState(states.QuantumState):
    """A class representing a bound state of a finite square well."""

    smallest_n = 0

    eigenvalues = states.Eigenvalues.DISCRETE
    binding = states.Binding.BOUND
    derivation = states.Derivation.VARIATIONAL

    def __init__(
        self,
        well_depth: float,
        well_width: float,
        mass: float,
        n: states.QuantumNumber = 0,
        well_center: float = 0,
        amplitude: states.ProbabilityAmplitude = 1,
    ):
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

        if n > 0:
            logger.warning('Analytic GaussianWellStates are not available for n > 0, and n = 0 is only approximate!')

        max_n = np.ceil(2 * np.sqrt(2 * mass * self.well_depth / (u.pi * (u.hbar ** 2))) * well_width) + 0.5
        if n > max_n:
            raise exceptions.IllegalQuantumState('Bound state energy must be less than zero')

        self.width = optimize.newton(lambda w: ((w ** 4) / (((well_width ** 2) + (w ** 2)) ** 1.5)) - ((u.hbar ** 2) / (4 * mass * well_width * self.well_depth)), well_width)
        self.energy = -(well_width * self.well_depth / np.sqrt(((well_width ** 2) + (self.width ** 2))) + ((u.hbar ** 2) / (8 * mass * (self.width ** 2))))

        super().__init__(amplitude = amplitude)

    @classmethod
    def from_potential(
        cls,
        potential: potentials.GaussianPotential,
        mass: float,
        n: states.QuantumNumber = 0,
        amplitude: states.ProbabilityAmplitude = 1,
    ):
        """
        Construct a FiniteSquareWellState from a FiniteSquareWell potential, the particle mass, and an energy index.

        :param potential: a FiniteSquareWell potential.
        :param mass: the mass of the particle
        :param n: the energy index of the state
        :param amplitude: the probability amplitude of the state
        :param dimension_label: a label indicating which dimension the particle is confined in
        :return: a FiniteSquareWellState instance
        """
        return cls(potential.potential_extrema, potential.width, mass, n = n, well_center = potential.center, amplitude = amplitude)

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

    def __repr__(self):
        return utils.fmt_fields(self, 'n', 'mass', 'energy', 'amplitude')

    @property
    def ket(self):
        return f'{states.fmt_amplitude(self.amplitude)}|{self.n}>'

    @property
    def tex(self):
        return rf'{states.fmt_amplitude_for_tex(self.amplitude)}\psi_{{{self.n}}}'

    @property
    def tex_ket(self):
        return rf'{states.fmt_amplitude_for_tex(self.amplitude)}\left| \psi_{{{self.n}}} \right\rangle'

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Mass', utils.fmt_quantity(self.mass, utils.MASS_UNITS))
        info.add_field('n', self.n)
        info.add_field('Energy', utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))

        return info


class OneDSoftCoulombState(states.QuantumState):
    smallest_n = 1

    eigenvalues = states.Eigenvalues.DISCRETE
    binding = states.Binding.BOUND
    derivation = states.Derivation.ANALYTIC

    def __init__(self, n = 1, amplitude: states.ProbabilityAmplitude = 1):
        self.n = n

        super().__init__(amplitude = amplitude)

    @classmethod
    def from_potential(cls, potential, test_mass, n = 1, **kwargs):
        return cls(n = n)

    @property
    def tuple(self):
        return self.n,

    def __call__(self, z):
        raise NotImplementedError("haven't implemented analytic eigenstates for this potential yet")

    def __repr__(self):
        return utils.fmt_fields(self, 'n', 'amplitude')

    @property
    def ket(self):
        return f'{states.fmt_amplitude(self.amplitude)}|{self.n}>'

    @property
    def tex(self):
        return rf'{states.fmt_amplitude_for_tex(self.amplitude)}\psi_{{{self.n}}}'

    @property
    def tex_ket(self):
        return rf'{states.fmt_amplitude_for_tex(self.amplitude)}\left| \psi_{{{self.n}}} \right\rangle'

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('n', self.n)

        return info


class NumericOneDState(states.QuantumState):
    eigenvalues = states.Eigenvalues.CONTINUOUS
    derivation = states.Derivation.ANALYTIC

    def __init__(
        self,
        *,
        wavefunction: 'mesh.PsiVector',
        energy: float,
        corresponding_analytic_state: states.QuantumState,
        binding: states.Binding.FREE,
        amplitude: states.ProbabilityAmplitude = 1,
    ):
        self.wavefunction = wavefunction
        self.energy = energy
        self.corresponding_analytic_state = corresponding_analytic_state
        self.binding = binding

        super().__init__(amplitude = amplitude)

    @property
    def n(self):
        return self.corresponding_analytic_state.n

    @property
    def wavenumber(self):
        return self.corresponding_analytic_state.wavenumber

    @property
    def tuple(self):
        return self.corresponding_analytic_state.tuple

    def __call__(self, z):
        return self.wavefunction

    @property
    def ket(self):
        return self.corresponding_analytic_state.ket + '_n'

    @property
    def tex(self):
        return self.corresponding_analytic_state.tex + '^{(n)}'

    @property
    def tex_ket(self):
        return self.corresponding_analytic_state.tex_ket + '^{(n)}'

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Energy', utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))

        analytic_info = self.corresponding_analytic_state.info()
        analytic_info.header = f'Analytic State: {self.corresponding_analytic_state.__class__.__name__}'
        info.add_info(analytic_info)

        return info
