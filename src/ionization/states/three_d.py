import logging
import functools
import abc
import warnings
from typing import Union

import mpmath
import numpy as np
import scipy as sp
import scipy.special as special

import simulacra as si
import simulacra.units as u

from .. import core, mesh, exceptions, utils

from . import states

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ThreeDPlaneWave(states.QuantumState):
    def __init__(self, wavenumber_x, wavenumber_y, wavenumber_z, amplitude: states.ProbabilityAmplitude = 1):
        if wavenumber_x != 0 or wavenumber_y != 0:
            warnings.warn('ThreeDPlaneWave states with non-zero x or y wavenumbers are not compatible with certain meshes because they do not have azimuthal symmetry!')

        self.wavenumber_x = wavenumber_x
        self.wavenumber_y = wavenumber_y
        self.wavenumber_z = wavenumber_z

        super().__init__(amplitude = amplitude)

    @property
    def wavenumber(self):
        return np.sqrt(self.wavenumber_x ** 2 + self.wavenumber_y ** 2 + self.wavenumber_z ** 2)

    @property
    def energy(self):
        return core.electron_energy_from_wavenumber(self.wavenumber)

    @property
    def tuple(self):
        return self.wavenumber_x, self.wavenumber_y, self.wavenumber_z

    def eval_x_y_z(self, x, y, z):
        return np.exp(1j * self.wavenumber_x * x) * np.exp(1j * self.wavenumber_y * y) * np.exp(1j * self.wavenumber_z * z) / (np.sqrt(u.pi) ** 3)

    def eval_r_theta_phi(self, r, theta, phi):
        x = r * np.cos(phi) * np.sin(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(theta)

        return self.eval_x_y_z(x, y, z)

    def __call__(self, r, theta, phi):
        return self.eval_r_theta_phi(r, theta, phi)

    def __repr__(self):
        return utils.fmt_fields(self, 'wavenumber_x', 'wavenumber_y', 'wavenumber_z')

    @property
    def ket(self):
        return f'{states.fmt_amplitude(self.amplitude)}|kx={u.uround(self.wavenumber_x, u.per_nm)} 1/nm, ky={u.uround(self.wavenumber_y, u.per_nm)} 1/nm, kz={u.uround(self.wavenumber_z, u.per_nm)} 1/nm>'

    @property
    def tex(self):
        return fr'{states.fmt_amplitude_for_tex(self.amplitude)}\phi_{{k_x={u.uround(self.wavenumber_x, u.per_nm)} \, \mathrm{{nm^-1}}, k_y={u.uround(self.wavenumber_y, u.per_nm)} \, \mathrm{{nm^-1}}, k_x={u.uround(self.wavenumber_z, u.per_nm)} \, \mathrm{{nm^-1}}}}'

    @property
    def tex_ket(self):
        return fr'{states.fmt_amplitude_for_tex(self.amplitude)}\left| \phi_{{k_x={u.uround(self.wavenumber_x, u.per_nm)} \, \mathrm{{nm^-1}}, k_y={u.uround(self.wavenumber_y, u.per_nm)} \, \mathrm{{nm^-1}}, k_x={u.uround(self.wavenumber_z, u.per_nm)} \, \mathrm{{nm^-1}}}} \right\rangle'

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('X Wavenumber', utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS))
        info.add_field('Y Wavenumber', utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS))
        info.add_field('Z Wavenumber', utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS))
        info.add_field('Energy', utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))
        info.add_field('Wavenumber', utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS))

        return info


class SphericalHarmonicState(states.QuantumState, abc.ABC):
    def __init__(self, l: int = 0, m: int = 0, amplitude: states.ProbabilityAmplitude = 1):
        self.l = l
        self.m = m

        super().__init__(amplitude = amplitude)

    @property
    def l(self):
        return self._l

    @l.setter
    def l(self, l):
        if int(l) != l or l < 0:
            raise exceptions.IllegalQuantumState('l ({}) must be an integer greater than or equal to zero and less than n ({})'.format(l, self.n))

        self._l = int(l)

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, m):
        if int(m) != m or not -self.l <= m <= self.l:
            raise exceptions.IllegalQuantumState('|m| (|{}|) must be an integer less than or equal to l ({})'.format(m, self.l))

        self._m = int(m)

    @si.utils.cached_property
    def spherical_harmonic(self):
        """Return the SphericalHarmonic for the state's angular momentum quantum numbers."""
        return si.math.SphericalHarmonic(l = self.l, m = self.m)


class FreeSphericalWave(SphericalHarmonicState):
    """A class that represents a free spherical wave."""

    eigenvalues = states.Eigenvalues.CONTINUOUS
    binding = states.Binding.FREE
    derivation = states.Derivation.ANALYTIC

    def __init__(self, energy: float = 1 * u.eV, l: int = 0, m: int = 0, amplitude: states.ProbabilityAmplitude = 1):
        """

        Parameters
        ----------
        energy
            The energy of the state.
        l
            The orbital angular momentum quantum number.
        m
            The quantum number for the z-component of the angular momentum.
        amplitude
            The probability amplitude of the state.
        """
        super().__init__(l = l, m = m, amplitude = amplitude)

        self.energy = energy

    @classmethod
    def from_wavenumber(cls, k, l = 0, m = 0, amplitude: states.ProbabilityAmplitude = 1):
        energy = core.electron_energy_from_wavenumber(k)

        return cls(energy, l = l, m = m, amplitude = amplitude)

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        if energy < 0:
            raise exceptions.IllegalQuantumState('energy must be greater than zero')
        self._energy = energy

    @property
    def wavenumber(self):
        return core.electron_wavenumber_from_energy(self.energy)

    @property
    def tuple(self):
        return self.energy, self.wavenumber, self.l, self.m

    def radial_function(self, r):
        return np.sqrt(2 * (self.wavenumber ** 2) / u.pi) * special.spherical_jn(self.l, self.wavenumber * r)

    def __call__(self, r, theta, phi):
        """
        Evaluate the free spherical wave wavefunction at a point, or vectorized over an array of points.

        :param r: radial coordinate
        :param theta: polar coordinate
        :param phi: azimuthal coordinate
        :return: the value(s) of the wavefunction at (r, theta, phi)
        """
        return self.amplitude * self.radial_function(r) * self.spherical_harmonic(theta, phi)

    def __repr__(self):
        return utils.fmt_fields(self, 'energy', 'wavenumber', 'l', 'm')

    @property
    def ket(self):
        return f'{states.fmt_amplitude(self.amplitude)}|{u.uround(self.energy, u.eV)} eV,{self.l},{self.m}>'

    @property
    def tex(self):
        return rf'{states.fmt_amplitude_for_tex(self.amplitude)}\phi_{{{u.uround(self.energy, u.eV)} \, \mathrm{{eV}}, {self.l}, {self.m}}}'

    @property
    def tex_ket(self):
        return rf'{states.fmt_amplitude_for_tex(self.amplitude)}\left| \phi_{{{u.uround(self.energy, u.eV)} \, \mathrm{{eV}}, {self.l}, {self.m}}} \right\rangle'

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Energy', utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))
        info.add_field('Wavenumber', utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS))
        info.add_field('l', self.l)
        info.add_field('m', self.m)

        return info


class HydrogenBoundState(SphericalHarmonicState):
    """A class that represents a hydrogen bound state."""

    eigenvalues = states.Eigenvalues.DISCRETE
    binding = states.Binding.BOUND
    derivation = states.Derivation.ANALYTIC

    def __init__(self, n: int = 1, l: int = 0, m: int = 0, amplitude: states.ProbabilityAmplitude = 1):
        """
        Parameters
        ----------
        n
            The principal quantum number.
        l
            The orbital angular momentum quantum number.
        m
            The quantum number for the z-component of the angular momentum.
        amplitude
            The probability amplitude of the state.
        """
        self.n = n
        super().__init__(l = l, m = m, amplitude = amplitude)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        if int(n) != n or n < 0:
            raise exceptions.IllegalQuantumState(f'n ({n}) must be an integer greater than zero')

        self._n = int(n)

    @SphericalHarmonicState.l.setter
    def l(self, l):
        if int(l) != l or not 0 <= l < self.n:
            raise exceptions.IllegalQuantumState(f'l ({l}) must be an integer greater than or equal to zero and less than n ({self.n})')

        self._l = int(l)

    @property
    def energy(self) -> float:
        return -u.rydberg / (self.n ** 2)

    @property
    def tuple(self):
        return self.n, self.l, self.m

    def __repr__(self):
        return utils.fmt_fields(self, 'n', 'l', 'm', 'amplitude')

    @property
    def ket(self):
        return f'{states.fmt_amplitude(self.amplitude)}|{self.n},{self.l},{self.m}>'

    @property
    def tex(self):
        return rf'{utils.complex_j_to_i(states.fmt_amplitude_for_tex(self.amplitude))}\psi_{{{self.n}, {self.l}, {self.m}}}'

    @property
    def tex_ket(self):
        return rf'{utils.complex_j_to_i(states.fmt_amplitude_for_tex(self.amplitude))}\left| \psi_{{{self.n}, {self.l}, {self.m}}} \right\rangle'

    def radial_function(self, r: float):
        """Return the radial part of the wavefunction, R, evaluated at r."""
        normalization = np.sqrt(((2 / (self.n * u.bohr_radius)) ** 3) * (sp.math.factorial(self.n - self.l - 1) / (2 * self.n * sp.math.factorial(self.n + self.l))))  # Griffith's normalization
        r_dep = np.exp(-r / (self.n * u.bohr_radius)) * ((2 * r / (self.n * u.bohr_radius)) ** self.l)
        lag_poly = special.eval_genlaguerre(self.n - self.l - 1, (2 * self.l) + 1, 2 * r / (self.n * u.bohr_radius))

        return self.amplitude * normalization * r_dep * lag_poly

    def __call__(self, r: float, theta: float, phi: float) -> Union['mesh.PsiVector', 'mesh.PsiMesh']:
        """
        Evaluate the wavefunction at a point, or vectorized over an array of points.

        Parameters
        ----------
        r
        theta
        phi

        Returns
        -------

        """
        return self.radial_function(r) * self.spherical_harmonic(theta, phi)

    def info(self):
        info = super().info()

        info.add_field('n', self.n)
        info.add_field('l', self.l)
        info.add_field('m', self.m)

        return info


def coulomb_phase_shift(l, k):
    """

    :param l: angular momentum quantum number
    :param k: wavenumber
    :return:
    """
    gamma = 1j / (k * u.bohr_radius)
    return np.angle(special.gamma(1 + l + gamma))


class HydrogenCoulombState(SphericalHarmonicState):
    """A class that represents a hydrogenic free state."""

    eigenvalues = states.Eigenvalues.CONTINUOUS
    binding = states.Binding.FREE
    derivation = states.Derivation.ANALYTIC

    def __init__(self, energy: float = 1 * u.eV, l: int = 0, m: int = 0, amplitude: states.ProbabilityAmplitude = 1):
        """
        Parameters
        ----------
        energy
            The energy of the state.
        l
            The orbital angular momentum quantum number.
        m
            The quantum number for the z-component of the angular momentum.
        amplitude
            The probability amplitude of the state.
        """
        self.energy = energy
        super().__init__(l = l, m = m, amplitude = amplitude)

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        if energy < 0:
            raise exceptions.IllegalQuantumState('energy must be greater than zero')
        self._energy = energy

    @property
    def wavenumber(self):
        return core.electron_wavenumber_from_energy(self.energy)

    @classmethod
    def from_wavenumber(cls, k, l = 0, m = 0):
        """Construct a HydrogenCoulombState from its wavenumber and angular momentum quantum numbers."""
        energy = core.electron_energy_from_wavenumber(k)

        return cls(energy, l, m)

    @property
    def tuple(self):
        return self.wavenumber, self.l, self.m, self.energy

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
            bessel = functools.partial(special.jv, bessel_order)

            return self.amplitude * prefactor * bessel(np.sqrt(8 * x)) * np.sqrt(x) / r

    def __call__(self, r, theta, phi):
        return self.radial_function(r) * self.spherical_harmonic(theta, phi)

    def __repr__(self):
        return utils.fmt_fields(self, 'energy', 'wavenumber', 'l', 'm', 'amplitude')

    @property
    def ket(self):
        return f'{states.fmt_amplitude(self.amplitude)}|{u.uround(self.energy, u.eV)} eV,{self.l},{self.m}>'

    @property
    def tex(self):
        return rf'{states.fmt_amplitude_for_tex(self.amplitude)}\phi_{{{u.uround(self.energy, u.eV)} \, \mathrm{{eV}}, {self.l}, {self.m}}}'

    @property
    def tex_ket(self):
        return rf'{states.fmt_amplitude_for_tex(self.amplitude)}\left| \phi_{{{u.uround(self.energy, u.eV)} \, \mathrm{{eV}}, {self.l}, {self.m}}} \right\rangle'

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Energy', utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))
        info.add_field('Wavenumber', utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS))
        info.add_field('l', self.l)
        info.add_field('m', self.m)

        return info


class NumericSphericalHarmonicState(SphericalHarmonicState):
    eigenvalues = states.Eigenvalues.DISCRETE
    derivation = states.Derivation.NUMERIC

    def __init__(
            self,
            *,
            radial_wavefunction: 'mesh.PsiVector',
            l: int = 0,
            m: int = 0,
            energy: float,
            corresponding_analytic_state: states.QuantumState,
            binding: states.Binding.FREE,
            amplitude: states.ProbabilityAmplitude = 1):
        self.radial_wavefunction = radial_wavefunction
        self.energy = energy
        self.corresponding_analytic_state = corresponding_analytic_state
        self.binding = binding

        super().__init__(l = l, m = m, amplitude = amplitude)

    @property
    def n(self):
        return self.corresponding_analytic_state.n

    @property
    def k(self):
        return self.corresponding_analytic_state.wavenumber

    @property
    def tuple(self):
        return self.corresponding_analytic_state.tuple

    def radial_function(self, r):
        return self.radial_wavefunction

    def __call__(self, r, theta, phi):
        return self.radial_function(r) * self.spherical_harmonic(theta, phi)

    def __repr__(self):
        return repr(self.corresponding_analytic_state) + '_n'

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
