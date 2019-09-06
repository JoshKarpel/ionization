import logging
import functools
import abc
import warnings

import mpmath
import numpy as np
import scipy.special as special

import simulacra as si
import simulacra.units as u

from .. import core, mesh, exceptions, utils

from . import state

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ThreeDPlaneWave(state.QuantumState):
    """
    A plane wave in three dimensions.

    NB: A ``ThreeDPlaneWave`` with non-zero :math:`x` or :math:`y` wavenumbers is not compatible with certain meshes because it does not have azimuthal symmetry!
    Use with caution!
    """

    eigenvalues = state.Eigenvalues.CONTINUOUS
    binding = state.Binding.FREE
    derivation = state.Derivation.ANALYTIC

    def __init__(
        self,
        wavenumber_x: float,
        wavenumber_y: float,
        wavenumber_z: float,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """
        Parameters
        ----------
        wavenumber_x
            The :math:`x`-component of the state's wavevector.
        wavenumber_y
            The :math:`y`-component of the state's wavevector.
        wavenumber_z
            The :math:`z`-component of the state's wavevector.
        """
        if wavenumber_x != 0 or wavenumber_y != 0:
            warnings.warn(
                "ThreeDPlaneWave states with non-zero x or y wavenumbers are not compatible with certain meshes because they do not have azimuthal symmetry!"
            )

        self.wavenumber_x = wavenumber_x
        self.wavenumber_y = wavenumber_y
        self.wavenumber_z = wavenumber_z

        super().__init__(amplitude=amplitude)

    @property
    def wavenumber(self):
        return np.sqrt(
            self.wavenumber_x ** 2 + self.wavenumber_y ** 2 + self.wavenumber_z ** 2
        )

    @property
    def energy(self):
        return core.electron_energy_from_wavenumber(self.wavenumber)

    @property
    def tuple(self):
        return self.wavenumber_x, self.wavenumber_y, self.wavenumber_z

    def eval_x_y_z(self, x, y, z):
        return (
            np.exp(1j * self.wavenumber_x * x)
            * np.exp(1j * self.wavenumber_y * y)
            * np.exp(1j * self.wavenumber_z * z)
            / (np.sqrt(u.pi) ** 3)
        )

    def eval_r_theta_phi(self, r, theta, phi):
        x = r * np.cos(phi) * np.sin(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(theta)

        return self.eval_x_y_z(x, y, z)

    def __call__(self, r, theta, phi):
        return self.eval_r_theta_phi(r, theta, phi)

    def __repr__(self):
        return utils.make_repr(self, "wavenumber_x", "wavenumber_y", "wavenumber_z")

    @property
    def ket(self):
        return f"{state.fmt_amplitude(self.amplitude)}|kx={self.wavenumber_x / u.per_nm:.3f} 1/nm, ky={self.wavenumber_y / u.per_nm:.3f} 1/nm, kz={self.wavenumber_z / u.per_nm:.3f} 1/nm>"

    @property
    def tex(self):
        return fr"{state.fmt_amplitude_for_tex(self.amplitude)}\phi_{{k_x={self.wavenumber_x / u.per_nm:.3f} \, \mathrm{{nm^-1}}, k_y={self.wavenumber_y / u.per_nm:.3f} \, \mathrm{{nm^-1}}, k_x={self.wavenumber_z / u.per_nm:.3f} \, \mathrm{{nm^-1}}}}"

    @property
    def tex_ket(self):
        return fr"{state.fmt_amplitude_for_tex(self.amplitude)}\left| \phi_{{k_x={self.wavenumber_x / u.per_nm:.3f} \, \mathrm{{nm^-1}}, k_y={self.wavenumber_y / u.per_nm:.3f} \, \mathrm{{nm^-1}}, k_x={self.wavenumber_z / u.per_nm:.3f} \, \mathrm{{nm^-1}}}} \right\rangle"

    def info(self) -> si.Info:
        info = super().info()

        info.add_field(
            "X Wavenumber",
            utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS),
        )
        info.add_field(
            "Y Wavenumber",
            utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS),
        )
        info.add_field(
            "Z Wavenumber",
            utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS),
        )
        info.add_field("Energy", utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))
        info.add_field(
            "Wavenumber",
            utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS),
        )

        return info


class SphericalHarmonicState(state.QuantumState, abc.ABC):
    """
    An abstract class for any state that is just a radial function times a spherical harmonic.

    Attributes
    ----------
    l: :class:`int`
        The orbital angular momentum quantum number.
    m: :class:`int`
        The quantum number for the z-component of the angular momentum.
    """

    def __init__(
        self,
        l: state.QuantumNumber = 0,
        m: state.QuantumNumber = 0,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """

        Parameters
        ----------
        l
            The orbital angular momentum quantum number of the state.
        m
            The quantum number for the :math:`z`-component of the angular momentum of the state.
        """
        self.l = l
        self.m = m

        super().__init__(amplitude=amplitude)

    @property
    def l(self):
        return self._l

    @l.setter
    def l(self, l):
        if int(l) != l or l < 0:
            raise exceptions.IllegalQuantumState(
                f"l ({l}) must be an integer greater than or equal to zero and less than n ({self.n})"
            )

        self._l = int(l)

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, m):
        if int(m) != m or not -self.l <= m <= self.l:
            raise exceptions.IllegalQuantumState(
                f"|m| (|{m}|) must be an integer less than or equal to l ({self.l})"
            )

        self._m = int(m)

    @si.utils.cached_property
    def spherical_harmonic(self):
        return si.math.SphericalHarmonic(l=self.l, m=self.m)

    @abc.abstractmethod
    def radial_function(self, r):
        raise NotImplementedError

    def __call__(self, r, theta, phi):
        return (
            self.amplitude
            * self.radial_function(r)
            * self.spherical_harmonic(theta, phi)
        )


class FreeSphericalWave(SphericalHarmonicState):
    """
    A class that represents a free spherical wave.

    Attributes
    ----------
    energy: :class:`float`
        The energy of the state.
    wavenumber: :class:`float`
        The wavenumber of the state.
    """

    eigenvalues = state.Eigenvalues.CONTINUOUS
    binding = state.Binding.FREE
    derivation = state.Derivation.ANALYTIC

    def __init__(
        self,
        energy: float = 1 * u.eV,
        l: state.QuantumNumber = 0,
        m: state.QuantumNumber = 0,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """

        Parameters
        ----------
        energy
            The energy of the state.
        """
        super().__init__(l=l, m=m, amplitude=amplitude)

        self.energy = energy

    @classmethod
    def from_wavenumber(cls, k, l=0, m=0, amplitude: state.ProbabilityAmplitude = 1):
        energy = core.electron_energy_from_wavenumber(k)

        return cls(energy, l=l, m=m, amplitude=amplitude)

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        if energy < 0:
            raise exceptions.IllegalQuantumState("energy must be greater than zero")
        self._energy = energy

    @property
    def wavenumber(self):
        return core.electron_wavenumber_from_energy(self.energy)

    @property
    def tuple(self):
        return self.energy, self.wavenumber, self.l, self.m

    def radial_function(self, r):
        return np.sqrt(2 * (self.wavenumber ** 2) / u.pi) * special.spherical_jn(
            self.l, self.wavenumber * r
        )

    def __repr__(self):
        return utils.make_repr(self, "energy", "wavenumber", "l", "m")

    @property
    def ket(self):
        return f"{state.fmt_amplitude(self.amplitude)}|{self.energy / u.eV:.3f} eV,{self.l},{self.m}>"

    @property
    def tex(self):
        return rf"{state.fmt_amplitude_for_tex(self.amplitude)}\phi_{{{self.energy / u.eV:.3f} \, \mathrm{{eV}}, {self.l}, {self.m}}}"

    @property
    def tex_ket(self):
        return rf"{state.fmt_amplitude_for_tex(self.amplitude)}\left| \phi_{{{self.energy / u.eV:.3f} \, \mathrm{{eV}}, {self.l}, {self.m}}} \right\rangle"

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Energy", utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))
        info.add_field(
            "Wavenumber",
            utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS),
        )
        info.add_field("l", self.l)
        info.add_field("m", self.m)

        return info


class HydrogenBoundState(SphericalHarmonicState):
    """
    A class that represents a hydrogen bound state.

    Attributes
    ----------
    n: :class:`int`
        The principal quantum number.
    energy: :class:`float`
        The energy of the state.
    """

    eigenvalues = state.Eigenvalues.DISCRETE
    binding = state.Binding.BOUND
    derivation = state.Derivation.ANALYTIC

    def __init__(
        self,
        n: state.QuantumNumber = 1,
        l: state.QuantumNumber = 0,
        m: state.QuantumNumber = 0,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """
        Parameters
        ----------
        n
            The principal quantum number.
        """
        self.n = n
        super().__init__(l=l, m=m, amplitude=amplitude)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        if int(n) != n or n < 0:
            raise exceptions.IllegalQuantumState(
                f"n ({n}) must be an integer greater than zero"
            )

        self._n = int(n)

    @SphericalHarmonicState.l.setter
    def l(self, l):
        if int(l) != l or not 0 <= l < self.n:
            raise exceptions.IllegalQuantumState(
                f"l ({l}) must be an integer greater than or equal to zero and less than n ({self.n})"
            )

        self._l = int(l)

    @property
    def energy(self) -> float:
        return -u.rydberg / (self.n ** 2)

    @property
    def tuple(self):
        return self.n, self.l, self.m

    def __repr__(self):
        return utils.make_repr(self, "n", "l", "m", "amplitude")

    @property
    def ket(self):
        return f"{state.fmt_amplitude(self.amplitude)}|{self.n},{self.l},{self.m}>"

    @property
    def tex(self):
        return rf"{utils.complex_j_to_i(state.fmt_amplitude_for_tex(self.amplitude))}\psi_{{{self.n}, {self.l}, {self.m}}}"

    @property
    def tex_ket(self):
        return rf"{utils.complex_j_to_i(state.fmt_amplitude_for_tex(self.amplitude))}\left| \psi_{{{self.n}, {self.l}, {self.m}}} \right\rangle"

    def radial_function(self, r: float):
        """Return the radial part of the wavefunction, R, evaluated at r."""
        normalization = np.sqrt(
            ((2 / (self.n * u.bohr_radius)) ** 3)
            * (
                special.factorial(self.n - self.l - 1)
                / (2 * self.n * special.factorial(self.n + self.l))
            )
        )  # Griffith's normalization
        r_dep = np.exp(-r / (self.n * u.bohr_radius)) * (
            (2 * r / (self.n * u.bohr_radius)) ** self.l
        )
        lag_poly = special.eval_genlaguerre(
            self.n - self.l - 1, (2 * self.l) + 1, 2 * r / (self.n * u.bohr_radius)
        )

        return self.amplitude * normalization * r_dep * lag_poly

    def info(self):
        info = super().info()

        info.add_field("n", self.n)
        info.add_field("l", self.l)
        info.add_field("m", self.m)

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
    """
    A class that represents a hydrogen free state.

    Attributes
    ----------
    energy: :class:`float`
        The energy of the state.
    wavenumber: :class:`float`
        The wavenumber of the state.
    """

    eigenvalues = state.Eigenvalues.CONTINUOUS
    binding = state.Binding.FREE
    derivation = state.Derivation.ANALYTIC

    def __init__(
        self,
        energy: float = 1 * u.eV,
        l: state.QuantumNumber = 0,
        m: state.QuantumNumber = 0,
        amplitude: state.ProbabilityAmplitude = 1.0,
    ):
        """
        Parameters
        ----------
        energy
            The energy of the state.
        """
        self.energy = energy
        super().__init__(l=l, m=m, amplitude=amplitude)

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        if energy < 0:
            raise exceptions.IllegalQuantumState("energy must be greater than zero")
        self._energy = energy

    @property
    def wavenumber(self):
        return core.electron_wavenumber_from_energy(self.energy)

    @classmethod
    def from_wavenumber(cls, k, l=0, m=0):
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
            hgf = functools.partial(
                mpmath.hyp1f1, a, b
            )  # construct a partial function, with a and b filled in
            hgf = np.vectorize(hgf, otypes=[np.complex128])  # vectorize using numpy

            A = (
                (kappa ** (-((2 * self.l) + 1)))
                * special.gamma(1 + self.l + kappa)
                / special.gamma(kappa - self.l)
            )
            B = A / (1 - np.exp(-u.twopi / np.sqrt(epsilon)))
            s_prefactor = np.sqrt(B / 2)

            l_prefactor = (2 ** (self.l + 1)) / special.factorial((2 * self.l) + 1)

            prefactor = s_prefactor * l_prefactor * unit_prefactor

            return (
                self.amplitude
                * prefactor
                * hgf(2 * x / kappa)
                * (x ** (self.l + 1))
                * np.exp(-x / kappa)
                / r
            )

        elif epsilon == 0:
            bessel_order = (2 * self.l) + 1
            prefactor = unit_prefactor
            bessel = functools.partial(special.jv, bessel_order)

            return self.amplitude * prefactor * bessel(np.sqrt(8 * x)) * np.sqrt(x) / r

    def __repr__(self):
        return utils.make_repr(self, "energy", "wavenumber", "l", "m", "amplitude")

    @property
    def ket(self):
        return f"{state.fmt_amplitude(self.amplitude)}|{self.energy / u.eV:.3f} eV,{self.l},{self.m}>"

    @property
    def tex(self):
        return rf"{state.fmt_amplitude_for_tex(self.amplitude)}\phi_{{{self.energy / u.eV:.3f} \, \mathrm{{eV}}, {self.l}, {self.m}}}"

    @property
    def tex_ket(self):
        return rf"{state.fmt_amplitude_for_tex(self.amplitude)}\left| \phi_{{{self.energy / u.eV:.3f} \, \mathrm{{eV}}, {self.l}, {self.m}}} \right\rangle"

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Energy", utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))
        info.add_field(
            "Wavenumber",
            utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS),
        )
        info.add_field("l", self.l)
        info.add_field("m", self.m)

        return info


class NumericSphericalHarmonicState(SphericalHarmonicState):
    """
    A numerically-derived spherical-harmonic-radial quantum state.

    Attributes
    ----------
    corresponding_analytic_state: :class:`QuantumState`
        The analytic state that this numeric state nominally approximates.
    n: :class:`int`
        The principal quantum number of a bound state.
        Will raise an exception if the corresponding analytic state is not bound.
    wavenumber: :class:`float`
        The wavenumber of a free state.
        Will raise an exception if the corresponding analytic state is not free.
    """

    eigenvalues = state.Eigenvalues.DISCRETE
    derivation = state.Derivation.NUMERIC

    def __init__(
        self,
        *,
        g: "mesh.PsiVector",
        l: state.QuantumNumber = 0,
        m: state.QuantumNumber = 0,
        energy: float,
        corresponding_analytic_state: state.QuantumState,
        binding: state.Binding,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """
        Parameters
        ----------
        g
            The numerically-determined wavefunction as a function of the radial coordinate.
        energy
            The numerically-determined energy of the state.
        corresponding_analytic_state
            The analytic state that this state approximates.
        binding
            Whether the state is bound or free.
        """
        self.g = g
        self.energy = energy
        self.corresponding_analytic_state = corresponding_analytic_state
        self.binding = binding

        super().__init__(l=l, m=m, amplitude=amplitude)

    @property
    def n(self):
        return self.corresponding_analytic_state.n

    @property
    def wavenumber(self):
        return self.corresponding_analytic_state.wavenumber

    @property
    def tuple(self):
        return self.corresponding_analytic_state.tuple

    def radial_function(self, r):
        return self.g

    def __repr__(self):
        return repr(self.corresponding_analytic_state) + "_n"

    @property
    def ket(self):
        return self.corresponding_analytic_state.ket + "_n"

    @property
    def tex(self):
        return self.corresponding_analytic_state.tex + "^{(n)}"

    @property
    def tex_ket(self):
        return self.corresponding_analytic_state.tex_ket + "^{(n)}"

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Energy", utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))

        analytic_info = self.corresponding_analytic_state.info()
        analytic_info.header = (
            f"Analytic State: {self.corresponding_analytic_state.__class__.__name__}"
        )
        info.add_info(analytic_info)

        return info
