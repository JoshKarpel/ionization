import logging

import numpy as np
import scipy.special as special

import simulacra as si
import simulacra.units as u

from .. import core, potentials, utils, exceptions

from . import state

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TwoDPlaneWave(state.QuantumState):
    """
    A plane wave in three dimensions.
    """

    eigenvalues = state.Eigenvalues.CONTINUOUS
    binding = state.Binding.FREE
    derivation = state.Derivation.ANALYTIC

    def __init__(
        self,
        wavenumber_x: float,
        wavenumber_z: float,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """
        Parameters
        ----------
        wavenumber_x
            The :math:`x`-component of the state's wavevector.
        wavenumber_z
            The :math:`z`-component of the state's wavevector.
        """
        self.wavenumber_x = wavenumber_x
        self.wavenumber_z = wavenumber_z

        super().__init__(amplitude=amplitude)

    @property
    def wavenumber(self):
        return np.sqrt(self.wavenumber_x ** 2 + self.wavenumber_z ** 2)

    @property
    def energy(self):
        return core.electron_energy_from_wavenumber(self.wavenumber)

    @property
    def tuple(self):
        return self.wavenumber_x, self.wavenumber_z

    def eval_z_x(self, z, x):
        return (
            np.exp(1j * self.wavenumber_x * x)
            * np.exp(1j * self.wavenumber_z * z)
            / u.pi
        )

    def eval_r_theta(self, r, theta):
        z = r * np.cos(theta)
        x = r * np.sin(theta)

        return self.eval_z_x(z, x)

    def __call__(self, r, theta):
        return self.eval_r_theta(r, theta)

    def __repr__(self):
        return utils.fmt_fields(self, "wavenumber_x", "wavenumber_y", "wavenumber_z")

    @property
    def ket(self):
        return f"{state.fmt_amplitude(self.amplitude)}|kx={u.uround(self.wavenumber_x, u.per_nm)} 1/nm, kz={u.uround(self.wavenumber_z, u.per_nm)} 1/nm>"

    @property
    def tex(self):
        return fr"{state.fmt_amplitude_for_tex(self.amplitude)}\phi_{{k_x={u.uround(self.wavenumber_x, u.per_nm)} \, \mathrm{{nm^-1}}, k_x={u.uround(self.wavenumber_z, u.per_nm)} \, \mathrm{{nm^-1}}}}"

    @property
    def tex_ket(self):
        return fr"{state.fmt_amplitude_for_tex(self.amplitude)}\left| \phi_{{k_x={u.uround(self.wavenumber_x, u.per_nm)} \, \mathrm{{nm^-1}}, k_x={u.uround(self.wavenumber_z, u.per_nm)} \, \mathrm{{nm^-1}}}} \right\rangle"

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
        info.add_field("Energy", utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))
        info.add_field(
            "Wavenumber",
            utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS),
        )

        return info


class TwoDQuantumHarmonicOscillator(state.QuantumState):
    eigenvalues = state.Eigenvalues.DISCRETE
    binding = state.Binding.BOUND
    derivation = state.Derivation.ANALYTIC

    def __init__(
        self,
        spring_constant_z: float,
        spring_constant_x: float,
        mass: float = u.electron_mass,
        n_z: state.QuantumNumber = 0,
        n_x: state.QuantumNumber = 0,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        if spring_constant_z <= 0:
            raise exceptions.InvalidQuantumStateParameter(
                "spring_constant_z must be positive"
            )
        if spring_constant_x <= 0:
            raise exceptions.InvalidQuantumStateParameter(
                "spring_constant_x must be positive"
            )
        self.spring_constant_z = spring_constant_z
        self.spring_constant_x = spring_constant_x

        self.mass = mass

        if n_z < 0:
            raise exceptions.InvalidQuantumStateParameter("n_z must be >= 0")
        if n_x < 0:
            raise exceptions.InvalidQuantumStateParameter("n_x must be >= 0")
        self.n_z = n_z
        self.n_x = n_x

        super().__init__(amplitude=amplitude)

    @classmethod
    def from_potential(
        cls,
        potential: potentials.HarmonicOscillator,
        mass: float,
        n_z: state.QuantumNumber = 0,
        n_x: state.QuantumNumber = 0,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """

        Parameters
        ----------
        potential
            A harmonic oscillator potential.
        mass
            The mass of the particle.
        n_z
            The energy quantum number of the state for the z direction.
        n_x
            The energy quantum number of the state for the x direction.

        Returns
        -------
        state
            The state constructed from the parameters.
        """
        try:
            spring_constant_z = potential.spring_constant_z
            spring_constant_x = potential.spring_constant_x
        except AttributeError:
            spring_constant_z = spring_constant_x = potential.spring_constant

        return cls(
            spring_constant_z=spring_constant_z,
            spring_constant_x=spring_constant_x,
            mass=mass,
            n_z=n_z,
            n_x=n_x,
            amplitude=amplitude,
        )

    @property
    def omega_z(self):
        return np.sqrt(self.spring_constant_z / self.mass)

    @property
    def omega_x(self):
        return np.sqrt(self.spring_constant_x / self.mass)

    @property
    def ksi_z(self):
        return np.sqrt(u.electron_mass * self.omega_z / u.hbar)

    @property
    def ksi_x(self):
        return np.sqrt(u.electron_mass * self.omega_x / u.hbar)

    @property
    def energy(self):
        return u.hbar * ((self.omega_z * self.n_z) + (self.omega_x + self.n_x) + 1)

    @property
    def tuple(self):
        return self.n_z, self.n_x, self.mass, self.omega_z, self.omega_x

    def eval_z_x(self, z, x):
        kz = self.ksi_z * z
        kx = self.ksi_x * x
        norm = np.sqrt(self.ksi_z * self.ksi_x) / np.sqrt(
            (2 ** (self.n_z + self.n_x))
            * special.factorial(self.n_z)
            * special.factorial(self.n_x)
            * u.pi
        )
        gaussian = np.exp(-((kz ** 2) + (kx ** 2)) / 2)
        hermite = special.hermite(self.n_z)(kz) * special.hermite(self.n_x)(kx)

        return (norm * gaussian * hermite).astype(np.complex128)

    def eval_r_theta(self, r, theta):
        z = r * np.cos(theta)
        x = r * np.sin(theta)

        return self.eval_z_x(z, x)

    def __call__(self, r, theta):
        return self.eval_r_theta(r, theta)

    def __repr__(self):
        return utils.fmt_fields(self, "n_z", "n_x")

    @property
    def ket(self):
        return f"{state.fmt_amplitude(self.amplitude)}|{self.n_z}, {self.n_x}>"

    @property
    def tex(self):
        return fr"{state.fmt_amplitude_for_tex(self.amplitude)}\psi_{{n_z={self.n_z}, \, n_x={self.n_x}}}"

    @property
    def tex_ket(self):
        return fr"{state.fmt_amplitude_for_tex(self.amplitude)}\left| \psi_{{n_z={self.n_z}, \, n_x={self.n_x}}} \right\rangle"

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Mass", utils.fmt_quantity(self.mass, utils.MASS_UNITS))
        info.add_field("n_z", self.n_z)
        info.add_field("n_x", self.n_x)
        info.add_field("Energy", utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))

        return info


class TwoDGaussianWavepacket(state.QuantumState):
    eigenvalues = state.Eigenvalues.CONTINUOUS
    binding = state.Binding.FREE
    derivation = state.Derivation.ANALYTIC

    def __init__(
        self,
        center_z=0,
        center_x=0,
        width_z=1 * u.nm,
        width_x=1 * u.nm,
        k_z=0,
        k_x=0,
        mass: float = u.electron_mass,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        self.center_z = center_z
        self.center_x = center_x

        if width_z <= 0:
            raise exceptions.InvalidQuantumStateParameter("width_z must be positive")
        if width_x <= 0:
            raise exceptions.InvalidQuantumStateParameter("width_x must be positive")

        self.width_z = width_z
        self.width_x = width_x

        self.k_z = k_z
        self.k_x = k_x

        self.mass = mass

        super().__init__(amplitude=amplitude)

    @property
    def tuple(self):
        return self.center_z, self.center_x, self.width_z, self.width_x

    def eval_z_x(self, z, x):
        centered_z = z - self.center_z
        centered_x = x - self.center_x
        norm = 1 / (np.sqrt(u.twopi) * np.sqrt(self.width_z * self.width_x))
        gaussian = np.exp(-0.5 * ((centered_z / self.width_z) ** 2)) * np.exp(
            -0.5 * ((centered_x / self.width_x) ** 2)
        )
        motion = np.exp(1j * self.k_z * z) * np.exp(1j * self.k_x * x)

        return norm * gaussian * motion

    def eval_r_theta(self, r, theta):
        z = r * np.cos(theta)
        x = r * np.sin(theta)

        return self.eval_z_x(z, x)

    def __call__(self, r, theta):
        return self.eval_r_theta(r, theta)

    def __repr__(self):
        return utils.fmt_fields(
            self, "center_z", "center_x", "width_z", "width_x", "k_z", "k_x"
        )

    @property
    def ket(self):
        return f"{state.fmt_amplitude(self.amplitude)}|Wavepacket>"

    @property
    def tex(self):
        return fr"{state.fmt_amplitude_for_tex(self.amplitude)}\phi"

    @property
    def tex_ket(self):
        return (
            fr"{state.fmt_amplitude_for_tex(self.amplitude)}\left| \phi \right\rangle"
        )

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Mass", utils.fmt_quantity(self.mass, utils.MASS_UNITS))
        info.add_field(
            "Z Center", utils.fmt_quantity(self.center_z, utils.LENGTH_UNITS)
        )
        info.add_field(
            "X Center", utils.fmt_quantity(self.center_x, utils.LENGTH_UNITS)
        )
        info.add_field("Z Width", utils.fmt_quantity(self.width_z, utils.LENGTH_UNITS))
        info.add_field("X Width", utils.fmt_quantity(self.width_x, utils.LENGTH_UNITS))
        info.add_field(
            "Z Wavenumber", utils.fmt_quantity(self.k_z, utils.INVERSE_LENGTH_UNITS)
        )
        info.add_field(
            "X Wavenumber", utils.fmt_quantity(self.k_x, utils.INVERSE_LENGTH_UNITS)
        )

        return info
