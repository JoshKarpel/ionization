import logging
import itertools
from typing import List

import numpy as np
import scipy.optimize as optimize
import scipy.special as special
import scipy.misc as spmisc

import simulacra as si
import simulacra.units as u

from .. import mesh, potentials, utils, exceptions

from . import state

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OneDPlaneWave(state.QuantumState):
    """
    A class representing a free particle in one dimension.

    Attributes
    ----------
    wavenumber: :class:`float`
        The wavenumber of the state.
    energy: :class:`float`
        The energy of the state.
    momentum: :class:`float`
        The momentum of the state.
    """

    eigenvalues = state.Eigenvalues.CONTINUOUS
    binding = state.Binding.FREE
    derivation = state.Derivation.ANALYTIC

    def __init__(
        self,
        wavenumber: float = u.twopi / u.nm,
        mass: float = u.electron_mass,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """
        Parameters
        ----------
        wavenumber
            The wavenumber of the plane wave.
        mass
            The mass of the particle.
        """
        self.wavenumber = wavenumber
        self.mass = mass

        super().__init__(amplitude=amplitude)

    @classmethod
    def from_energy(
        cls,
        energy: float = 1.50412 * u.eV,
        k_sign: int = 1,
        mass: float = u.electron_mass,
        amplitude: state.ProbabilityAmplitude = 1,
    ) -> "OneDPlaneWave":
        """
        Parameters
        ----------
        energy: :class:`float`
            The energy of the state.
        k_sign: {1, -1}
            The desired sign of the momentum.
        mass: :class:`float`
            The mass of the particle.

        Returns
        -------
        state
            The state constructed from the parameters.
        """
        if k_sign not in (1, -1):
            raise exceptions.IllegalQuantumState("k_sign must be +1 or -1")
        return cls(
            k_sign * np.sqrt(2 * mass * energy) / u.hbar, mass, amplitude=amplitude
        )

    @property
    def energy(self) -> float:
        return ((u.hbar * self.wavenumber) ** 2) / (2 * self.mass)

    @property
    def momentum(self) -> float:
        return u.hbar * self.wavenumber

    @property
    def tuple(self):
        return self.wavenumber, self.mass

    def __call__(self, x):
        """Evaluate the wavefunction."""
        return np.exp(1j * self.wavenumber * x) / np.sqrt(u.twopi)

    def __repr__(self):
        return utils.fmt_fields(self, "wavenumber", "energy", "mass", "amplitude")

    @property
    def ket(self):
        return rf"{state.fmt_amplitude(self.amplitude)}|wavenumber = {u.uround(self.wavenumber, u.per_nm)} 1/nm, E = {u.uround(self.energy, u.eV)} eV>"

    @property
    def tex(self):
        return rf"{state.fmt_amplitude_for_tex(self.amplitude)}\phi_{{{u.uround(self.energy, u.eV)} \, \mathrm{{eV}}}}"

    @property
    def tex_ket(self):
        return rf"{state.fmt_amplitude_for_tex(self.amplitude)}\left| \phi_{{{u.uround(self.energy, u.eV)} \, \mathrm{{eV}}}} \right\rangle"

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Mass", utils.fmt_quantity(self.mass, utils.MASS_UNITS))
        info.add_field("Energy", utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))
        info.add_field(
            "Wavenumber",
            utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS),
        )

        return info


class QHOState(state.QuantumState):
    """
    A class representing a bound state of the quantum harmonic oscillator.

    Attributes
    ----------
    n: QuantumNumber
        The quantum number of the state.
    spring_constant: :class:`float`
        The spring constant of the harmonic oscillator this is a state of.
    mass: :class:`float`
        The mass of the particle associated with this state.
    energy: :class:`float`
        The energy of the state.
    omega: :class:`float`
        The angular frequency of the state.
    frequency: :class:`float`
        The cyclic frequency of the state.
    period: :class:`float`
        The period of the state.
    """

    smallest_n = 0

    eigenvalues = state.Eigenvalues.DISCRETE
    binding = state.Binding.BOUND
    derivation = state.Derivation.ANALYTIC

    def __init__(
        self,
        spring_constant: float,
        mass: float = u.electron_mass,
        n: state.QuantumNumber = 0,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """
        Parameters
        ----------
        spring_constant
            The spring constant for the quantum harmonic oscillator.
        mass
            The mass of the particle.
        n
            The energy quantum number of the state.
        """
        self.n = n
        self.spring_constant = spring_constant
        self.mass = mass

        super().__init__(amplitude=amplitude)

    @classmethod
    def from_omega_and_mass(
        cls,
        omega: float,
        mass: float = u.electron_mass,
        n: state.QuantumNumber = 0,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """
        Parameters
        ----------
        omega
            The angular frequency of the state.
        mass
            The mass of the particle.
        n
            The energy quantum number of the state.

        Returns
        -------
        state
            The state constructed from the parameters.
        """
        return cls(
            spring_constant=mass * (omega ** 2), mass=mass, n=n, amplitude=amplitude
        )

    @classmethod
    def from_potential(
        cls,
        potential: potentials.HarmonicOscillator,
        mass: float,
        n: state.QuantumNumber = 0,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """

        Parameters
        ----------
        potential
            A harmonic oscillator potential.
        mass
            The mass of the particle.
        n
            The energy quantum number of the state.

        Returns
        -------
        state
            The state constructed from the parameters.
        """
        return cls(
            spring_constant=potential.spring_constant,
            mass=mass,
            n=n,
            amplitude=amplitude,
        )

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
        Warning: for large enough n (>= ~60) this will fail due to n! overflowing.
        """
        norm = ((self.mass * self.omega / (u.pi * u.hbar)) ** (1 / 4)) / (
            np.float64(2 ** (self.n / 2))
            * np.sqrt(np.float64(spmisc.factorial(self.n)))
        )
        exp = np.exp(-self.mass * self.omega * (x ** 2) / (2 * u.hbar))
        herm = special.hermite(self.n)(np.sqrt(self.mass * self.omega / u.hbar) * x)

        return self.amplitude * (norm * exp * herm).astype(np.complex128)

    def __repr__(self):
        return utils.fmt_fields(self, "n", "mass", "energy", "amplitude")

    @property
    def ket(self):
        return f"{state.fmt_amplitude(self.amplitude)}|{self.n}>"

    @property
    def tex(self):
        return rf"{state.fmt_amplitude_for_tex(self.amplitude)}\psi_{{{self.n}}}"

    @property
    def tex_ket(self):
        return rf"{state.fmt_amplitude_for_tex(self.amplitude)}\left| \psi_{{{self.n}}} \right\rangle"

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Mass", utils.fmt_quantity(self.mass, utils.MASS_UNITS))
        info.add_field("n", self.n)
        info.add_field("Energy", utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))
        info.add_field("Period", utils.fmt_quantity(self.period, utils.TIME_UNITS))

        return info


class FiniteSquareWellState(state.QuantumState):
    """
    A class representing a bound state of a finite square well.

    Attributes
    ----------
    n: QuantumNumber
        The quantum number of the state.
    well_depth: :class:`float`
        The depth of the potential well, in energy units.
    well_width: :class:`float`
        The width of the potential well.
    well_center: :class:`float`
        The position of the center of the well.
    left_edge: :class:`float`
        The position of the left edge of the well.
    right_edge: :class:`float`
        The position of the right edge of the well.
    mass: :class:`float`
        The mass of the particle.
    """

    smallest_n = 1

    eigenvalues = state.Eigenvalues.DISCRETE
    binding = state.Binding.BOUND
    derivation = state.Derivation.ANALYTIC

    def __init__(
        self,
        well_depth: float,
        well_width: float,
        mass: float,
        n: state.QuantumNumber = 1,
        well_center: float = 0,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """
        Parameters
        ----------
        well_depth
            The depth of the potential well, in energy units.
        well_width
            The width of the potential well.
        mass
            The mass of the particle.
        n
            The energy quantum number of the state.
        well_center
            The position of the center of the well.
        """
        self.well_depth = well_depth
        self.well_width = well_width
        self.well_center = well_center
        self.mass = mass
        self.n = n

        z_0 = (well_width / 2) * np.sqrt(2 * mass * well_depth) / u.hbar

        if n - 1 > z_0 // (u.pi / 2):
            raise exceptions.IllegalQuantumState(
                "There is no bound state with the given parameters"
            )

        left_bound = (n - 1) * u.pi / 2
        right_bound = min(z_0, left_bound + (u.pi / 2))

        with np.errstate(divide="ignore"):  # ignore division by zero on the edges
            # determine the energy of the state by solving a transcendental equation
            if n % 2 != 0:  # n is odd
                z = optimize.brentq(
                    lambda z: np.tan(z) - np.sqrt(((z_0 / z) ** 2) - 1),
                    left_bound,
                    right_bound,
                )
                self.function_inside_well = np.cos
                self.symmetry = "symmetric"
            else:  # n is even
                z = optimize.brentq(
                    lambda z: (1 / np.tan(z)) + np.sqrt(((z_0 / z) ** 2) - 1),
                    left_bound,
                    right_bound,
                )
                self.function_inside_well = np.sin
                self.symmetry = "antisymmetric"

        self.wavenumber_inside_well = z / (well_width / 2)
        self.energy = (
            ((u.hbar * self.wavenumber_inside_well) ** 2) / (2 * mass)
        ) - well_depth
        self.wavenumber_outside_well = np.sqrt(-2 * mass * self.energy) / u.hbar

        self.normalization_factor_inside_well = 1 / np.sqrt(
            (self.well_width / 2) + (1 / self.wavenumber_outside_well)
        )
        self.normalization_factor_outside_well = (
            np.exp(self.wavenumber_outside_well * (self.well_width / 2))
            * self.function_inside_well(
                self.wavenumber_inside_well * (self.well_width / 2)
            )
            / np.sqrt((self.well_width / 2) + (1 / self.wavenumber_outside_well))
        )

        super().__init__(amplitude=amplitude)

    @classmethod
    def from_potential(
        cls,
        potential: potentials.FiniteSquareWell,
        mass: float,
        n: state.QuantumNumber = 1,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """
        Parameters
        ----------
        potential
            A finite square well potential.
        mass
            The mass of the particle.
        n
            The energy quantum number of the state.

        Returns
        -------
        state
            The state constructed from the parameters.
        """
        return cls(
            potential.potential_depth,
            potential.width,
            mass,
            n=n,
            well_center=potential.center,
            amplitude=amplitude,
        )

    @classmethod
    def all_states_of_well_from_parameters(
        cls,
        well_depth: float,
        well_width: float,
        mass: float,
        well_center: float = 0,
        amplitude: state.ProbabilityAmplitude = 1,
    ) -> List[state.QuantumState]:
        """
        Parameters
        ----------
        well_depth
            The depth of the potential well, in energy units.
        well_width
            The width of the potential well.
        mass
            The mass of the particle.
        well_center
            The position of the center of the well.

        Returns
        -------
        state
            The bound states of the finite square well with the given parameters.
        """
        states = []
        for n in itertools.count(1):
            try:
                states.append(
                    cls(
                        well_depth,
                        well_width,
                        mass,
                        n=n,
                        well_center=well_center,
                        amplitude=amplitude,
                    )
                )
            except exceptions.IllegalQuantumState:
                return states

    @classmethod
    def all_states_of_well_from_well(
        cls,
        potential: potentials.FiniteSquareWell,
        mass: float,
        amplitude: state.ProbabilityAmplitude = 1,
    ) -> List[state.QuantumState]:
        """
        Parameters
        ----------
        potential
            A finite square well.
        mass
            The mass of the particle.

        Returns
        -------
        state
            The bound states of the finite square well.
        """
        return cls.all_states_of_well_from_parameters(
            potential.potential_depth,
            potential.width,
            mass,
            well_center=potential.center,
            amplitude=amplitude,
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
        cond = np.greater_equal(x, self.left_edge) * np.less_equal(x, self.right_edge)

        if self.symmetry == "antisymmetric":
            sym = -1
        else:
            sym = 1

        psi = np.where(
            cond,
            self.normalization_factor_inside_well
            * self.function_inside_well(self.wavenumber_inside_well * x),
            self.normalization_factor_outside_well
            * np.exp(-self.wavenumber_outside_well * np.abs(x)),
        ).astype(np.complex128)

        symmetrization = np.where(np.less_equal(x, self.left_edge), sym, 1)

        return psi * symmetrization

    def __repr__(self):
        return utils.fmt_fields(
            self, "n", "mass", "well_depth", "well_width", "energy", "amplitude"
        )

    @property
    def ket(self):
        return f"{state.fmt_amplitude(self.amplitude)}|{self.n}>"

    @property
    def tex(self):
        return rf"{state.fmt_amplitude_for_tex(self.amplitude)}\psi_{{{self.n}}}"

    @property
    def tex_ket(self):
        return rf"{state.fmt_amplitude_for_tex(self.amplitude)}\left| \psi_{{{self.n}}} \right\rangle"

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Mass", utils.fmt_quantity(self.mass, utils.MASS_UNITS))
        info.add_field("n", self.n)
        info.add_field("Energy", utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))
        info.add_field(
            "Well Width", utils.fmt_quantity(self.well_width, utils.LENGTH_UNITS)
        )
        info.add_field(
            "Well Depth", utils.fmt_quantity(self.well_depth, utils.ENERGY_UNITS)
        )

        return info


class GaussianWellState(state.QuantumState):
    """
    A class representing a bound state of a Gaussian well.

    Attributes
    ----------
    n: :class:`int`
        The quantum number of the state.
    well_depth: :class:`float`
        The depth of the potential well, in energy units.
    well_width: :class:`float`
        The width of the potential well.
    well_center: :class:`float`
        The position of the center of the well.
    mass: :class:`float`
        The mass of the particle.
    """

    smallest_n = 0

    eigenvalues = state.Eigenvalues.DISCRETE
    binding = state.Binding.BOUND
    derivation = state.Derivation.VARIATIONAL

    def __init__(
        self,
        well_depth: float,
        well_width: float,
        mass: float,
        n: state.QuantumNumber = 0,
        well_center: float = 0,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """

        Parameters
        ----------
        well_depth
            The depth of the potential well, in energy units.
        well_width
            The width of the potential well.
        mass
            The mass of the particle.
        n
            The energy quantum number of the state.
        well_center
            The position of the center of the well.
        """
        self.well_depth = np.abs(well_depth)
        self.well_width = well_width
        self.well_center = well_center
        self.mass = mass
        self.n = n

        if n > 0:
            logger.warning(
                "Analytic GaussianWellStates are not available for n > 0, and n = 0 is only approximate!"
            )

        max_n = (
            np.ceil(
                2
                * np.sqrt(2 * mass * self.well_depth / (u.pi * (u.hbar ** 2)))
                * well_width
            )
            + 0.5
        )
        if n > max_n:
            raise exceptions.IllegalQuantumState(
                "Bound state energy must be less than zero"
            )

        self.width = optimize.newton(
            lambda w: ((w ** 4) / (((well_width ** 2) + (w ** 2)) ** 1.5))
            - ((u.hbar ** 2) / (4 * mass * well_width * self.well_depth)),
            well_width,
        )
        self.energy = -(
            well_width
            * self.well_depth
            / np.sqrt(((well_width ** 2) + (self.width ** 2)))
            + ((u.hbar ** 2) / (8 * mass * (self.width ** 2)))
        )

        super().__init__(amplitude=amplitude)

    @classmethod
    def from_potential(
        cls,
        potential: potentials.GaussianPotential,
        mass: float,
        n: state.QuantumNumber = 0,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """
        Parameters
        ----------
        potential
            A Gaussian well potential.
        mass
            The mass of the particle.
        n
            The energy quantum number of the state.

        Returns
        -------
        state
            The state constructed from the parameters.
        """
        return cls(
            potential.potential_extrema,
            potential.width,
            mass,
            n=n,
            well_center=potential.center,
            amplitude=amplitude,
        )

    @property
    def tuple(self):
        return self.well_depth, self.well_width, self.mass, self.n

    def __call__(self, x):
        return np.exp(-0.25 * (x / self.width) ** 2) / (
            np.sqrt(np.sqrt(u.twopi) * self.width)
        )

    def __repr__(self):
        return utils.fmt_fields(self, "n", "mass", "energy", "amplitude")

    @property
    def ket(self):
        return f"{state.fmt_amplitude(self.amplitude)}|{self.n}>"

    @property
    def tex(self):
        return rf"{state.fmt_amplitude_for_tex(self.amplitude)}\psi_{{{self.n}}}"

    @property
    def tex_ket(self):
        return rf"{state.fmt_amplitude_for_tex(self.amplitude)}\left| \psi_{{{self.n}}} \right\rangle"

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Mass", utils.fmt_quantity(self.mass, utils.MASS_UNITS))
        info.add_field("n", self.n)
        info.add_field("Energy", utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))

        return info


class OneDSoftCoulombState(state.QuantumState):
    """
    A class representing a bound state of the soft Coulomb potential in one dimension.

    Attributes
    ----------
    n: QuantumNumber
        The quantum number of the state.

    """

    smallest_n = 1

    eigenvalues = state.Eigenvalues.DISCRETE
    binding = state.Binding.BOUND
    derivation = state.Derivation.ANALYTIC

    def __init__(self, n=1, amplitude: state.ProbabilityAmplitude = 1):
        """
        Parameters
        ----------
        n
            The energy quantum number of the state.
        """
        self.n = n

        super().__init__(amplitude=amplitude)

    @classmethod
    def from_potential(cls, potential, test_mass, n=1, **kwargs):
        return cls(n=n)

    @property
    def tuple(self):
        return (self.n,)

    def __call__(self, z):
        raise NotImplementedError(
            "haven't implemented analytic eigenstates for this potential yet"
        )

    def __repr__(self):
        return utils.fmt_fields(self, "n", "amplitude")

    @property
    def ket(self):
        return f"{state.fmt_amplitude(self.amplitude)}|{self.n}>"

    @property
    def tex(self):
        return rf"{state.fmt_amplitude_for_tex(self.amplitude)}\psi_{{{self.n}}}"

    @property
    def tex_ket(self):
        return rf"{state.fmt_amplitude_for_tex(self.amplitude)}\left| \psi_{{{self.n}}} \right\rangle"

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("n", self.n)

        return info


class NumericOneDState(state.QuantumState):
    """
    A numerically-derived one-dimensional quantum state.

    Attributes
    ----------
    corresponding_analytic_state: :class:`QuantumState`
        The analytic state that this numeric state nominally approximates.
    n: :class:`int`
        The quantum number of a bound state.
        Will raise an exception if the corresponding analytic state is not bound.
    wavenumber: :class:`float`
        The wavenumber of a free state.
        Will raise an exception if the corresponding analytic state is not free.
    """

    eigenvalues = state.Eigenvalues.CONTINUOUS
    derivation = state.Derivation.ANALYTIC

    def __init__(
        self,
        *,
        g: "mesh.PsiVector",
        energy: float,
        corresponding_analytic_state: state.QuantumState,
        binding: state.Binding,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """
        Parameters
        ----------
        g
            The numerically-determined wavefunction as a function of space.
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

        super().__init__(amplitude=amplitude)

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
        return self.g

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
