import logging

import numpy as np
import numpy.fft as nfft
import scipy.integrate as integ
import scipy.optimize as optim

import simulacra as si
import simulacra.units as u

from .. import utils, exceptions

from . import potential, windows

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ElectricPotential(potential.PotentialEnergy):
    pass


class UniformLinearlyPolarizedElectricPotential(ElectricPotential):
    def __init__(self, window: windows.TimeWindow = windows.NoTimeWindow()):
        super().__init__()

        self.window = window

    def get_electric_field_amplitude(self, t: float) -> float:
        """Return the electric field amplitude at time ``t``."""
        return self.window(t)

    def get_vector_potential_amplitude(self, t: float) -> float:
        """Return the vector potential amplitude at time ``t``."""
        raise NotImplementedError

    def __call__(self, *, t, z, test_charge: float, **kwargs):
        """
        Return the electric potential at time ``t``, distance along the polarization axis ``z``, for a given ``test_charge``.

        Parameters
        ----------
        t
            The time.
        z
            The distance along the polarization axis of the field.
        test_charge
            The charge of the test particle.
        kwargs
            Ignores additional keyword arguments.

        Returns
        -------

        """
        return -z * test_charge * self.get_electric_field_amplitude(t)

    def get_electric_field_integral_numeric(self, times, rule: str = "simps"):
        """
        Return the integral of the electric field over the ``times``.

        Parameters
        ----------
        times
            The times to consider.
        rule : {``'trapz'``, ``'simps'``}
            The integration rule to use.
        Returns
        -------
        integral
            The result of the integration.
        """
        return getattr(integ, rule)(y=self.get_electric_field_amplitude(times), x=times)

    def get_vector_potential_amplitude_numeric(self, times, rule: str = "simps"):
        """Return the vector potential amplitude at ``times[-1]``."""
        return -self.get_electric_field_integral_numeric(times, rule=rule)

    def get_electric_field_integral_numeric_cumulative(self, times):
        """Return the integral of the electric field amplitude from the start of ``times`` to each time in ``times``."""
        return integ.cumtrapz(
            y=self.get_electric_field_amplitude(times), x=times, initial=0
        )

    def get_vector_potential_amplitude_numeric_cumulative(self, times):
        """Return the vector potential amplitude from the start of ``times`` to each time in ``times``."""
        return -self.get_electric_field_integral_numeric_cumulative(times)

    def get_fluence_numeric(self, times, rule: str = "simps"):
        """Return the pulse power integrated over the ``times`` (i.e., the fluence)."""
        return (
            u.epsilon_0
            * u.c
            * getattr(integ, rule)(
                y=np.abs(self.get_electric_field_amplitude(times)) ** 2, x=times
            )
        )


class NoElectricPotential(UniformLinearlyPolarizedElectricPotential):
    """A class representing the lack of an electric field."""

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time ``t``."""
        return np.zeros(np.shape(t)) * super().get_electric_field_amplitude(t)

    def get_vector_potential_amplitude(self, t):
        return np.zeros(np.shape(t))


class Rectangle(UniformLinearlyPolarizedElectricPotential):
    """A class representing an electric with a sharp turn-on and turn-off time."""

    def __init__(
        self,
        start_time: float = 0 * u.asec,
        end_time: float = 50 * u.asec,
        amplitude: float = 1 * u.atomic_electric_field,
        **kwargs,
    ):
        """
        Parameters
        ----------
        start_time
            The time the electric field turns on.
        end_time
            The time the electric field turns off.
        amplitude
            The amplitude of the electric field while it's on.
        kwargs
            Any additional keyword arguments are passed to the :class:`UniformLinearlyPolarizedElectricPotential` constructor.
        """
        if start_time >= end_time:
            raise exceptions.InvalidPotentialParameter(
                "end_time must be later than start_time"
            )

        super().__init__(**kwargs)

        self.start_time = start_time
        self.end_time = end_time
        self.amplitude = amplitude

    def get_electric_field_amplitude(self, t):
        cond = np.greater_equal(t, self.start_time) * np.less_equal(t, self.end_time)
        on = np.ones(np.shape(t))
        off = np.zeros(np.shape(t))

        out = (
            np.where(cond, on, off)
            * self.amplitude
            * super().get_electric_field_amplitude(t)
        )

        return out

    def __repr__(self):
        return utils.make_repr(self, "start_time", "end_time", "amplitude", "window")

    def __str__(self):
        return utils.make_repr(
            self,
            ("start_time", "asec"),
            ("end_time", "asec"),
            ("amplitude", "atomic_electric_field"),
            "window",
        )

    def info(self) -> si.Info:
        info = super().info()

        info.add_field(
            "Amplitude", utils.fmt_quantity(self.amplitude, utils.ELECTRIC_FIELD_UNITS)
        )
        info.add_field(
            "Start Time", utils.fmt_quantity(self.start_time, utils.TIME_UNITS)
        )
        info.add_field("End Time", utils.fmt_quantity(self.end_time, utils.TIME_UNITS))

        info.add_info(self.window.info())

        return info


def keldysh_parameter(
    omega: float,
    electric_field_amplitude: float,
    *,
    ionization_potential: float = u.rydberg,
    test_charge: float = u.electron_charge,
    test_mass: float = u.electron_mass_reduced,
) -> float:
    return (
        omega
        * np.sqrt(2 * test_mass * np.abs(ionization_potential))
        / (-test_charge * electric_field_amplitude)
    )


def electric_field_amplitude_from_keldysh_parameter(
    keldysh_parameter: float,
    omega: float,
    *,
    ionization_potential: float = u.rydberg,
    test_charge: float = u.electron_charge,
    test_mass: float = u.electron_mass_reduced,
) -> float:
    return (
        omega
        * np.sqrt(2 * test_mass * np.abs(ionization_potential))
        / (-test_charge * keldysh_parameter)
    )


class SineWave(UniformLinearlyPolarizedElectricPotential):
    """
    A sine-wave electric field.
    """

    def __init__(
        self,
        omega: float,
        amplitude: float = 1 * u.atomic_electric_field,
        phase=0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        omega
            The angular frequency of the sine wave.
        amplitude
            The peak electric field amplitude of the sine wave.
        phase
            The phase of the sine wave (:math:`0` is a sine wave, :math:`\pi/2` is cosine wave).
        kwargs
            Any additional keyword arguments are passed to the :class:`UniformLinearlyPolarizedElectricPotential` constructor.
        """
        if omega <= 0:
            raise exceptions.InvalidPotentialParameter("omega must be positive")

        super().__init__(**kwargs)

        self.omega = omega
        self.phase = phase % u.twopi
        self.amplitude = amplitude

    @classmethod
    def from_frequency(
        cls,
        frequency: float,
        amplitude: float = 1 * u.atomic_electric_field,
        phase: float = 0,
        **kwargs,
    ) -> "SineWave":
        """
        frequency
            The cyclic frequency of the sine wave.
        amplitude
            The peak electric field amplitude of the sine wave.
        phase
            The phase of the sine wave (:math:`0` is a sine wave, :math:`\pi/2` is cosine wave).
        kwargs
            Any additional keyword arguments are passed to the :class:`UniformLinearlyPolarizedElectricPotential` constructor.
        """
        return cls(frequency * u.twopi, amplitude=amplitude, phase=phase, **kwargs)

    @classmethod
    def from_period(
        cls,
        period: float,
        amplitude: float = 1 * u.atomic_electric_field,
        phase: float = 0,
        **kwargs,
    ) -> "SineWave":
        """
        Parameters
        ----------
        period
            The period of the sine wave.
        amplitude
            The peak electric field amplitude of the sine wave.
        phase
            The phase of the sine wave (:math:`0` is a sine wave, :math:`\pi/2` is cosine wave).
        kwargs
            Any additional keyword arguments are passed to the :class:`UniformLinearlyPolarizedElectricPotential` constructor.
        """
        return cls.from_frequency(
            1 / period, amplitude=amplitude, phase=phase, **kwargs
        )

    @classmethod
    def from_wavelength(
        cls,
        wavelength: float,
        amplitude: float = 1 * u.atomic_electric_field,
        phase: float = 0,
        **kwargs,
    ) -> "SineWave":
        """
        Parameters
        ----------
        wavelength
            The wavelength of the sine wave.
        amplitude
            The peak electric field amplitude of the sine wave.
        phase
            The phase of the sine wave (:math:`0` is a sine wave, :math:`\pi/2` is cosine wave).
        kwargs
            Any additional keyword arguments are passed to the :class:`UniformLinearlyPolarizedElectricPotential` constructor.
        """
        return cls.from_frequency(
            u.c / wavelength, amplitude=amplitude, phase=phase, **kwargs
        )

    @classmethod
    def from_photon_energy(
        cls,
        photon_energy: float,
        amplitude: float = 1 * u.atomic_electric_field,
        phase: float = 0,
        **kwargs,
    ) -> "SineWave":
        """
        Parameters
        ----------
        photon_energy
            The energy of the photons of the electric field.
        amplitude
            The peak electric field amplitude of the sine wave.
        phase
            The phase of the sine wave (:math:`0` is a sine wave, :math:`\pi/2` is cosine wave).
        kwargs
            Any additional keyword arguments are passed to the :class:`UniformLinearlyPolarizedElectricPotential` constructor.
        """
        return cls(photon_energy / u.hbar, amplitude=amplitude, phase=phase, **kwargs)

    @classmethod
    def from_photon_energy_and_intensity(
        cls,
        photon_energy: float,
        intensity: float = 1 * u.TWcm2,
        phase: float = 0,
        **kwargs,
    ) -> "SineWave":
        """
        Parameters
        ----------
        photon_energy
            The energy of the photons of the electric field.
        intensity
            The peak electric field intensity of the sine wave.
        phase
            The phase of the sine wave (:math:`0` is a sine wave, :math:`\pi/2` is cosine wave).
        kwargs
            Any additional keyword arguments are passed to the :class:`UniformLinearlyPolarizedElectricPotential` constructor.
        """
        if intensity < 0:
            raise exceptions.InvalidPotentialParameter("intensity must be non-negative")

        return cls(
            photon_energy / u.hbar,
            amplitude=np.sqrt(2 * intensity / (u.epsilon_0 * u.c)),
            phase=phase,
            **kwargs,
        )

    @property
    def intensity(self):
        """Cycle-averaged intensity"""
        return 0.5 * u.epsilon_0 * u.c * (self.amplitude ** 2)

    @property
    def frequency(self):
        return self.omega / u.twopi

    @frequency.setter
    def frequency(self, frequency):
        self.omega = frequency * u.twopi

    @property
    def period(self):
        return 1 / self.frequency

    @period.setter
    def period(self, period):
        self.frequency = 1 / period

    @property
    def wavelength(self):
        return u.c / self.frequency

    @wavelength.setter
    def wavelength(self, wavelength):
        self.frequency = u.c / wavelength

    @property
    def photon_energy(self):
        return u.hbar * self.omega

    @photon_energy.setter
    def photon_energy(self, photon_energy):
        self.omega = photon_energy / u.hbar

    def keldysh_parameter(
        self,
        ionization_potential=u.rydberg,
        test_mass=u.electron_mass,
        test_charge=u.electron_charge,
    ):
        return keldysh_parameter(
            self.omega,
            self.amplitude,
            ionization_potential=ionization_potential,
            test_mass=test_mass,
            test_charge=test_charge,
        )

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        return (
            np.sin((self.omega * t) + self.phase)
            * self.amplitude
            * super().get_electric_field_amplitude(t)
        )

    def get_peak_amplitude(self):
        return self.amplitude

    def get_peak_power_density(self):
        return u.c * u.epsilon_0 * (np.abs(self.amplitude) ** 2)

    def get_average_power_density(self):
        return 0.5 * u.c * u.epsilon_0 * (np.abs(self.amplitude) ** 2)

    def __repr__(self):
        return utils.make_repr(
            self, "wavelength", "photon_energy", "amplitude", "phase", "window"
        )

    def __str__(self):
        return utils.make_repr(
            self,
            ("wavelength", "nm"),
            ("photon_energy", "eV"),
            ("amplitude", "atomic_electric_field"),
            ("phase", "rad"),
            "window",
        )

    def info(self) -> si.Info:
        info = super().info()

        info.add_field(
            "Amplitude", utils.fmt_quantity(self.amplitude, utils.ELECTRIC_FIELD_UNITS)
        )
        info.add_field(
            "Intensity", f"{u.uround(self.intensity, u.TW / (u.cm ** 2))} TW/cm^2"
        )
        info.add_field(
            "Photon Energy", utils.fmt_quantity(self.photon_energy, utils.ENERGY_UNITS)
        )
        info.add_field(
            "Frequency", utils.fmt_quantity(self.frequency, utils.FREQUENCY_UNITS)
        )
        info.add_field("Period", utils.fmt_quantity(self.period, utils.TIME_UNITS))
        info.add_field(
            "Wavelength", utils.fmt_quantity(self.wavelength, utils.LENGTH_UNITS)
        )
        info.add_field("Phase", utils.fmt_quantity(self.phase, utils.ANGLE_UNITS))

        info.add_info(self.window.info())

        return info


class SumOfSinesPulse(UniformLinearlyPolarizedElectricPotential):
    def __init__(
        self,
        pulse_width: float = 200 * u.asec,
        pulse_frequency_ratio: float = 5,
        fluence: float = 1 * u.Jcm2,
        phase: float = 0,
        pulse_center: float = 0 * u.asec,
        number_of_modes: int = 71,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if phase != 0:
            raise NotImplementedError("phase != 0 not implemented for SumOfSinesPulse")

        self.pulse_width = pulse_width

        self.number_of_modes = number_of_modes
        self.mode_spacing = u.twopi / (self.number_of_modes * self.pulse_width)
        self.pulse_frequency_ratio = pulse_frequency_ratio

        self.omega_min = self.pulse_frequency_ratio * self.mode_spacing

        # self.phase = phase
        if phase != 0:
            logger.warning("Phase not implemented for SumOfSines pulse!")

        self.fluence = fluence
        self.pulse_center = pulse_center

        self.delta_omega = u.twopi / self.pulse_width
        self.omega_max = self.omega_min + self.delta_omega
        # self.omega_carrier = (self.omega_min + self.omega_max) / 2

        self.amplitude_omega = np.sqrt(
            self.fluence
            * self.delta_omega
            / (u.twopi * number_of_modes * u.c * u.epsilon_0)
        )
        self.amplitude = self.amplitude_omega

        self.cycle_period = u.twopi / self.mode_spacing

    @property
    def photon_energy_min(self):
        return u.hbar * self.omega_min

    @property
    def photon_energy_max(self):
        return u.hbar * self.omega_max

    @property
    def frequency_min(self):
        return self.omega_min / u.twopi

    @property
    def frequency_max(self):
        return self.omega_max / u.twopi

    @property
    def frequency_delta(self):
        return self.delta_omega / u.twopi

    @property
    def amplitude_per_frequency(self):
        return np.sqrt(u.twopi) * self.amplitude_omega

    def __repr__(self):
        return utils.make_repr(
            self,
            "pulse_width",
            "pulse_center",
            "fluence",
            # 'phase',
            "pulse_frequency_ratio",
            "window",
        )

    def __str__(self):
        out = utils.make_repr(
            self,
            ("pulse_width", "asec"),
            ("pulse_center", "asec"),
            ("fluence", "J/cm^2"),
            # 'phase',
            "pulse_frequency_ratio",
        )

        return out + super().__str__()

    def get_electric_field_amplitude(self, t):
        tau = t - self.pulse_center

        cond = np.not_equal(tau, 0)

        on = np.real(
            np.exp(-1j * self.pulse_frequency_ratio * self.mode_spacing * tau)
            * (1 - np.exp(-1j * self.mode_spacing * self.number_of_modes * tau))
            / (1 - np.exp(-1j * self.mode_spacing * tau))
        )
        off = self.number_of_modes

        amp = np.where(cond, on, off)

        return amp * self.amplitude * super().get_electric_field_amplitude(t)


DEFAULT_PULSE_WIDTH = 200 * u.asec
DEFAULT_FLUENCE = 1 * u.Jcm2
DEFAULT_PHASE = 0
DEFAULT_OMEGA_MIN = u.twopi * 30 * u.THz
DEFAULT_OMEGA_CARRIER = u.twopi * 2530 * u.THz
DEFAULT_PULSE_CENTER = 0 * u.asec
DEFAULT_KELDYSH_PARAMETER = 1


def sinc(x):
    """A wrapper over np.sinc, which is really sinc(pi * x). This version is sinc(x)."""
    return np.sinc(x / u.pi)


class SincPulse(UniformLinearlyPolarizedElectricPotential):
    """
    Attributes
    ----------
    pulse_width
        The time from the pulse center to the first envelope zero; the inverse of the pulse bandwidth.
    fluence
        The fluence (total energy flux) of the pulse.
    phase
        The carrier-envelope phase of the pulse.
    pulse_center
        The time of the center of the pulse.

    time_fwhm
    time_fwhm_power

    omega_carrier
        The center angular frequency of the pulse's power spectrum.
    omega_min
        The minimum angular frequency of the pulse's power spectrum.
    omega_max
        The maximum angular frequency of the pulse's power spectrum.
    delta_omega
        The bandwidth of angular frequencies in the pulse's power spectrum.

    frequency_carrier
        The center cyclic frequency of the pulse's power spectrum.
    frequency_min
        The minimum cyclic frequency of the pulse's power spectrum.
    frequency_max
        The maximum cyclic frequency of the pulse's power spectrum.
    frequency_delta
        The bandwidth of cyclic frequencies in the pulse's power spectrum.

    photon_energy_carrier
        The center photon energy of the pulse's power spectrum.
    photon_energy_min
        The minimum photon energy of the pulse's power spectrum.
    photon_energy_max
        The maximum photon energy of the pulse's power spectrum.
    photon_energy_bandwidth
        The bandwidth of photon energies in the pulse's power spectrum.

    amplitude_per_frequency
        The absolute value of the "amplitude density" of the pulse's amplitude spectrum.
    """

    def __init__(
        self,
        pulse_width: float = DEFAULT_PULSE_WIDTH,
        fluence: float = DEFAULT_FLUENCE,
        phase: float = DEFAULT_PHASE,
        pulse_center: float = DEFAULT_PULSE_CENTER,
        omega_min: float = DEFAULT_OMEGA_MIN,
        **kwargs,
    ):
        """
        Parameters
        ----------
        pulse_width
            The time from the pulse center to the first envelope zero; the inverse of the pulse bandwidth.
        fluence
            The fluence of the pulse.
        phase
            The carrier-envelope phase of the pulse.
        pulse_center
            The time of the center of the pulse.
        omega_min
            The minimum angular frequency of the pulse's power spectrum.
        kwargs
            Any additional keyword arguments are passed to the :class:`UniformLinearlyPolarizedElectricPotential` constructor.
        """
        if pulse_width <= 0:
            raise exceptions.InvalidPotentialParameter("pulse width must be positive")
        if fluence < 0:
            raise exceptions.InvalidPotentialParameter("fluence must be non-negative")
        if omega_min <= 0:
            raise exceptions.InvalidPotentialParameter("omega_min must be positive")

        super().__init__(**kwargs)

        self.omega_min = omega_min
        self.pulse_width = pulse_width
        self.phase = phase % u.twopi
        self.fluence = fluence
        self.pulse_center = pulse_center

        self.delta_omega = u.twopi / self.pulse_width
        self.omega_max = self.omega_min + self.delta_omega
        self.omega_carrier = (self.omega_min + self.omega_max) / 2

        self.amplitude_omega = np.sqrt(
            self.fluence / (2 * u.epsilon_0 * u.c * self.delta_omega)
        )
        self.amplitude = np.sqrt(
            self.fluence * self.delta_omega / (u.pi * u.epsilon_0 * u.c)
        )

    @classmethod
    def from_omega_min(cls, *args, **kwargs):
        """Alias for the default constructor."""
        return cls(*args, **kwargs)

    @classmethod
    def from_omega_carrier(
        cls,
        pulse_width=DEFAULT_PULSE_WIDTH,
        fluence=DEFAULT_FLUENCE,
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        omega_carrier=DEFAULT_OMEGA_CARRIER,
        **kwargs,
    ) -> "SincPulse":
        """
        Parameters
        ----------
        pulse_width
            The time from the pulse center to the first envelope zero; the inverse of the pulse bandwidth.
        fluence
            The fluence of the pulse.
        phase
            The carrier-envelope phase of the pulse.
        pulse_center
            The time of the center of the pulse.
        omega_carrier
            The center angular frequency of the pulse's power spectrum.
        kwargs
            Any additional keyword arguments are passed to the :class:`UniformLinearlyPolarizedElectricPotential` constructor.

        Returns
        -------

        """
        delta_omega = u.twopi / pulse_width
        omega_min = omega_carrier - (delta_omega / 2)

        return cls(
            pulse_width=pulse_width,
            omega_min=omega_min,
            fluence=fluence,
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

    @classmethod
    def from_amplitude(
        cls,
        pulse_width=DEFAULT_PULSE_WIDTH,
        amplitude=1 * u.atomic_electric_field,
        phase=DEFAULT_PHASE,
        omega_min=DEFAULT_OMEGA_MIN,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "SincPulse":
        """
        Parameters
        ----------
        pulse_width
            The time from the pulse center to the first envelope zero; the inverse of the pulse bandwidth.
        amplitude
            The peak of a cosine-like sinc pulse (i.e., this ignores the ``phase``).
        phase
            The carrier-envelope phase of the pulse.
        pulse_center
            The time of the center of the pulse.
        omega_min
            The minimum angular frequency of the pulse's power spectrum.
        kwargs
            Any additional keyword arguments are passed to the :class:`UniformLinearlyPolarizedElectricPotential` constructor.
        """
        delta_omega = u.twopi / pulse_width
        fluence = u.pi * u.epsilon_0 * u.c * (amplitude ** 2) / delta_omega

        pot = cls(
            pulse_width=pulse_width,
            omega_min=omega_min,
            fluence=fluence,
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

        if np.isclose(amplitude / pot.amplitude, 1):
            pot.amplitude = amplitude
        else:
            raise ValueError("Given amplitude not close enough to calculated amplitude")

        return pot

    @classmethod
    def from_number_of_cycles(
        cls,
        pulse_width=DEFAULT_PULSE_WIDTH,
        fluence=DEFAULT_FLUENCE,
        phase=DEFAULT_PHASE,
        number_of_cycles=1,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "SincPulse":
        """
        Parameters
        ----------
        pulse_width
            The time from the pulse center to the first envelope zero; the inverse of the pulse's cyclic frequency bandwidth.
        fluence
            The fluence of the pulse.
        phase
            The carrier-envelope phase of the pulse.
        pulse_center
            The time of the center of the pulse.
        number_of_cycles
            The number of carrier cycles per pulse width.
        kwargs
            Any additional keyword arguments are passed to the :class:`UniformLinearlyPolarizedElectricPotential` constructor.
        """
        if number_of_cycles <= 0.5:
            raise exceptions.InvalidPotentialParameter("number_of_cycles must be > 0.5")
        omega_carrier = u.twopi * number_of_cycles / pulse_width

        pot = cls.from_omega_carrier(
            pulse_width=pulse_width,
            omega_carrier=omega_carrier,
            fluence=fluence,
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

        return pot

    @classmethod
    def from_keldysh_parameter(
        cls,
        pulse_width=DEFAULT_PULSE_WIDTH,
        omega_min=DEFAULT_OMEGA_MIN,
        keldysh_parameter=DEFAULT_KELDYSH_PARAMETER,
        ionization_potential=u.rydberg,
        test_charge=u.electron_charge,
        test_mass=u.electron_mass_reduced,
        keldysh_omega_selector="carrier",
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "SincPulse":
        delta_omega = u.twopi / pulse_width

        keldysh_omega = {
            "carrier": omega_min + (delta_omega / 2),
            "bandwidth": delta_omega,
        }[keldysh_omega_selector]

        amplitude = electric_field_amplitude_from_keldysh_parameter(
            keldysh_parameter,
            keldysh_omega,
            ionization_potential=ionization_potential,
            test_charge=test_charge,
            test_mass=test_mass,
        )

        fluence = u.pi * u.epsilon_0 * u.c * (amplitude ** 2) / delta_omega

        return cls(
            pulse_width=pulse_width,
            omega_min=omega_min,
            fluence=fluence,
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

    @property
    def photon_energy_min(self):
        return u.hbar * self.omega_min

    @property
    def photon_energy_carrier(self):
        return u.hbar * self.omega_carrier

    @property
    def photon_energy_max(self):
        return u.hbar * self.omega_max

    @property
    def photon_energy_bandwidth(self):
        return self.photon_energy_max - self.photon_energy_min

    @property
    def frequency_min(self):
        return self.omega_min / u.twopi

    @property
    def frequency_carrier(self):
        return self.omega_carrier / u.twopi

    @property
    def frequency_max(self):
        return self.omega_max / u.twopi

    @property
    def frequency_delta(self):
        return self.delta_omega / u.twopi

    @property
    def amplitude_per_omega(self):
        return np.sqrt(u.twopi) * self.amplitude_omega

    @property
    def number_of_cycles(self):
        return self.omega_carrier / self.delta_omega

    def keldysh_parameter(
        self,
        ionization_potential=u.rydberg,
        test_mass=u.electron_mass_reduced,
        test_charge=u.electron_charge,
        keldysh_omega_selector="carrier",
    ):
        keldysh_omega = {"carrier": self.omega_carrier, "bandwidth": self.delta_omega}[
            keldysh_omega_selector
        ]

        return keldysh_parameter(
            keldysh_omega,
            self.amplitude,
            ionization_potential=ionization_potential,
            test_mass=test_mass,
            test_charge=test_charge,
        )

    def get_electric_field_envelope(self, t):
        tau = np.array(t) - self.pulse_center
        return sinc(self.delta_omega * tau / 2)

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        tau = np.array(t) - self.pulse_center
        amp = self.get_electric_field_envelope(t) * np.cos(
            (self.omega_carrier * tau) + self.phase
        )

        return amp * self.amplitude * super().get_electric_field_amplitude(t)

    def __repr__(self):
        return utils.make_repr(
            self,
            "pulse_width",
            "pulse_center",
            "fluence",
            "phase",
            "photon_energy_carrier",
            "window",
        )

    def __str__(self):
        return utils.make_repr(
            self,
            ("pulse_width", "asec"),
            ("pulse_center", "asec"),
            ("fluence", "J/cm^2"),
            ("phase", "rad"),
            ("photon_energy_carrier", "eV"),
            "window",
        )

    def info(self) -> si.Info:
        info = super().info()

        info.add_field(
            "Pulse Width", utils.fmt_quantity(self.pulse_width, utils.TIME_UNITS)
        )
        info.add_field(
            "Pulse Center", utils.fmt_quantity(self.pulse_center, utils.TIME_UNITS)
        )
        info.add_field(
            "Electric Field Amplitude Prefactor",
            utils.fmt_quantity(self.amplitude, utils.ELECTRIC_FIELD_UNITS),
        )
        info.add_field("Fluence", utils.fmt_quantity(self.fluence, utils.FLUENCE_UNITS))
        info.add_field("Number of Cycles", self.number_of_cycles)
        info.add_field(
            "Carrier-Envelope Phase", utils.fmt_quantity(self.phase, utils.ANGLE_UNITS)
        )
        info.add_field(
            "Carrier Photon Energy",
            utils.fmt_quantity(self.photon_energy_carrier, utils.ENERGY_UNITS),
        )
        info.add_field(
            "Photon Energy Range",
            f"{u.uround(self.photon_energy_min, u.eV)} eV to {u.uround(self.photon_energy_max, u.eV)} eV",
        )
        info.add_field(
            "Photon Energy Bandwidth",
            utils.fmt_quantity(self.photon_energy_bandwidth, utils.ENERGY_UNITS),
        )
        info.add_field(
            "Carrier Frequency",
            utils.fmt_quantity(self.frequency_carrier, utils.FREQUENCY_UNITS),
        )
        info.add_field(
            "Frequency Range",
            f"{u.uround(self.frequency_min, u.THz)} THz to {u.uround(self.frequency_max, u.THz)} THz",
        )
        info.add_field(
            "Frequency Bandwidth",
            utils.fmt_quantity(self.frequency_delta, utils.FREQUENCY_UNITS),
        )
        info.add_field(
            "Keldysh Parameter (hydrogen ground state)",
            f'{u.uround(self.keldysh_parameter(keldysh_omega_selector = "carrier"))} (Carrier) | {u.uround(self.keldysh_parameter(keldysh_omega_selector = "bandwidth"))} (Bandwidth)',
        )

        info.add_info(self.window.info())

        return info


class GaussianPulse(UniformLinearlyPolarizedElectricPotential):
    """A pulse with a Gaussian-shaped amplitude envelope."""

    def __init__(
        self,
        pulse_width=DEFAULT_PULSE_WIDTH,
        omega_carrier=DEFAULT_OMEGA_CARRIER,
        fluence=DEFAULT_FLUENCE,
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "GaussianPulse":
        """
        Parameters
        ----------
        pulse_width
        omega_carrier
        fluence
        phase
        pulse_center
        kwargs
        """
        if pulse_width <= 0:
            raise exceptions.InvalidPotentialParameter("pulse width must be positive")
        if fluence < 0:
            raise exceptions.InvalidPotentialParameter("fluence must be non-negative")
        if omega_carrier < 0:
            raise exceptions.InvalidPotentialParameter(
                "omega_carrier must be non-negative"
            )

        super().__init__(**kwargs)

        self.omega_carrier = omega_carrier
        self.pulse_width = pulse_width
        self.phase = phase % u.twopi
        self.fluence = fluence
        self.pulse_center = pulse_center

        self.delta_omega = 1 / pulse_width

        self.amplitude = np.sqrt(
            2 * self.fluence / (np.sqrt(u.pi) * u.epsilon_0 * u.c * self.pulse_width)
        )
        self.amplitude_omega = self.amplitude * self.pulse_width / 2

    @classmethod
    def from_omega_min(
        cls,
        pulse_width=DEFAULT_PULSE_WIDTH,
        omega_min=DEFAULT_OMEGA_MIN,
        fluence=DEFAULT_FLUENCE,
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "GaussianPulse":
        """
        Construct a new GaussianPulse, using omega_min to set the carrier frequency to the same carrier frequency as a sinc pulse with that omega_min and the same pulse width.

        Parameters
        ----------
        pulse_width
        omega_min
        fluence
        phase
        pulse_center

        Returns
        -------

        """
        dummy = SincPulse(pulse_width=pulse_width, omega_min=omega_min)

        pulse = cls(
            pulse_width=pulse_width,
            omega_carrier=dummy.omega_carrier,
            fluence=fluence,
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

        pulse.omega_min = omega_min

        return pulse

    @classmethod
    def from_omega_carrier(cls, *args, **kwargs):
        """Alias for the default constructor."""

        return cls(*args, **kwargs)

    @classmethod
    def from_keldysh_parameter(
        cls,
        pulse_width=DEFAULT_PULSE_WIDTH,
        omega_min=DEFAULT_OMEGA_MIN,
        keldysh_parameter=DEFAULT_KELDYSH_PARAMETER,
        ionization_potential=u.rydberg,
        test_charge=u.electron_charge,
        test_mass=u.electron_mass,
        keldysh_omega_selector="carrier",
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "GaussianPulse":
        dummy = SincPulse(pulse_width=pulse_width, omega_min=omega_min)
        omega_fwhm = 2 * np.sqrt(2 * np.log(2)) / pulse_width

        keldysh_omega = {
            "carrier": dummy.omega_carrier,
            "bandwidth": omega_fwhm,
            "bandwidth_power": omega_fwhm / np.sqrt(2),
        }[keldysh_omega_selector]

        amplitude = electric_field_amplitude_from_keldysh_parameter(
            keldysh_parameter,
            keldysh_omega,
            ionization_potential=ionization_potential,
            test_charge=test_charge,
            test_mass=test_mass,
        )

        fluence = cls.fluence_from_pulse_width_and_amplitude(pulse_width, amplitude)

        return cls.from_omega_min(
            pulse_width=pulse_width,
            omega_min=omega_min,
            fluence=fluence,
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

    @classmethod
    def from_amplitude(
        cls,
        pulse_width=DEFAULT_PULSE_WIDTH,
        omega_min=DEFAULT_OMEGA_MIN,
        amplitude=1 * u.atomic_electric_field,
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "GaussianPulse":
        fluence = cls.fluence_from_pulse_width_and_amplitude(pulse_width, amplitude)

        pot = cls.from_omega_min(
            pulse_width=pulse_width,
            omega_min=omega_min,
            fluence=fluence,
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

        if np.isclose(amplitude / pot.amplitude, 1):
            pot.amplitude = amplitude
        else:
            raise ValueError("Given amplitude not close enough to calculated amplitude")

        return pot

    @classmethod
    def from_power_exclusion(
        cls,
        pulse_width=DEFAULT_PULSE_WIDTH,
        exclusion=3,
        fluence=DEFAULT_FLUENCE,
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "GaussianPulse":
        omega_carrier = exclusion * (u.pi / (np.sqrt(2) * pulse_width))

        return cls(
            pulse_width=pulse_width,
            omega_carrier=omega_carrier,
            fluence=fluence,
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

    @classmethod
    def from_number_of_cycles(
        cls,
        pulse_width=DEFAULT_PULSE_WIDTH,
        number_of_cycles=3,
        number_of_pulse_widths=3,
        fluence=DEFAULT_FLUENCE,
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "GaussianPulse":
        """
        Construct a GaussianPulse from the number of cycles over a certain range of pulse widths.

        Parameters
        ----------
        pulse_width
        number_of_cycles
            The number of cycles under the envelope (the cutoff is given by number_of_pulse_widths).
        number_of_pulse_widths
            The number of pulse widths on either side of the center of the pulse to consider when determining the carrier frequency from the number of cycles.
        fluence
        phase
        pulse_center
        kwargs

        Returns
        -------

        """
        omega_carrier = u.pi * number_of_cycles / (number_of_pulse_widths * pulse_width)

        pulse = cls(
            pulse_width=pulse_width,
            omega_carrier=omega_carrier,
            fluence=fluence,
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

        pulse.number_of_cycles = number_of_cycles
        pulse.number_of_pulse_widths = number_of_pulse_widths

        return pulse

    @classmethod
    def from_number_of_cycles_and_amplitude(
        cls,
        pulse_width=DEFAULT_PULSE_WIDTH,
        number_of_cycles=3,
        number_of_pulse_widths=3,
        amplitude=0.1 * u.atomic_electric_field,
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "GaussianPulse":
        """
        Construct a GaussianPulse from the number of cycles over a certain range of pulse widths.

        Parameters
        ----------
        pulse_width
        number_of_cycles
            The number of cycles under the envelope (the cutoff is given by number_of_pulse_widths).
        number_of_pulse_widths
            The number of pulse widths on either side of the center of the pulse to consider when determining the carrier frequency from the number of cycles.
        amplitude
        phase
        pulse_center
        kwargs

        Returns
        -------

        """
        omega_carrier = u.pi * number_of_cycles / (number_of_pulse_widths * pulse_width)

        pulse = cls(
            pulse_width=pulse_width,
            omega_carrier=omega_carrier,
            fluence=cls.fluence_from_pulse_width_and_amplitude(pulse_width, amplitude),
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

        pulse.number_of_cycles = number_of_cycles
        pulse.number_of_pulse_widths = number_of_pulse_widths

        if np.isclose(amplitude / pulse.amplitude, 1):
            pulse.amplitude = amplitude
        else:
            raise ValueError("Given amplitude not close enough to calculated amplitude")

        return pulse

    @classmethod
    def fluence_from_pulse_width_and_amplitude(cls, pulse_width, amplitude):
        return np.sqrt(u.pi) * pulse_width * u.epsilon_0 * u.c * (amplitude ** 2) / 2

    @property
    def photon_energy_carrier(self):
        return u.hbar * self.omega_carrier

    @property
    def frequency_carrier(self):
        return self.omega_carrier / u.twopi

    @property
    def time_fwhm(self):
        return 2 * np.sqrt(2 * np.log(2)) * self.pulse_width

    @property
    def time_fwhm_power(self):
        return self.time_fwhm / np.sqrt(2)

    @property
    def omega_fwhm(self):
        return 2 * np.sqrt(2 * np.log(2)) / self.pulse_width

    @property
    def omega_fwhm_power(self):
        return self.omega_fwhm / np.sqrt(2)

    @property
    def photon_energy_fwhm(self):
        return u.hbar * self.omega_fwhm

    @property
    def photon_energy_fwhm_power(self):
        return self.photon_energy_fwhm / np.sqrt(2)

    @property
    def frequency_fwhm(self):
        return self.omega_fwhm / u.twopi

    @property
    def frequency_fwhm_power(self):
        return self.frequency_fwhm / np.sqrt(2)

    def keldysh_parameter(
        self,
        ionization_potential=u.rydberg,
        test_mass=u.electron_mass,
        test_charge=u.electron_charge,
        keldysh_omega_selector="carrier",
    ) -> float:
        keldysh_omega = {
            "carrier": self.omega_carrier,
            "bandwidth": self.omega_fwhm,
            "bandwidth_power": self.omega_fwhm_power,
        }[keldysh_omega_selector]

        return keldysh_parameter(
            keldysh_omega,
            self.amplitude,
            ionization_potential=ionization_potential,
            test_mass=test_mass,
            test_charge=test_charge,
        )

    def get_electric_field_envelope(self, t):
        tau = np.array(t) - self.pulse_center
        return np.exp(-0.5 * ((tau / self.pulse_width) ** 2))

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        tau = t - self.pulse_center
        amp = self.get_electric_field_envelope(t) * np.cos(
            (self.omega_carrier * tau) + self.phase
        )

        return amp * self.amplitude * super().get_electric_field_amplitude(t)

    def __repr__(self):
        return utils.make_repr(
            self,
            "pulse_width",
            "pulse_center",
            "fluence",
            "phase",
            "photon_energy_carrier",
            "window",
        )

    def __str__(self):
        return utils.make_repr(
            self,
            ("pulse_width", "asec"),
            ("pulse_center", "asec"),
            ("fluence", "J/cm^2"),
            ("phase", "rad"),
            ("photon_energy_carrier", "eV"),
            "window",
        )

    def info(self) -> si.Info:
        info = super().info()

        info.add_field(
            "Pulse Width", utils.fmt_quantity(self.pulse_width, utils.TIME_UNITS)
        )
        info.add_field(
            "Pulse Center", utils.fmt_quantity(self.pulse_center, utils.TIME_UNITS)
        )
        info.add_field(
            "Electric Field Amplitude Prefactor",
            utils.fmt_quantity(self.amplitude, utils.ELECTRIC_FIELD_UNITS),
        )
        info.add_field("Fluence", utils.fmt_quantity(self.fluence, utils.FLUENCE_UNITS))
        info.add_field(
            "Carrier-Envelope Phase", utils.fmt_quantity(self.phase, utils.ANGLE_UNITS)
        )
        info.add_field(
            "Carrier Photon Energy",
            utils.fmt_quantity(self.photon_energy_carrier, utils.ENERGY_UNITS),
        )
        info.add_field(
            "Photon Energy FWHM (Amplitude)",
            utils.fmt_quantity(self.photon_energy_fwhm, utils.ENERGY_UNITS),
        )
        info.add_field(
            "Photon Energy FWHM (Power)",
            utils.fmt_quantity(self.photon_energy_fwhm_power, utils.ENERGY_UNITS),
        )
        info.add_field(
            "Carrier Frequency",
            utils.fmt_quantity(self.frequency_carrier, utils.FREQUENCY_UNITS),
        )
        info.add_field(
            "Frequency FWHM (Amplitude)",
            utils.fmt_quantity(self.frequency_fwhm, utils.FREQUENCY_UNITS),
        )
        info.add_field(
            "Frequency FWHM (Power)",
            utils.fmt_quantity(self.frequency_fwhm_power, utils.FREQUENCY_UNITS),
        )
        info.add_field(
            "Keldysh Parameter (hydrogen ground state)",
            f'{u.uround(self.keldysh_parameter(keldysh_omega_selector = "carrier"))} (Carrier) | {u.uround(self.keldysh_parameter(keldysh_omega_selector = "bandwidth"))} (Bandwidth)',
        )

        info.add_info(self.window.info())

        return info


class SechPulse(UniformLinearlyPolarizedElectricPotential):
    def __init__(
        self,
        pulse_width=DEFAULT_PULSE_WIDTH,
        omega_carrier=DEFAULT_OMEGA_CARRIER,
        fluence=DEFAULT_FLUENCE,
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ):
        """
        Parameters
        ----------
        pulse_width
        omega_carrier
        fluence
        phase
        pulse_center
        kwargs
        """
        if pulse_width <= 0:
            raise exceptions.InvalidPotentialParameter("pulse width must be positive")
        if fluence < 0:
            raise exceptions.InvalidPotentialParameter("fluence must be non-negative")
        if omega_carrier < 0:
            raise exceptions.InvalidPotentialParameter(
                "omega_carrier must be non-negative"
            )

        super().__init__(**kwargs)

        self.omega_carrier = omega_carrier
        self.pulse_width = pulse_width
        self.phase = phase % u.twopi
        self.fluence = fluence
        self.pulse_center = pulse_center

        self.delta_omega = 2 / (u.pi * pulse_width)

        self.amplitude = np.sqrt(self.fluence / (u.epsilon_0 * u.c * self.pulse_width))
        self.amplitude_omega = self.amplitude * self.pulse_width * np.sqrt(u.pi / 2)

    @classmethod
    def from_omega_min(
        cls,
        pulse_width=DEFAULT_PULSE_WIDTH,
        omega_min=DEFAULT_OMEGA_MIN,
        fluence=DEFAULT_FLUENCE,
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "SechPulse":
        """
        Construct a new SechPulse, using omega_min to set the carrier frequency to the same carrier frequency as a sinc pulse with that omega_min and the same pulse width.

        Parameters
        ----------
        pulse_width
        omega_min
        fluence
        phase
        pulse_center

        Returns
        -------

        """
        dummy = SincPulse(pulse_width=pulse_width, omega_min=omega_min)

        pulse = cls(
            pulse_width=pulse_width,
            omega_carrier=dummy.omega_carrier,
            fluence=fluence,
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

        pulse.omega_min = omega_min

        return pulse

    @classmethod
    def from_omega_carrier(cls, *args, **kwargs):
        """Alias for the default constructor."""

        return cls(*args, **kwargs)

    @classmethod
    def from_keldysh_parameter(
        cls,
        pulse_width=DEFAULT_PULSE_WIDTH,
        omega_min=DEFAULT_OMEGA_MIN,
        keldysh_parameter=DEFAULT_KELDYSH_PARAMETER,
        ionization_potential=u.rydberg,
        test_charge=u.electron_charge,
        test_mass=u.electron_mass,
        keldysh_omega_selector="carrier",
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "SechPulse":
        dummy = SincPulse(pulse_width=pulse_width, omega_min=omega_min)
        omega_fwhm = 2 * np.log(2 + np.sqrt(3)) / (2 * pulse_width / u.pi)

        keldysh_omega = {"carrier": dummy.omega_carrier, "bandwidth": omega_fwhm}[
            keldysh_omega_selector
        ]

        amplitude = electric_field_amplitude_from_keldysh_parameter(
            keldysh_parameter,
            keldysh_omega,
            ionization_potential=ionization_potential,
            test_charge=test_charge,
            test_mass=test_mass,
        )

        fluence = u.epsilon_0 * u.c * pulse_width * (amplitude ** 2)

        return cls.from_omega_min(
            pulse_width=pulse_width,
            omega_min=omega_min,
            fluence=fluence,
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

    @property
    def photon_energy_carrier(self):
        return u.hbar * self.omega_carrier

    @property
    def frequency_carrier(self):
        return self.omega_carrier / u.twopi

    @property
    def time_fwhm(self):
        return 2 * np.log(2 + np.sqrt(3)) * self.pulse_width

    @property
    def omega_fwhm(self):
        return 2 * np.log(2 + np.sqrt(3)) / (2 * self.pulse_width / u.pi)

    @property
    def photon_energy_fwhm(self):
        return u.hbar * self.omega_fwhm

    @property
    def frequency_fwhm(self):
        return self.omega_fwhm / u.twopi

    def keldysh_parameter(
        self,
        ionization_potential=u.rydberg,
        test_mass=u.electron_mass,
        test_charge=u.electron_charge,
        keldysh_omega_selector="carrier",
    ):
        keldysh_omega = {"carrier": self.omega_carrier, "bandwidth": self.omega_fwhm}[
            keldysh_omega_selector
        ]

        return keldysh_parameter(
            keldysh_omega,
            self.amplitude,
            ionization_potential=ionization_potential,
            test_mass=test_mass,
            test_charge=test_charge,
        )

    def get_electric_field_envelope(self, t):
        tau = t - self.pulse_center
        return 1 / np.cosh(tau / self.pulse_width)

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        tau = t - self.pulse_center
        amp = self.get_electric_field_envelope(t) * np.cos(
            (self.omega_carrier * tau) + self.phase
        )

        return amp * self.amplitude * super().get_electric_field_amplitude(t)

    def __repr__(self):
        return utils.make_repr(
            self,
            "pulse_width",
            "pulse_center",
            "fluence",
            "phase",
            "photon_energy_carrier",
            "window",
        )

    def __str__(self):
        return utils.make_repr(
            self,
            ("pulse_width", "asec"),
            ("pulse_center", "asec"),
            ("fluence", "J/cm^2"),
            ("phase", "rad"),
            ("photon_energy_carrier", "eV"),
            "window",
        )

    def info(self) -> si.Info:
        info = super().info()

        info.add_field(
            "Pulse Width", utils.fmt_quantity(self.pulse_width, utils.TIME_UNITS)
        )
        info.add_field(
            "Pulse Center", utils.fmt_quantity(self.pulse_center, utils.TIME_UNITS)
        )
        info.add_field(
            "Electric Field Amplitude Prefactor",
            utils.fmt_quantity(self.amplitude, utils.ELECTRIC_FIELD_UNITS),
        )
        info.add_field("Fluence", utils.fmt_quantity(self.fluence, utils.FLUENCE_UNITS))
        info.add_field(
            "Carrier-Envelope Phase", utils.fmt_quantity(self.phase, utils.ANGLE_UNITS)
        )
        info.add_field(
            "Carrier Photon Energy",
            utils.fmt_quantity(self.photon_energy_carrier, utils.ENERGY_UNITS),
        )
        info.add_field(
            "Photon Energy FWHM",
            utils.fmt_quantity(self.photon_energy_fwhm, utils.ENERGY_UNITS),
        )
        info.add_field(
            "Carrier Frequency",
            utils.fmt_quantity(self.frequency_carrier, utils.FREQUENCY_UNITS),
        )
        info.add_field(
            "Frequency FWHM",
            utils.fmt_quantity(self.frequency_fwhm, utils.FREQUENCY_UNITS),
        )
        info.add_field(
            "Keldysh Parameter (hydrogen ground state)",
            f'{u.uround(self.keldysh_parameter(keldysh_omega_selector = "carrier"))} (Carrier) | {u.uround(self.keldysh_parameter(keldysh_omega_selector = "bandwidth"))} (Bandwidth)',
        )

        info.add_info(self.window.info())

        return info


class CosSquaredPulse(UniformLinearlyPolarizedElectricPotential):
    """A sine-squared pulse, parameterized by number of cycles."""

    def __init__(
        self,
        amplitude=0.01 * u.atomic_electric_field,
        wavelength=800 * u.nm,
        number_of_cycles=4,
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.wavelength_carrier = wavelength
        self.number_of_cycles = number_of_cycles
        self.phase = phase
        self.pulse_center = pulse_center

    @classmethod
    def from_omega_carrier(
        cls,
        amplitude=0.1 * u.atomic_electric_field,
        omega_carrier=u.twopi * u.c / (800 * u.nm),
        number_of_cycles=4,
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "CosSquaredPulse":
        wavelength = u.c / (omega_carrier / u.twopi)

        return cls(
            amplitude=amplitude,
            wavelength=wavelength,
            number_of_cycles=number_of_cycles,
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

    @classmethod
    def from_period(
        cls,
        amplitude=0.1 * u.atomic_electric_field,
        period=200 * u.asec,
        number_of_cycles=4,
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "CosSquaredPulse":
        omega = u.twopi / period

        return cls.from_omega_carrier(
            amplitude=amplitude,
            omega_carrier=omega,
            number_of_cycles=number_of_cycles,
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

    @classmethod
    def from_pulse_width(
        cls,
        amplitude=0.1 * u.atomic_electric_field,
        pulse_width=DEFAULT_PULSE_WIDTH,
        number_of_cycles=4,
        phase=DEFAULT_PHASE,
        pulse_center=DEFAULT_PULSE_CENTER,
        **kwargs,
    ) -> "CosSquaredPulse":
        period = pulse_width / number_of_cycles

        return cls.from_period(
            amplitude=amplitude,
            period=period,
            number_of_cycles=number_of_cycles,
            phase=phase,
            pulse_center=pulse_center,
            **kwargs,
        )

    @property
    def frequency_carrier(self):
        return u.c / self.wavelength_carrier

    @property
    def period_carrier(self):
        return 1 / self.frequency_carrier

    @property
    def omega_carrier(self):
        return u.twopi * u.c / self.wavelength_carrier

    @property
    def pulse_width(self):
        return self.number_of_cycles * self.period_carrier

    @property
    def total_time(self):
        return self.period_carrier * self.number_of_cycles

    def get_electric_field_envelope(self, t):
        tau = t - self.pulse_center
        return np.where(
            np.less_equal(-self.total_time / 2, tau)
            * np.less_equal(tau, self.total_time / 2),
            np.cos(self.omega_carrier * tau / (2 * self.number_of_cycles)) ** 2,
            0,
        )

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        tau = t - self.pulse_center
        amp = self.get_electric_field_envelope(t) * np.cos(
            (self.omega_carrier * tau) + self.phase
        )

        return amp * self.amplitude * super().get_electric_field_amplitude(t)

    def info(self) -> si.Info:
        info = super().info()

        info.add_field(
            "Amplitude", f"{u.uround(self.amplitude, u.atomic_electric_field)} a.u."
        )
        info.add_field(
            "Center Wavelength",
            f"{u.uround(self.wavelength_carrier, u.nm)} u.nm | {u.uround(self.wavelength_carrier, um)} um",
        )
        info.add_field("Number of Cycles", self.number_of_cycles)

        return info


class GenericElectricPotential(UniformLinearlyPolarizedElectricPotential):
    """Generate an electric field from a Fourier transform of a frequency-amplitude spectrum."""

    def __init__(
        self,
        frequencies,
        amplitudes,
        phases=0,
        fluence=None,
        name="GenericElectricField",
        **kwargs,
    ):
        """

        Parameters
        ----------
        amplitude_function
        phase_function
        frequency_upper_limit
        frequency_points
        fluence
            Nominal fluence for the pulse.
        name
        extra_information
        kwargs
        """
        super().__init__(**kwargs)

        self.name = name
        self.fluence = fluence

        self.frequency = frequencies
        self.df = np.abs(self.frequency[1] - self.frequency[0])

        self.amplitude_vs_frequency = amplitudes
        self.phase_vs_frequency = phases

        self.complex_amplitude_vs_frequency = self.amplitude_vs_frequency * np.exp(
            1j * self.phase_vs_frequency
        )

        self.times = nfft.fftshift(nfft.fftfreq(len(self.frequency), self.df))
        self.dt = np.abs(self.times[1] - self.times[0])

        self.complex_electric_field_vs_time = nfft.ifft(
            nfft.ifftshift(self.complex_amplitude_vs_frequency), norm="ortho"
        )

    @classmethod
    def from_funcs(
        cls, frequencies, amplitude_function, phase_function=lambda f: 0, **kwargs
    ):
        return cls(
            frequencies,
            amplitude_function(frequencies),
            phase_function(frequencies),
            **kwargs,
        )

    @classmethod
    def from_pulse(cls, pulse, times, phase_function=lambda f: 0, **kwargs):
        """
        Construct a GenericElectricField pulse that has the same power spectrum as the provided pulse.

        The power spectrum is extracted by taking the FFT of the pulse over the times.

        Parameters
        ----------
        pulse
        phase_function

        Returns
        -------

        """
        electric_field_vs_time = pulse.get_electric_field_amplitude(times)
        dt = np.abs(times[1] - times[0])
        frequencies = nfft.fftshift(nfft.fftfreq(len(times), dt))
        amplitudes = nfft.fftshift(nfft.fft(electric_field_vs_time, norm="ortho"))

        return cls.from_funcs(
            frequencies, lambda a: amplitudes, phase_function, **kwargs
        )

    @property
    def omega(self):
        return u.twopi * self.frequency

    @property
    def dw(self):
        return u.twopi * self.df

    @property
    def power_vs_frequency(self):
        return np.abs(self.complex_amplitude_vs_frequency) ** 2

    @property
    def fluence_numeric(self):
        from_field = (
            u.epsilon_0
            * u.c
            * integ.simps(
                y=np.real(self.complex_electric_field_vs_time) ** 2, dx=self.dt
            )
        )
        # from_spectrum = u.epsilon_0 * u.c * integ.simps(y = self.complex_amplitude_vs_frequency ** 2,
        #                                             dx = self.df)

        # return (from_field + from_spectrum) / 2
        return from_field

    def __str__(self):
        return utils.make_repr(self, "name")

    def __repr__(self):
        return utils.make_repr(self, "name")

    def get_electric_field_amplitude(self, t):
        try:
            index, value, target = si.utils.find_nearest_entry(self.times, t)
            amp = self.complex_electric_field_vs_time[index]
        except ValueError:  # t is actually an ndarray
            amp = np.zeros(len(t), dtype=np.complex128) * np.NaN
            for ii, time in enumerate(t):
                index, value, target = si.utils.find_nearest_entry(self.times, time)
                amp[ii] = self.complex_electric_field_vs_time[index]

        return np.real(amp) * super().get_electric_field_amplitude(t)


def DC_correct_electric_potential(
    electric_potential: UniformLinearlyPolarizedElectricPotential, times: np.array
) -> potential.PotentialEnergySum:
    """
    Correct the given electric potential so that it has no net electric field over the given ``times``.

    Parameters
    ----------
    electric_potential
    times

    Returns
    -------

    """

    def func_to_minimize(amp, original_pulse):
        test_correction_field = Rectangle(
            start_time=times[0],
            end_time=times[-1],
            amplitude=amp,
            window=electric_potential.window,
        )
        test_pulse = original_pulse + test_correction_field

        return np.abs(
            test_pulse.get_electric_field_integral_numeric_cumulative(times)[-1]
        )

    correction_amp = optim.minimize_scalar(
        func_to_minimize, args=(electric_potential,)
    ).x
    correction_field = Rectangle(
        start_time=times[0],
        end_time=times[-1],
        amplitude=correction_amp,
        window=electric_potential.window,
    )

    return electric_potential + correction_field


class FluenceCorrector(UniformLinearlyPolarizedElectricPotential):
    def __init__(self, electric_potential, times, target_fluence):
        self.electric_potential = electric_potential
        self.target_fluence = target_fluence

        fluence = electric_potential.get_fluence_numeric(times)
        self.amplitude_correction_ratio = np.sqrt(target_fluence / fluence)

        super().__init__()

    def get_electric_field_amplitude(self, t):
        return (
            self.electric_potential.get_electric_field_amplitude(t)
            * self.amplitude_correction_ratio
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(electric_potential = {self.electric_potential}, target_fluence = {self.target_fluence}, amplitude_correction_ratio = {self.amplitude_correction_ratio})"

    def info(self):
        info = super().info()

        info.add_info(self.electric_potential.info())

        info.add_field(
            "Target Fluence", f"{u.uround(self.target_fluence, u.Jcm2)} J/cm2"
        )
        info.add_field("Amplitude Correction Ratio", self.amplitude_correction_ratio)

        return info
