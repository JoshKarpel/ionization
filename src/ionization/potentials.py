import functools as ft
import logging

import numpy as np
import numpy.fft as nfft
import scipy.integrate as integ
import scipy.optimize as opt

import simulacra as si
from simulacra.units import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PotentialEnergy(si.Summand):
    """A class representing some kind of potential energy. Can be summed to form a PotentialEnergySum."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.summation_class = PotentialEnergySum


class PotentialEnergySum(si.Sum, PotentialEnergy):
    """A class representing a combination of potential energies."""

    container_name = 'potentials'

    def get_electric_field_amplitude(self, t):
        return sum(x.get_electric_field_amplitude(t) for x in self._container)

    def get_electric_field_integral_numeric(self, t):
        return sum(x.get_electric_field_integral_numeric(t) for x in self._container)

    def get_vector_potential_amplitude(self, t):
        return sum(x.get_vector_potential_amplitude(t) for x in self._container)

    def get_vector_potential_amplitude_numeric(self, times, rule = 'simps'):
        return sum(x.get_vector_potential_amplitude_numeric(times, rule = rule) for x in self._container)

    def get_electric_field_integral_numeric_cumulative(self, times):
        """Return the integral of the electric field amplitude from the start of times for each interval in times."""
        return sum(x.get_electric_field_integral_numeric_cumulative(times) for x in self._container)

    def get_vector_potential_amplitude_numeric_cumulative(self, times):
        return sum(x.get_vector_potential_amplitude_numeric_cumulative(times) for x in self._container)

    def get_fluence_numeric(self, times, rule = 'simps'):
        return sum(x.get_fluence_numeric(times, rule = rule) for x in self._container)


class NoPotentialEnergy(PotentialEnergy):
    """A class representing no potential energy from any source."""

    def __call__(self, *args, **kwargs):
        """Return 0 for any arguments."""
        return 0


class TimeWindow(si.Summand):
    """A class representing a time-window that can be attached to another potential."""

    def __init__(self):
        super().__init__()
        self.summation_class = TimeWindowSum


class TimeWindowSum(si.Sum, TimeWindow):
    """A class representing a combination of time-windows."""

    container_name = 'windows'

    def __call__(self, *args, **kwargs):
        return ft.reduce(lambda a, b: a * b, (x(*args, **kwargs) for x in self._container))  # windows should be multiplied together, not summed


class NoTimeWindow(TimeWindow):
    """A class representing the lack of a time-window."""

    def __call__(self, t):
        return 1


class Mask(si.Summand):
    """A class representing a spatial 'mask' that can be applied to the wavefunction to reduce it in certain regions."""

    def __init__(self):
        super().__init__()
        self.summation_class = MaskSum


class MaskSum(si.Sum, Mask):
    """A class representing a combination of masks."""

    container_name = 'masks'

    def __call__(self, *args, **kwargs):
        return ft.reduce(lambda a, b: a * b, (x(*args, **kwargs) for x in self._container))  # masks should be multiplied together, not summed


class NoMask(Mask):
    """A class representing the lack of a mask."""

    def __call__(self, *args, **kwargs):
        return 1


class Coulomb(PotentialEnergy):
    """A class representing the electric potential energy caused by the Coulomb potential."""

    def __init__(self, charge = 1 * proton_charge):
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
        return coulomb_constant * self.charge * test_charge / r

    def info(self):
        info = super().info()

        info.add_field('Charge', f'{uround(self.charge, proton_charge, 3)} e')

        return info


class SoftCoulomb(PotentialEnergy):
    """A class representing the electric potential energy caused by the Coulomb potential."""

    def __init__(self, charge = 1 * proton_charge, softening_distance = .05 * bohr_radius):
        """
        Construct a Coulomb from a charge.

        :param charge: the charge of the particle providing the potential
        """
        super().__init__()

        self.charge = charge
        self.softening_distance = softening_distance

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
        return coulomb_constant * self.charge * test_charge / np.sqrt((r ** 2) + (self.softening_distance ** 2))

    def info(self):
        info = super().info()

        info.add_field('Charge', f'{uround(self.charge, proton_charge)} e')
        info.add_field('Softening Distance', f'{uround(self.softening_distance, bohr_radius)} Bohr radii')

        return info


class HarmonicOscillator(PotentialEnergy):
    """A PotentialEnergy representing the potential energy of a harmonic oscillator."""

    def __init__(self, spring_constant = 4.20521 * N / m, center = 0 * nm, cutoff_distance = None):
        """Construct a HarmonicOscillator object with the given spring constant and center position."""
        self.spring_constant = spring_constant
        self.center = center

        self.cutoff_distance = cutoff_distance

        super().__init__()

    @classmethod
    def from_frequency_and_mass(cls, omega = 1.5192675e15 * Hz, mass = electron_mass, **kwargs):
        """Return a HarmonicOscillator constructed from the given angular frequency and mass."""
        return cls(spring_constant = mass * (omega ** 2), **kwargs)

    @classmethod
    def from_ground_state_energy_and_mass(cls, ground_state_energy = 0.5 * eV, mass = electron_mass, **kwargs):
        """
        Return a HarmonicOscillator constructed from the given ground state energy and mass.

        Note: the ground state energy is half of the energy spacing of the oscillator.
        """
        return cls.from_frequency_and_mass(omega = 2 * ground_state_energy / hbar, mass = mass, **kwargs)

    @classmethod
    def from_energy_spacing_and_mass(cls, energy_spacing = 1 * eV, mass = electron_mass, **kwargs):
        """
        Return a HarmonicOscillator constructed from the given state energy spacing and mass.

        Note: the ground state energy is half of the energy spacing of the oscillator.
        """
        return cls.from_frequency_and_mass(omega = energy_spacing / hbar, mass = mass, **kwargs)

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
        return self.omega(mass) / twopi

    def __str__(self):
        return '{}(spring_constant = {} N/m, center = {} nm)'.format(self.__class__.__name__, np.around(self.spring_constant, 3), uround(self.center, nm, 3))

    def __repr__(self):
        return '{}(spring_constant = {}, center = {})'.format(self.__class__.__name__, self.spring_constant, self.center)

    def info(self):
        info = super().info()

        info.add_field('Spring Constant', f'{uround(self.spring_constant, 1, 3)} N/m | {uround(self.spring_constant, atomic_force, 3)} a.u./Bohr Radius')
        info.add_field('Center', f'{uround(self.center, bohr_radius, 3)} Bohr radii | {uround(self.center, nm, 3)} nm')
        if self.cutoff_distance is not None:
            info.add_field('Cutoff Distance', f'{uround(self.cutoff_distance, bohr_radius, 3)} Bohr radii | {uround(self.cutoff_distance, nm, 3)} nm')

        return info


class FiniteSquareWell(PotentialEnergy):
    def __init__(self, potential_depth = 1 * eV, width = 10 * nm, center = 0 * nm):
        self.potential_depth = potential_depth
        self.width = width
        self.center = center

        super().__init__()

    def __str__(self):
        return si.utils.field_str(self, ('potential_depth', 'eV'), ('width', 'nm'), ('center', 'nm'))

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

        info.add_field('Potential Depth', f'{uround(self.potential_depth, eV)} eV')
        info.add_field('Width', f'{uround(self.width, nm, 3)} nm')
        info.add_field('Center', f'{uround(self.center, nm, 3)} nm')

        return info


class RadialImaginary(PotentialEnergy):
    def __init__(self, center = 20 * bohr_radius, width = 2 * bohr_radius, decay_time = 100 * asec):
        """
        Construct a RadialImaginary potential. The potential is shaped like a Gaussian wrapped around a ring and has an imaginary amplitude.

        A positive/negative amplitude yields an imaginary potential that causes decay/amplification.

        :param center: the radial coordinate to center the potential on
        :param width: the width (FWHM) of the Gaussian
        :param decay_time: the decay time (1/e time) of a wavefunction packet at the peak of the imaginary potential
        """
        self.center = center
        self.width = width
        self.decay_time = decay_time
        self.decay_rate = 1 / decay_time

        self.prefactor = -1j * self.decay_rate * hbar

        super().__init__()

    def __repr__(self):
        return '{}(center = {}, width = {}, decay_time = {})'.format(self.__class__.__name__, self.center, self.width, self.decay_time)

    def __str__(self):
        return '{}(center = {} Bohr radii, width = {} Bohr radii, decay time = {} as)'.format(self.__class__.__name__,
                                                                                              uround(self.center, bohr_radius, 3),
                                                                                              uround(self.width, bohr_radius, 3),
                                                                                              uround(self.decay_time, asec))

    def __call__(self, *, r, **kwargs):
        return self.prefactor * np.exp(-(((r - self.center) / self.width) ** 2))


def DC_correct_electric_potential(electric_field, times):
    def func_to_minimize(amp, original_pulse):
        test_correction_field = Rectangle(start_time = times[0], end_time = times[-1], amplitude = amp, window = electric_field.window)
        test_pulse = original_pulse + test_correction_field

        return np.abs(test_pulse.get_electric_field_integral_numeric_cumulative(times)[-1])

    correction_amp = opt.minimize_scalar(func_to_minimize, args = (electric_field,)).x
    correction_field = Rectangle(start_time = times[0], end_time = times[-1], amplitude = correction_amp, window = electric_field.window)

    return electric_field + correction_field


class UniformLinearlyPolarizedElectricPotential(PotentialEnergy):
    def __init__(self, window = NoTimeWindow()):
        super().__init__()

        self.window = window

    def __str__(self):
        return ' with {}'.format(self.window)

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        return self.window(t)

    def get_vector_potential_amplitude(self, t):
        raise NotImplementedError

    def __call__(self, *, t, distance_along_polarization, test_charge, **kwargs):
        return -distance_along_polarization * test_charge * self.get_electric_field_amplitude(t)

    def get_electric_field_integral_numeric(self, times, rule = 'simps'):
        """
        Return the electric field integral from ``times[0]`` to ``times[-1]``.

        Parameters
        ----------
        times
        rule : {``'trapz'``, ``'simps'``}
        Returns
        -------
        """
        return getattr(integ, rule)(y = self.get_electric_field_amplitude(times),
                                    x = times)

    def get_vector_potential_amplitude_numeric(self, times, rule = 'simps'):
        return -self.get_electric_field_integral_numeric(times, rule = rule)

    def get_electric_field_integral_numeric_cumulative(self, times):
        """Return the integral of the electric field amplitude from the start of times for each interval in times."""
        return integ.cumtrapz(y = self.get_electric_field_amplitude(times),
                              x = times,
                              initial = 0)

    def get_vector_potential_amplitude_numeric_cumulative(self, times):
        return -self.get_electric_field_integral_numeric_cumulative(times)

    def get_fluence_numeric(self, times, rule = 'simps'):
        return epsilon_0 * c * getattr(integ, rule)(y = np.abs(self.get_electric_field_amplitude(times)) ** 2,
                                                    x = times,
                                                    initial = 0)


class NoElectricPotential(UniformLinearlyPolarizedElectricPotential):
    """A class representing the lack of an electric field."""

    def __str__(self):
        return self.__class__.__name__ + super().__str__()

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        return np.zeros(np.shape(t)) * super().get_electric_field_amplitude(t)

    def get_vector_potential_amplitude(self, t):
        return np.zeros(np.shape(t))


class Rectangle(UniformLinearlyPolarizedElectricPotential):
    """A class representing an electric with a sharp turn-on and turn-off time."""

    def __init__(self, start_time = 0 * asec, end_time = 50 * asec, amplitude = 1 * atomic_electric_field, **kwargs):
        """
        Construct a Rectangle from a start time, end time, and electric field amplitude.

        :param start_time: the time the electric field turns on
        :param end_time: the time the electric field turns off
        :param amplitude: the amplitude of the electric field between start_time and end_time
        :param kwargs: kwargs are passed to UniformLinearlyPolarizedElectricField
        """
        super().__init__(**kwargs)

        self.start_time = start_time
        self.end_time = end_time
        self.amplitude = amplitude

    def __str__(self):
        out = '{}(start time = {} as, end time = {} as, amplitude = {} AEF)'.format(self.__class__.__name__,
                                                                                    uround(self.start_time, asec),
                                                                                    uround(self.end_time, asec),
                                                                                    uround(self.amplitude, atomic_electric_field, 3))

        return out + super().__str__()

    def __repr__(self):
        out = '{}(start_time = {}, end_time = {}, amplitude = {}, window = {})'.format(self.__class__.__name__,
                                                                                       self.start_time,
                                                                                       self.end_time,
                                                                                       self.amplitude,
                                                                                       repr(self.window))

        return out

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        cond = np.greater_equal(t, self.start_time) * np.less_equal(t, self.end_time)
        on = np.ones(np.shape(t))
        off = np.zeros(np.shape(t))

        out = np.where(cond, on, off) * self.amplitude * super().get_electric_field_amplitude(t)

        return out

    def info(self):
        info = super().info()

        info.add_field('Amplitude', f'{uround(self.amplitude, atomic_electric_field)} AEF')
        info.add_field('Time On', f'{uround(self.start_time, asec)} as')
        info.add_field('Time Off', f'{uround(self.end_time, asec)} as')

        info.add_info(self.window.info())

        return info


class SineWave(UniformLinearlyPolarizedElectricPotential):
    def __init__(self, omega, amplitude = 1 * atomic_electric_field, phase = 0, **kwargs):
        """
        Construct a SineWave from the angular frequency, electric field amplitude, and phase.

        :param omega: the photon angular frequency
        :param amplitude: the electric field amplitude
        :param phase: the phase of the electric field (0 corresponds to a sine wave)
        :param kwargs: kwargs are passed to UniformLinearlyPolarizedElectricField
        """
        super().__init__(**kwargs)

        self.omega = omega
        self.phase = phase % twopi
        self.amplitude = amplitude

    def __str__(self):
        out = '{}(omega = 2pi * {} THz, wavelength = {} nm, photon energy = {} eV, amplitude = {} AEF, phase = 2pi * {})'.format(self.__class__.__name__,
                                                                                                                                 uround(self.frequency, THz),
                                                                                                                                 uround(self.wavelength, nm, 3),
                                                                                                                                 uround(self.photon_energy, eV),
                                                                                                                                 uround(self.amplitude, atomic_electric_field, 3),
                                                                                                                                 uround(self.phase, twopi, 3))

        return out + super().__str__()

    def __repr__(self):
        out = '{}(omega = {}, amplitude = {}, phase = {}, window = {})'.format(self.__class__.__name__,
                                                                               self.omega,
                                                                               self.amplitude,
                                                                               self.phase,
                                                                               repr(self.window))

        return out

    @classmethod
    def from_frequency(cls, frequency, amplitude = 1 * atomic_electric_field, phase = 0, **kwargs):
        """
        Construct a SineWave from the frequency, electric field amplitude, and phase.

        :param frequency: the photon frequency
        :param amplitude: the electric field amplitude
        :param phase: the phase of the electric field (0 corresponds to a sine wave)
        :param kwargs: kwargs are passed to UniformLinearlyPolarizedElectricField
        :return: a SineWave instance
        """
        return cls(frequency * twopi, amplitude = amplitude, phase = phase, **kwargs)

    @classmethod
    def from_period(cls, period, amplitude = 1 * atomic_electric_field, phase = 0, **kwargs):
        """

        Parameters
        ----------
        period
        amplitude
        phase
        kwargs

        Returns
        -------

        """
        return cls.from_frequency(1 / period, amplitude = amplitude, phase = phase, **kwargs)

    @classmethod
    def from_wavelength(cls, wavelength, amplitude = 1 * atomic_electric_field, phase = 0, **kwargs):
        """
        Construct a :class:`SineWave` from a wavelength and amplitude.

        Parameters
        ----------
        wavelength
            The wavelength of the sine wave.
        amplitude
            The maximum amplitude of the electric field of the sine wave.
        phase
            The phase of the sine wave. ``0`` is sine, ``pi`` is cosine.
        kwargs

        Returns
        -------
        :class:`SineWave`
        """
        return cls(c / wavelength, amplitude = amplitude, phase = phase, **kwargs)

    @classmethod
    def from_photon_energy(cls, photon_energy, amplitude = 1 * atomic_electric_field, phase = 0, **kwargs):
        """
        Construct a SineWave from the photon energy, electric field amplitude, and phase.

        :param photon_energy: the photon energy
        :param amplitude: the electric field amplitude
        :param phase: the phase of the electric field (0 corresponds to a sine wave)
        :param kwargs: kwargs are passed to UniformLinearlyPolarizedElectricField
        :return: a SineWave instance
        """
        return cls(photon_energy / hbar, amplitude = amplitude, phase = phase, **kwargs)

    @classmethod
    def from_photon_energy_and_intensity(cls, photon_energy, intensity = 1e14 * W / (cm ** 2), phase = 0, **kwargs):
        """
        Construct a SineWave from the photon energy, electric field amplitude, and phase.

        :param photon_energy: the photon energy
        :param phase: the phase of the electric field (0 corresponds to a sine wave)
        :param kwargs: kwargs are passed to UniformLinearlyPolarizedElectricField
        :return: a SineWave instance
        """
        return cls(photon_energy / hbar, amplitude = np.sqrt(2 * intensity / (epsilon_0 * c)), phase = phase, **kwargs)

    @property
    def intensity(self):
        """Cycle-averaged intensity"""
        return .5 * epsilon_0 * c * (self.amplitude ** 2)

    @property
    def frequency(self):
        return self.omega / twopi

    @frequency.setter
    def frequency(self, frequency):
        self.omega = frequency * twopi

    @property
    def period(self):
        return 1 / self.frequency

    @period.setter
    def period(self, period):
        self.frequency = 1 / period

    @property
    def wavelength(self):
        return c / self.frequency

    @wavelength.setter
    def wavelength(self, wavelength):
        self.frequency = c / wavelength

    @property
    def photon_energy(self):
        return hbar * self.omega

    @photon_energy.setter
    def photon_energy(self, photon_energy):
        self.omega = photon_energy / hbar

    def keldysh_parameter(self, ionization_potential, test_mass = electron_mass, test_charge = electron_charge):
        return self.omega * np.sqrt(2 * test_mass * np.abs(ionization_potential)) / (-test_charge * self.amplitude)

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        return np.sin((self.omega * t) + self.phase) * self.amplitude * super().get_electric_field_amplitude(t)

    def get_peak_amplitude(self):
        return self.amplitude

    # def get_peak_power_density(self):
    #     return 0.5 * c * epsilon_0 * (np.abs(self.amplitude) ** 2)  # TODO: check factor of 1/2 here

    def info(self):
        info = super().info()

        info.add_field('Amplitude', f'{uround(self.amplitude, atomic_electric_field)} AEF')
        info.add_field('Intensity', f'{uround(self.intensity, TW / (cm ** 2))} TW/cm^2')
        info.add_field('Photon Energy', f'{uround(self.photon_energy, eV)} eV')
        info.add_field('Frequency', f'{uround(self.frequency, THz)} THz')
        info.add_field('Period', f'{uround(self.period, asec)} as | {uround(self.period, fsec)} fs')
        info.add_field('Wavelength', f'{uround(self.wavelength, nm)} nm | {uround(self.wavelength, bohr_radius)} Bohr radii')

        info.add_info(self.window.info())

        return info


class SumOfSinesPulse(UniformLinearlyPolarizedElectricPotential):
    def __init__(self, pulse_width = 200 * asec, pulse_frequency_ratio = 5, fluence = 1 * Jcm2, phase = 0, pulse_center = 0 * asec,
                 number_of_modes = 71,
                 **kwargs):
        """

        Parameters
        ----------
        pulse_width
        pulse_frequency_ratio
        fluence
        phase
        pulse_center
        number_of_modes
        kwargs
        """
        super().__init__(**kwargs)

        if phase != 0:
            raise ValueError('phase != 0 not implemented for SumOfSinesPulse')

        self.pulse_width = pulse_width

        self.number_of_modes = number_of_modes
        self.mode_spacing = twopi / (self.number_of_modes * self.pulse_width)
        self.pulse_frequency_ratio = pulse_frequency_ratio

        self.omega_min = self.pulse_frequency_ratio * self.mode_spacing

        # self.phase = phase
        if phase != 0:
            logger.warning('Phase not implemented for SumOfSines pulse!')

        self.fluence = fluence
        self.pulse_center = pulse_center

        self.delta_omega = twopi / self.pulse_width
        self.omega_max = self.omega_min + self.delta_omega
        # self.omega_carrier = (self.omega_min + self.omega_max) / 2

        self.amplitude_omega = np.sqrt(self.fluence * self.delta_omega / (twopi * number_of_modes * c * epsilon_0))
        self.amplitude_time = self.amplitude_omega

        self.cycle_period = twopi / self.mode_spacing

    @property
    def photon_energy_min(self):
        return hbar * self.omega_min

    @property
    def photon_energy_max(self):
        return hbar * self.omega_max

    @property
    def frequency_min(self):
        return self.omega_min / twopi

    @property
    def frequency_max(self):
        return self.omega_max / twopi

    @property
    def frequency_delta(self):
        return self.delta_omega / twopi

    @property
    def amplitude_per_frequency(self):
        return np.sqrt(twopi) * self.amplitude_omega

    def __str__(self):
        out = si.utils.field_str(self,
                                 ('pulse_width', 'asec'),
                                 ('pulse_center', 'asec'),
                                 ('fluence', 'J/cm^2'),
                                 # 'phase',
                                 ('photon_energy_min', 'eV'),
                                 ('photon_energy_max', 'eV'),
                                 )

        return out + super().__str__()

    def __repr__(self):
        return si.utils.field_str(self,
                                  'pulse_width',
                                  'pulse_center',
                                  'fluence',
                                  # 'phase',
                                  'photon_energy_min',
                                  'photon_energy_max',
                                  )

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        tau = t - self.pulse_center

        cond = np.not_equal(tau, 0)

        on = np.real(np.exp(-1j * self.pulse_frequency_ratio * self.mode_spacing * tau) * (1 - np.exp(-1j * self.mode_spacing * self.number_of_modes * tau)) / (1 - np.exp(-1j * self.mode_spacing * tau)))
        off = self.number_of_modes

        amp = np.where(cond, on, off)

        return amp * self.amplitude_time * super().get_electric_field_amplitude(t)


class SincPulse(UniformLinearlyPolarizedElectricPotential):
    """
    Attributes
    ----------
    pulse_width
    fluence
    phase
    pulse_center

    time_fwhm
    time_fwhm_power

    omega_carrier
    omega_min
    omega_max
    delta_omega
        Sue me.

    frequency_carrier
    frequency_min
    frequency_max
    frequency_delta

    photon_energy_carrier
    photon_energy_min
    photon_energy_max

    amplitude_per_frequency
    """

    def __init__(self, pulse_width = 200 * asec, omega_min = twopi * 500 * THz, fluence = 1 * J / (cm ** 2), phase = 0, pulse_center = 0 * asec,
                 **kwargs):
        """

        Parameters
        ----------
        pulse_width
        omega_min
        fluence
        phase
        pulse_center
        kwargs
        """
        super().__init__(**kwargs)

        self.omega_min = omega_min
        self.pulse_width = pulse_width
        self.phase = phase
        self.fluence = fluence
        self.pulse_center = pulse_center

        self.delta_omega = twopi / self.pulse_width
        self.omega_max = self.omega_min + self.delta_omega
        self.omega_carrier = (self.omega_min + self.omega_max) / 2

        self.amplitude_omega = np.sqrt(self.fluence / (2 * epsilon_0 * c * self.delta_omega))
        self.amplitude_time = np.sqrt(self.fluence * self.delta_omega / (pi * epsilon_0 * c))

    @classmethod
    def from_omega_carrier(cls, pulse_width = 200 * asec, omega_carrier = twopi * 3500 * THz, fluence = 1 * J / (cm ** 2), phase = 0, pulse_center = 0 * asec,
                           **kwargs):
        delta_omega = twopi / pulse_width
        omega_min = omega_carrier - delta_omega / 2

        return cls(pulse_width = pulse_width, omega_min = omega_min, fluence = fluence, phase = phase, pulse_center = pulse_center, **kwargs)

    @property
    def photon_energy_min(self):
        return hbar * self.omega_min

    @property
    def photon_energy_carrier(self):
        return hbar * self.omega_carrier

    @property
    def photon_energy_max(self):
        return hbar * self.omega_max

    @property
    def photon_energy_bandwidth(self):
        return self.photon_energy_max - self.photon_energy_min

    @property
    def frequency_min(self):
        return self.omega_min / twopi

    @property
    def frequency_carrier(self):
        return self.omega_carrier / twopi

    @property
    def frequency_max(self):
        return self.omega_max / twopi

    @property
    def frequency_delta(self):
        return self.delta_omega / twopi

    @property
    def amplitude_per_frequency(self):
        return np.sqrt(twopi) * self.amplitude_omega

    def keldysh_parameter(self, ionization_potential, test_mass = electron_mass, test_charge = electron_charge):
        if self.phase != 0:
            logger.warning(f'Evaluated Keldysh parameter for {self}, but phase is not zero.')
        amp = self.get_electric_field_amplitude(0)
        return self.omega_carrier * np.sqrt(2 * test_mass * np.abs(ionization_potential)) / (-test_charge * amp)

    def __str__(self):
        out = si.utils.field_str(self,
                                 ('pulse_width', 'asec'),
                                 ('pulse_center', 'asec'),
                                 ('fluence', 'J/cm^2'),
                                 'phase',
                                 ('photon_energy_min', 'eV'),
                                 ('photon_energy_carrier', 'eV'),
                                 ('photon_energy_max', 'eV'),
                                 )

        return out + super().__str__()

    def __repr__(self):
        return si.utils.field_str(self,
                                  'pulse_width',
                                  'pulse_center',
                                  'fluence',
                                  'phase',
                                  'photon_energy_min',
                                  'photon_energy_carrier',
                                  'photon_energy_max',
                                  'omega_carrier',
                                  )

    def get_electric_field_envelope(self, t):
        tau = np.array(t) - self.pulse_center
        return si.math.sinc(self.delta_omega * tau / 2)

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        tau = np.array(t) - self.pulse_center
        amp = self.get_electric_field_envelope(t) * np.cos((self.omega_carrier * tau) + self.phase)

        return amp * self.amplitude_time * super().get_electric_field_amplitude(t)

    def info(self):
        info = super().info()

        info.add_field('Pulse Width', f'{uround(self.pulse_width, asec)} as | {uround(self.pulse_width, fsec, 3)} fs')
        info.add_field('Fluence', f'{uround(self.fluence, Jcm2)} J/cm^2')
        info.add_field('Carrier Photon Energy', f'{uround(self.photon_energy_carrier, eV)} eV')
        info.add_field('Photon Energy Range', f'{uround(self.photon_energy_min, eV)} eV to {uround(self.photon_energy_max, eV)} eV')
        info.add_field('Photon Energy Bandwidth', f'{uround(self.photon_energy_bandwidth, eV)} eV')
        info.add_field('Carrier Frequency', f'{uround(self.frequency_carrier, THz)} THz')
        info.add_field('Frequency Range', f'{uround(self.frequency_min, THz)} THz to {uround(self.frequency_max, THz)} THz')
        info.add_field('Frequency Bandwidth', f'{uround(self.frequency_delta, THz)} THz')

        info.add_info(self.window.info())

        return info


class GaussianPulse(UniformLinearlyPolarizedElectricPotential):
    def __init__(self, pulse_width = 200 * asec, omega_carrier = twopi * 3500 * THz, fluence = 1 * J / (cm ** 2), phase = 0, pulse_center = 0 * asec,
                 **kwargs):
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
        super().__init__(**kwargs)

        self.omega_carrier = omega_carrier
        self.pulse_width = pulse_width
        self.phase = phase
        self.fluence = fluence
        self.pulse_center = pulse_center

        self.amplitude_time = np.sqrt(2 * self.fluence / (np.sqrt(pi) * epsilon_0 * c * self.pulse_width))
        self.amplitude_omega = self.amplitude_time * self.pulse_width / 2

    @property
    def photon_energy_carrier(self):
        return hbar * self.omega_carrier

    @property
    def frequency_carrier(self):
        return self.omega_carrier / twopi

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
        return hbar * self.omega_fwhm

    @property
    def photon_energy_fwhm_power(self):
        return self.photon_energy_fwhm / np.sqrt(2)

    @property
    def frequency_fwhm(self):
        return self.omega_fwhm / twopi

    @property
    def frequency_fwhm_power(self):
        return self.frequency_fwhm / np.sqrt(2)

    def __str__(self):
        out = si.utils.field_str(self,
                                 ('pulse_width', 'asec'),
                                 ('pulse_center', 'asec'),
                                 ('fluence', 'J/cm^2'),
                                 'phase',
                                 ('frequency_carrier', 'THz'),
                                 ('photon_energy_carrier', 'eV'),
                                 )

        return out + super().__str__()

    def __repr__(self):
        return si.utils.field_str(self,
                                  'pulse_width',
                                  'pulse_center',
                                  'fluence',
                                  'phase',
                                  'omega_carrier',
                                  'photon_energy_carrier',
                                  )

    def keldysh_parameter(self, ionization_potential, test_mass = electron_mass, test_charge = electron_charge):
        if self.phase != 0:
            logger.warning(f'Evaluated Keldysh parameter for {self}, but phase is not zero.')
        amp = self.get_electric_field_amplitude(0)
        return self.omega_carrier * np.sqrt(2 * test_mass * np.abs(ionization_potential)) / (-test_charge * amp)

    def get_electric_field_envelope(self, t):
        tau = np.array(t) - self.pulse_center
        return np.exp(-0.5 * ((tau / self.pulse_width) ** 2))

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        tau = t - self.pulse_center
        amp = self.get_electric_field_envelope(t) * np.cos((self.omega_carrier * tau) + self.phase)

        return amp * self.amplitude_time * super().get_electric_field_amplitude(t)

    def info(self):
        info = super().info()

        info.add_field('Pulse Width', f'{uround(self.pulse_width, asec)} as | {uround(self.pulse_width, fsec)} fs')
        info.add_field('Fluence', f'{uround(self.fluence, Jcm2)} J/cm^2')
        info.add_field('Carrier Photon Energy', f'{uround(self.photon_energy_carrier, eV)} eV')
        info.add_field('Photon Energy FWHM (Amplitude)', f'{uround(self.photon_energy_fwhm, eV)} eV')
        info.add_field('Photon Energy FWHM (Power)', f'{uround(self.photon_energy_fwhm_power, eV)} eV')
        info.add_field('Carrier Frequency', f'{uround(self.frequency_carrier, THz)} THz')
        info.add_field('Frequency FWHM (Amplitude)', f'{uround(self.frequency_fwhm, THz)} THz')
        info.add_field('Frequency FWHM (Power)', f'{uround(self.frequency_fwhm_power, THz)} THz')

        info.add_info(self.window.info())

        return info


class SechPulse(UniformLinearlyPolarizedElectricPotential):
    def __init__(self, pulse_width = 200 * asec, omega_carrier = twopi * 3500 * THz, fluence = 1 * J / (cm ** 2), phase = 0, pulse_center = 0 * asec,
                 **kwargs):
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
        super().__init__(**kwargs)

        self.omega_carrier = omega_carrier
        self.pulse_width = pulse_width
        self.phase = phase
        self.fluence = fluence
        self.pulse_center = pulse_center

        self.amplitude_time = np.sqrt(self.fluence / (epsilon_0 * c * self.pulse_width))
        self.amplitude_omega = self.amplitude_time * self.pulse_width * np.sqrt(pi / 2)

    @property
    def photon_energy_carrier(self):
        return hbar * self.omega_carrier

    @property
    def frequency_carrier(self):
        return self.omega_carrier / twopi

    @property
    def time_fwhm(self):
        return 2 * np.log(2 + np.sqrt(3)) * self.pulse_width

    @property
    def omega_fwhm(self):
        return 2 * np.log(2 + np.sqrt(3)) / (2 * self.pulse_width / pi)

    @property
    def photon_energy_fwhm(self):
        return hbar * self.omega_fwhm

    @property
    def frequency_fwhm(self):
        return self.omega_fwhm / twopi

    def keldysh_parameter(self, ionization_potential, test_mass = electron_mass, test_charge = electron_charge):
        if self.phase != 0:
            logger.warning(f'Evaluated Keldysh parameter for {self}, but phase is not zero.')
        amp = self.get_electric_field_amplitude(0)
        return self.omega_carrier * np.sqrt(2 * test_mass * np.abs(ionization_potential)) / (-test_charge * amp)

    def __str__(self):
        out = si.utils.field_str(self,
                                 ('pulse_width', 'asec'),
                                 ('pulse_center', 'asec'),
                                 ('fluence', 'J/cm^2'),
                                 'phase',
                                 ('frequency_carrier', 'THz'),
                                 ('photon_energy_carrier', 'eV'),
                                 )

        return out + super().__str__()

    def __repr__(self):
        return si.utils.field_str(self,
                                  'pulse_width',
                                  'pulse_center',
                                  'fluence',
                                  'phase',
                                  'omega_carrier',
                                  'photon_energy_carrier'
                                  )

    def get_electric_field_envelope(self, t):
        tau = t - self.pulse_center
        return 1 / np.cosh(tau / self.pulse_width)

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        tau = t - self.pulse_center
        amp = self.get_electric_field_envelope(t) * np.cos((self.omega_carrier * tau) + self.phase)

        return amp * self.amplitude_time * super().get_electric_field_amplitude(t)

    def info(self):
        info = super().info()

        info.add_field('Pulse Width', f'{uround(self.pulse_width, asec)} as | {uround(self.pulse_width, fsec, 3)} fs')
        info.add_field('Fluence', f'{uround(self.fluence, Jcm2)} J/cm^2')
        info.add_field('Carrier Photon Energy', f'{uround(self.photon_energy_carrier, eV)} eV')
        info.add_field('Photon Energy FWHM', f'{uround(self.photon_energy_fwhm, eV)} eV')
        info.add_field('Carrier Frequency', f'{uround(self.frequency_carrier, THz)} THz')
        info.add_field('Frequency FWHM', f'{uround(self.frequency_fwhm, THz)} THz')

        info.add_info(self.window.info())

        return info


class GenericElectricPotential(UniformLinearlyPolarizedElectricPotential):
    """Generate an electric field from a Fourier transform of a frequency-amplitude spectrum."""

    def __init__(self, amplitude_function, phase_function = lambda f: 0,
                 frequency_upper_limit = 10000 * THz, frequency_points = 2 ** 18,
                 # fluence = None,
                 name = 'GenericElectricField',
                 extra_information = None,
                 **kwargs):
        """

        Parameters
        ----------
        amplitude_function
        phase_function
        frequency_upper_limit
        frequency_points
        fluence
        name
        extra_information
        kwargs
        """
        super().__init__(**kwargs)

        self.name = name

        self.frequency = np.linspace(-frequency_upper_limit, frequency_upper_limit, frequency_points)
        self.df = np.abs(self.frequency[1] - self.frequency[0])
        amplitude_vs_frequency = amplitude_function(self.frequency)
        phase_vs_frequency = phase_function(self.frequency)
        self.complex_amplitude_vs_frequency = amplitude_vs_frequency * np.exp(1j * phase_vs_frequency) * np.ones(len(self.frequency))

        self.times = nfft.fftshift(nfft.fftfreq(len(self.frequency), self.df))
        self.dt = np.abs(self.times[1] - self.times[0])

        self.complex_electric_field_vs_time = self.df * len(self.frequency) * nfft.fftshift(nfft.ifft(nfft.ifftshift(self.complex_amplitude_vs_frequency)))

        self.extra_attributes = tuple(extra_information.keys())
        for k, v in extra_information.items():
            if k not in self.__dict__:
                setattr(self, k, v)
            else:
                logger.warning('Key collision in extra_arguments of {}: an attribute named {} was already defined'.format(self.name, k))

    @property
    def angular_frequency(self):
        return twopi * self.frequency

    @property
    def dw(self):
        return twopi * self.df

    @property
    def power_vs_frequency(self):
        return np.abs(self.complex_amplitude_vs_frequency) ** 2

    @property
    def fluence(self):
        from_field = epsilon_0 * c * integ.simps(y = np.real(self.complex_electric_field_vs_time) ** 2,
                                                 dx = self.dt)
        # from_spectrum = epsilon_0 * c * integ.simps(y = self.complex_amplitude_vs_frequency ** 2,
        #                                             dx = self.df)

        # return (from_field + from_spectrum) / 2
        return from_field

    def __str__(self):
        return si.utils.field_str(self, 'name', 'fluence', *self.extra_attributes)

    def __repr__(self):
        return si.utils.field_str(self, 'name', 'fluence', *self.extra_attributes)

    def get_electric_field_amplitude(self, t):
        try:
            index, value, target = si.utils.find_nearest_entry(self.times, t)
            amp = self.complex_electric_field_vs_time[index]
        except ValueError:  # t is actually an ndarray
            amp = np.zeros(len(t), dtype = np.complex128) * np.NaN
            for ii, time in enumerate(t):
                index, value, target = si.utils.find_nearest_entry(self.times, time)
                # print(ii, index, value / asec, target / asec, time / asec, self.complex_electric_field_vs_time[index])
                amp[ii] = self.complex_electric_field_vs_time[index]

        return np.real(amp) * super().get_electric_field_amplitude(t)


# class RandomizedSincPulse(UniformLinearlyPolarizedElectricField):
#     def __init__(self, pulse_width = 100 * asec, fluence = 1 * J / (cm ** 2), divisions = 100, **kwargs):
#         super(RandomizedSincPulse, self).__init__(**kwargs)
#
#         self.pulse_width = pulse_width
#
#         self.fluence = fluence
#         self.divisions = divisions
#         self.phases = twopi * np.random.rand(divisions)
#
#         self.omega_cutoff = twopi / self.pulse_width
#         self.amplitude_density = np.sqrt(self.fluence / (2 * epsilon_0 * c * self.omega_cutoff))
#         self.amplitude_prefactor = np.sqrt(2 / pi) * self.amplitude_density
#
#     @property
#     def largest_photon_energy(self):
#         return hbar * self.omega_cutoff
#
#     def __str__(self):
#         out = '{}(pulse width = {} as, pulse center = {} as, fluence = {} J/cm^2, phase = {}, largest photon energy = {} eV)'.format(self.__class__.__name__,
#                                                                                                                                      uround(self.pulse_width, asec),
#                                                                                                                                      uround(self.pulse_center, asec),
#                                                                                                                                      uround(self.fluence, J / (cm ** 2), 3),
#                                                                                                                                      self.phase,
#                                                                                                                                      uround(self.largest_photon_energy, eV))
#
#         return out + super(RandomizedSincPulse, self).__str__()
#
#     def __repr__(self):
#         out = '{}(pulse width = {}, pulse center = {}, fluence = {}, phase = {}, window = {})'.format(self.__class__.__name__,
#                                                                                                       self.pulse_width,
#                                                                                                       self.pulse_center,
#                                                                                                       self.fluence,
#                                                                                                       self.phase,
#                                                                                                       repr(self.window))
#
#         return out
#
#     def get_electric_field_amplitude(self, t):
#         raise NotImplementedError
#
#         # return amp * self.amplitude_prefactor * super(RandomizedSincPulse, self).get_amplitude(t)


class RectangularTimeWindow(TimeWindow):
    def __init__(self, on_time = 0 * asec, off_time = 50 * asec):
        self.on_time = on_time
        self.off_time = off_time

        super().__init__()

    def __str__(self):
        return '{}(on at = {} as, off at = {} as)'.format(self.__class__.__name__,
                                                          uround(self.on_time, asec),
                                                          uround(self.off_time, asec))

    def __repr__(self):
        return '{}(on_time = {}, off_time = {})'.format(self.__class__.__name__,
                                                        self.on_time,
                                                        self.off_time)

    def __call__(self, t):
        cond = np.greater_equal(t, self.on_time) * np.less_equal(t, self.off_time)
        on = 1
        off = 0

        return np.where(cond, on, off)

    def info(self):
        info = super().info()

        info.add_field('Time On', f'{uround(self.on_time, asec)} asec')
        info.add_field('Time Off', f'{uround(self.off_time, asec)} asec')

        return info


class LinearRampTimeWindow(TimeWindow):
    def __init__(self, ramp_on_time = 0 * asec, ramp_time = 50 * asec):
        self.ramp_on_time = ramp_on_time
        self.ramp_time = ramp_time

        # TODO: ramp_from and ramp_to

        super().__init__()

    def __str__(self):
        return '{}(ramp on at = {} as, ramp time = {} as)'.format(self.__class__.__name__,
                                                                  uround(self.ramp_on_time, asec),
                                                                  uround(self.ramp_time, asec))

    def __repr__(self):
        return '{}(ramp_on_time = {}, ramp_time = {})'.format(self.__class__.__name__,
                                                              self.ramp_on_time,
                                                              self.ramp_time)

    def __call__(self, t):
        cond = np.greater_equal(t, self.ramp_on_time)
        on = 1
        off = 0

        out_1 = np.where(cond, on, off)

        cond = np.less_equal(t, self.ramp_on_time + self.ramp_time)
        on = np.ones(np.shape(t)) * (t - self.ramp_on_time) / self.ramp_time
        off = 1

        out_2 = np.where(cond, on, off)

        return out_1 * out_2

    def info(self):
        info = super().info()

        info.add_field('Time Ramp Start', f'{uround(self.ramp_on_time, asec)} asec')
        info.add_field('Time Ramp Finish', f'{uround(self.ramp_off_time, asec)} asec')

        return info


class SymmetricExponentialTimeWindow(TimeWindow):
    def __init__(self, window_time = 500 * asec, window_width = 10 * asec, window_center = 0 * asec):
        self.window_time = window_time
        self.window_width = window_width
        self.window_center = window_center

        super().__init__()

    def __str__(self):
        return '{}(window time = {} as, window width = {} as, window center = {})'.format(self.__class__.__name__,
                                                                                          uround(self.window_time, asec),
                                                                                          uround(self.window_width, asec),
                                                                                          uround(self.window_center, asec))

    def __repr__(self):
        return '{}(window_time = {}, window_width = {}, window_center = {})'.format(self.__class__.__name__,
                                                                                    self.window_time,
                                                                                    self.window_width,
                                                                                    self.window_center)

    def __call__(self, t):
        tau = np.array(t) - self.window_center
        return np.abs(1 / (1 + np.exp(-(tau + self.window_time) / self.window_width)) - 1 / (1 + np.exp(-(tau - self.window_time) / self.window_width)))

    def info(self):
        info = super().info()

        info.add_field('Window Time', f'{uround(self.window_time, asec)} asec')
        info.add_field('Window Width', f'{uround(self.window_width, asec)} asec')
        info.add_field('Window Center', f'{uround(self.window_center, asec)} asec')

        return info


class SmoothedTrapezoidalWindow(TimeWindow):
    def __init__(self, *, time_front, time_plateau):
        super().__init__()

        self.time_front = time_front
        self.time_plateau = time_plateau

    def __call__(self, t):
        cond_before = np.less(t, self.time_front)
        cond_middle = np.less_equal(self.time_front, t) * np.less_equal(t, self.time_front + self.time_plateau)
        cond_after = np.less(self.time_front + self.time_plateau, t) * np.less_equal(t, (2 * self.time_front) + self.time_plateau)

        out = np.where(cond_before, np.sin(pi * t / (2 * self.time_front)) ** 2, 0)
        out += np.where(cond_middle, 1, 0)
        out += np.where(cond_after, np.cos(pi * (t - (self.time_front + self.time_plateau)) / (2 * self.time_front)) ** 2, 0)

        return out

    def info(self):
        info = super().info()

        info.add_field('Front Time', f'{uround(self.time_front, asec)} asec | {uround(self.time_front, fsec)} fsec')
        info.add_field('Plateau Time', f'{uround(self.time_plateau, asec)} asec | {uround(self.time_plateau, fsec)} fsec')

        return info


class RadialCosineMask(Mask):
    """A class representing a masks which begins at some radius and smoothly decreases to 0 as the nth-root of cosine."""

    def __init__(self, inner_radius = 50 * bohr_radius, outer_radius = 100 * bohr_radius, smoothness = 8):
        """Construct a RadialCosineMask from an inner radius, outer radius, and cosine 'smoothness' (the cosine will be raised to the 1/smoothness power)."""
        super().__init__()

        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.smoothness = smoothness

    def __str__(self):
        return '{}(inner radius = {} Bohr radii, outer radius = {} Bohr radii, smoothness = {})'.format(self.__class__.__name__,
                                                                                                        uround(self.inner_radius, bohr_radius, 3),
                                                                                                        uround(self.outer_radius, bohr_radius, 3),
                                                                                                        self.smoothness)

    def __repr__(self):
        return '{}(inner_radius = {}, outer_radius = {}, smoothness = {})'.format(self.__class__.__name__,
                                                                                  self.inner_radius,
                                                                                  self.outer_radius,
                                                                                  self.smoothness)

    def __call__(self, *, r, **kwargs):
        """
        Return the value(s) of the mask at radial position(s) r.

        Accepts only keyword arguments.

        :param r: the radial position coordinate
        :param kwargs: absorbs keyword arguments.
        :return: the value(s) of the mask at r
        """
        return np.where(np.greater_equal(r, self.inner_radius) * np.less_equal(r, self.outer_radius),
                        np.abs(np.cos(0.5 * pi * (r - self.inner_radius) / np.abs(self.outer_radius - self.inner_radius))) ** (1 / self.smoothness),
                        np.where(np.greater_equal(r, self.outer_radius), 0, 1))

    def info(self):
        info = super().info()

        info.add_field('Inner Radius', f'{uround(self.inner_radius, bohr_radius, 3)} Bohr radii | {uround(self.inner_radius, nm, 3)} nm')
        info.add_field('Outer Radius', f'{uround(self.outer_radius, bohr_radius, 3)} Bohr radii | {uround(self.outer_radius, nm, 3)} nm')
        info.add_field('Smoothness', self.smoothness)

        return info
