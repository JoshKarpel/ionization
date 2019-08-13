import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np

import simulacra.units as u

import ionization as ion

PULSES_TYPES = [
    ion.potentials.SincPulse,
    ion.potentials.GaussianPulse,
    ion.potentials.SechPulse,
]


class TestNoElectricPotential:
    @hyp.given(t=st.floats(allow_nan=False, allow_infinity=False))
    def test_electric_field_amplitude_is_always_zero(self, t):
        assert ion.potentials.NoElectricPotential().get_electric_field_amplitude(t) == 0

    @hyp.given(t=st.floats(allow_nan=False, allow_infinity=False))
    def test_vector_potential_amplitude_is_always_zero(self, t):
        assert (
            ion.potentials.NoElectricPotential().get_vector_potential_amplitude(t) == 0
        )


class TestSineWave:
    def test_can_construct_with_positive_omega(self):
        ion.potentials.SineWave(omega=1 * u.atomic_angular_frequency)

    @pytest.mark.parametrize("omega", [0, -1 * u.atomic_angular_frequency])
    def test_cannot_construct_with_non_positive_omega(self, omega):
        with pytest.raises(ion.exceptions.InvalidPotentialParameter):
            ion.potentials.SineWave(omega=omega)

    def test_from_frequency(self):
        frequency = 5 * u.atomic_frequency

        pot = ion.potentials.SineWave.from_frequency(frequency=frequency)

        assert np.allclose(pot.frequency, frequency)

    def test_from_period(self):
        period = 1 * u.atomic_time

        pot = ion.potentials.SineWave.from_period(period=period)

        assert np.allclose(pot.period, period)

    def test_from_wavelength(self):
        wavelength = 1 * u.bohr_radius

        pot = ion.potentials.SineWave.from_wavelength(wavelength=wavelength)

        assert np.allclose(pot.wavelength, wavelength)

    def test_from_photon_energy(self):
        photon_energy = 1 * u.eV

        pot = ion.potentials.SineWave.from_photon_energy(photon_energy=photon_energy)

        assert np.allclose(pot.photon_energy, photon_energy)

    def test_from_photon_energy_and_intensity(self):
        photon_energy = 1 * u.eV
        intensity = 1 * u.atomic_intensity

        pot = ion.potentials.SineWave.from_photon_energy_and_intensity(
            photon_energy=photon_energy, intensity=intensity
        )

        assert np.allclose(pot.photon_energy, photon_energy)
        assert np.allclose(pot.intensity, intensity)

    @pytest.mark.parametrize("intensity", [1 * u.atomic_intensity, 0])
    def test_can_construct_with_non_negative_intensity(self, intensity):
        ion.potentials.SineWave.from_photon_energy_and_intensity(
            photon_energy=1 * u.eV, intensity=intensity
        )

    def test_cannot_construct_with_negative_intensity(self):
        with pytest.raises(ion.exceptions.InvalidPotentialParameter):
            ion.potentials.SineWave.from_photon_energy_and_intensity(
                photon_energy=1 * u.eV, intensity=-1 * u.atomic_intensity
            )


class TestRectangle:
    def test_can_construct_with_start_time_earlier_than_end_time(self):
        ion.potentials.Rectangle(start_time=0 * u.asec, end_time=1 * u.asec)

    def test_cannot_construct_with_start_time_later_than_end_time(self):
        with pytest.raises(ion.exceptions.InvalidPotentialParameter):
            ion.potentials.Rectangle(start_time=1 * u.asec, end_time=0 * u.asec)


@pytest.mark.filterwarnings("ignore: overflow")
@pytest.mark.parametrize("pulse_type", PULSES_TYPES)
def test_can_construct_pulse_with_positive_pulse_width(pulse_type):
    pulse_type(pulse_width=100 * u.asec)


@pytest.mark.parametrize("pulse_type", PULSES_TYPES)
@pytest.mark.parametrize("pulse_width", [0, -100 * u.asec])
def test_cannot_construct_pulse_with_negative_pulse_width(pulse_type, pulse_width):
    with pytest.raises(ion.exceptions.InvalidPotentialParameter):
        pulse_type(pulse_width=pulse_width)


@pytest.mark.parametrize("pulse_type", PULSES_TYPES)
@pytest.mark.parametrize("fluence", [1 * u.Jcm2, 0])
def test_can_construct_pulses_with_non_negative_fluence(pulse_type, fluence):
    pulse_type(fluence=fluence)


@pytest.mark.parametrize("pulse_type", PULSES_TYPES)
def test_pulse_construction_fluences_can_be_non_negative(pulse_type):
    with pytest.raises(ion.exceptions.InvalidPotentialParameter):
        pulse_type(fluence=-1 * u.Jcm2)


def test_can_construct_sinc_pulse_with_positive_omega_min():
    ion.potentials.SincPulse(omega_min=1 * u.atomic_angular_frequency)


@pytest.mark.parametrize("omega_min", [-1 * u.atomic_angular_frequency, 0])
def test_cannot_construct_sinc_pulse_with_non_positive_omega_min(omega_min):
    with pytest.raises(ion.exceptions.InvalidPotentialParameter):
        ion.potentials.SincPulse(omega_min=omega_min)


def test_can_construct_sinc_pulse_with_good_number_of_cycles():
    pulse = ion.potentials.SincPulse.from_number_of_cycles(number_of_cycles=1)

    assert np.allclose(pulse.number_of_cycles, 1)


@pytest.mark.parametrize("number_of_cycles", [0.5, 0.25, 0, -1])
def test_cannot_construct_sinc_pulse_with_bad_number_of_cycles(number_of_cycles):
    with pytest.raises(ion.exceptions.InvalidPotentialParameter):
        ion.potentials.SincPulse.from_number_of_cycles(
            number_of_cycles=number_of_cycles
        )


@pytest.mark.parametrize(
    "pulse_type", [ion.potentials.GaussianPulse, ion.potentials.SechPulse]
)
@pytest.mark.parametrize("omega_carrier", [1 * u.atomic_angular_frequency, 0])
def test_can_construct_gaussian_and_sech_pulses_with_non_negative_omega_carrier(
    pulse_type, omega_carrier
):
    pulse_type(omega_carrier=omega_carrier)


@pytest.mark.parametrize(
    "pulse_type", [ion.potentials.GaussianPulse, ion.potentials.SechPulse]
)
def test_cannot_construct_gaussian_and_sech_pulses_with_negative_omega_carrier(
    pulse_type
):
    with pytest.raises(ion.exceptions.InvalidPotentialParameter):
        pulse_type(omega_carrier=-1 * u.atomic_angular_frequency)


@pytest.mark.parametrize("pulse_type", PULSES_TYPES)
@hyp.given(
    pulse_width=st.floats(min_value=1 * u.asec, max_value=10 * u.fsec),
    pulse_center=st.floats(min_value=-10 * u.fsec, max_value=10 * u.fsec),
    phase=st.floats(min_value=0, max_value=u.twopi),
)
def test_envelope_is_one_at_center_of_pulse(
    pulse_type, pulse_width, pulse_center, phase
):
    pulse = pulse_type(pulse_width=pulse_width, pulse_center=pulse_center, phase=phase)

    np.testing.assert_allclose(pulse.get_electric_field_envelope(pulse_center), 1)


@pytest.mark.parametrize("pulse_type", PULSES_TYPES)
@hyp.given(
    pulse_width=st.floats(min_value=1 * u.asec, max_value=10 * u.fsec),
    pulse_center=st.floats(min_value=-10 * u.fsec, max_value=10 * u.fsec),
    fluence=st.floats(min_value=1e-6 * u.Jcm2, max_value=100 * u.Jcm2),
)
def test_electric_field_is_amplitude_at_pulse_center_for_cosine_like_pulse(
    pulse_type, pulse_width, pulse_center, fluence
):
    pulse = pulse_type(
        pulse_width=pulse_width, pulse_center=pulse_center, fluence=fluence, phase=0
    )

    np.testing.assert_allclose(
        pulse.get_electric_field_amplitude(pulse_center), pulse.amplitude
    )


@pytest.mark.parametrize("pulse_type", PULSES_TYPES)
@hyp.given(
    pulse_width=st.floats(min_value=1 * u.asec, max_value=10 * u.fsec),
    pulse_center=st.floats(min_value=-10 * u.fsec, max_value=10 * u.fsec),
    fluence=st.floats(min_value=1e-6 * u.Jcm2, max_value=100 * u.Jcm2),
)
def test_electric_field_is_zero_at_pulse_center_for_sine_like_pulse(
    pulse_type, pulse_width, pulse_center, fluence
):
    pulse = pulse_type(
        pulse_width=pulse_width,
        pulse_center=pulse_center,
        fluence=fluence,
        phase=u.pi / 2,
    )

    np.testing.assert_allclose(
        pulse.get_electric_field_amplitude(pulse_center), 0, atol=1e-9 * pulse.amplitude
    )  # smaller than part per billion of the pulse amplitude
