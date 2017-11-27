import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np

import ionization as ion
import ionization.integrodiff as ide
from simulacra.units import *

PULSE_TYPES = [
    ion.SincPulse,
    ion.GaussianPulse,
    ion.SechPulse,
]


@hyp.given(
    omega = st.floats(min_value = 0, allow_nan = False, allow_infinity = False)
)
def test_can_construct_sine_wave_with_positive_omega(omega):
    hyp.assume(omega > 0)

    ion.SineWave(omega = omega)


@hyp.given(
    omega = st.floats(max_value = 0, allow_nan = False, allow_infinity = False)
)
def test_cannot_construct_sine_wave_with_non_positive_omega(omega):
    with pytest.raises(ion.InvalidPotentialParameter):
        ion.SineWave(omega = omega)


@hyp.given(
    start_time = st.floats(allow_nan = False, allow_infinity = False),
    data = st.data(),
)
def test_can_construct_rectangle_with_start_time_earlier_than_end_time(start_time, data):
    end_time = data.draw(st.floats(min_value = start_time))
    hyp.assume(end_time > start_time)

    ion.Rectangle(start_time = start_time, end_time = end_time)


@hyp.given(
    start_time = st.floats(allow_nan = False, allow_infinity = False),
    data = st.data(),
)
def test_cannot_construct_rectangle_with_start_time_later_than_end_time(start_time, data):
    end_time = data.draw(st.floats(max_value = start_time))

    with pytest.raises(ion.InvalidPotentialParameter):
        ion.Rectangle(start_time = start_time, end_time = end_time)


@pytest.mark.filterwarnings('ignore: overflow')
@pytest.mark.parametrize(
    'pulse_type',
    PULSE_TYPES
)
@hyp.given(
    pulse_width = st.floats(min_value = 0, allow_infinity = False, allow_nan = False),
)
def test_can_construct_pulse_with_positive_pulse_width(pulse_type, pulse_width):
    hyp.assume(pulse_width > 0)

    pulse = pulse_type(pulse_width = pulse_width)


@pytest.mark.parametrize(
    'pulse_type',
    PULSE_TYPES
)
@hyp.given(
    pulse_width = st.floats(max_value = 0, allow_infinity = False, allow_nan = False),
)
def test_cannot_construct_pulse_with_non_positive_pulse_width(pulse_type, pulse_width):
    with pytest.raises(ion.InvalidPotentialParameter):
        pulse = pulse_type(pulse_width = pulse_width)


@pytest.mark.filterwarnings('ignore: overflow')
@pytest.mark.parametrize(
    'pulse_type',
    PULSE_TYPES
)
@hyp.given(
    fluence = st.floats(min_value = 0, allow_infinity = False, allow_nan = False),
)
def test_can_construct_pulse_with_non_negative_fluence(pulse_type, fluence):
    pulse = pulse_type(fluence = fluence)


@pytest.mark.parametrize(
    'pulse_type',
    PULSE_TYPES
)
@hyp.given(
    fluence = st.floats(max_value = 0, allow_infinity = False, allow_nan = False),
)
def test_cannot_construct_pulse_with_negative_fluence(pulse_type, fluence):
    hyp.assume(fluence != 0)

    with pytest.raises(ion.InvalidPotentialParameter):
        pulse = pulse_type(fluence = fluence)


@hyp.given(
    omega_min = st.floats(min_value = 0, allow_infinity = False, allow_nan = False),
)
def test_can_construct_sinc_pulse_with_non_negative_omega_min(omega_min):
    pulse = ion.SincPulse(omega_min = omega_min)


@hyp.given(
    omega_min = st.floats(max_value = 0, allow_infinity = False, allow_nan = False),
)
def test_cannot_construct_sinc_pulse_with_negative_omega_min(omega_min):
    hyp.assume(omega_min < 0)

    with pytest.raises(ion.InvalidPotentialParameter):
        pulse = ion.SincPulse(omega_min = omega_min)


@pytest.mark.parametrize(
    'pulse_type',
    [
        ion.GaussianPulse,
        ion.SechPulse,
    ]
)
@hyp.given(
    omega_carrier = st.floats(min_value = 0, allow_infinity = False, allow_nan = False),
)
def test_can_construct_gaussian_and_sech_pulses_with_non_negative_omega_carrier(pulse_type, omega_carrier):
    pulse = pulse_type(omega_carrier = omega_carrier)


@pytest.mark.parametrize(
    'pulse_type',
    [
        ion.GaussianPulse,
        ion.SechPulse,
    ]
)
@hyp.given(
    omega_carrier = st.floats(max_value = 0, allow_infinity = False, allow_nan = False),
)
def test_cannot_construct_gaussian_and_sech_pulses_with_negative_omega_carrier(pulse_type, omega_carrier):
    hyp.assume(omega_carrier < 0)

    with pytest.raises(ion.InvalidPotentialParameter):
        pulse = pulse_type(omega_carrier = omega_carrier)


@pytest.mark.parametrize(
    'pulse_type',
    PULSE_TYPES
)
@hyp.given(
    pulse_width = st.floats(min_value = 1 * asec, max_value = 10 * fsec),
    pulse_center = st.floats(min_value = -10 * fsec, max_value = 10 * fsec),
    phase = st.floats(min_value = 0, max_value = twopi),
)
def test_envelope_is_one_at_center_of_pulse(pulse_type, pulse_width, pulse_center, phase):
    pulse = pulse_type(
        pulse_width = pulse_width,
        pulse_center = pulse_center,
        phase = phase,
    )

    np.testing.assert_allclose(pulse.get_electric_field_envelope(pulse_center), 1)


@pytest.mark.parametrize(
    'pulse_type',
    PULSE_TYPES
)
@hyp.given(
    pulse_width = st.floats(min_value = 1 * asec, max_value = 10 * fsec),
    pulse_center = st.floats(min_value = -10 * fsec, max_value = 10 * fsec),
    fluence = st.floats(min_value = 1e-6 * Jcm2, max_value = 100 * Jcm2),
)
def test_electric_field_is_amplitude_at_pulse_center_for_cosine_like_pulse(pulse_type, pulse_width, pulse_center, fluence):
    pulse = pulse_type(
        pulse_width = pulse_width,
        pulse_center = pulse_center,
        fluence = fluence,
        phase = 0,
    )

    np.testing.assert_allclose(pulse.get_electric_field_amplitude(pulse_center), pulse.amplitude)


@pytest.mark.parametrize(
    'pulse_type',
    PULSE_TYPES
)
@hyp.given(
    pulse_width = st.floats(min_value = 1 * asec, max_value = 10 * fsec),
    pulse_center = st.floats(min_value = -10 * fsec, max_value = 10 * fsec),
    fluence = st.floats(min_value = 1e-6 * Jcm2, max_value = 100 * Jcm2),
)
def test_electric_field_is_zero_at_pulse_center_for_sine_like_pulse(pulse_type, pulse_width, pulse_center, fluence):
    pulse = pulse_type(
        pulse_width = pulse_width,
        pulse_center = pulse_center,
        fluence = fluence,
        phase = pi / 2,
    )

    np.testing.assert_allclose(pulse.get_electric_field_amplitude(pulse_center), 0, atol = 1e-9 * pulse.amplitude)  # smaller than part per billion of the pulse amplitude


@pytest.mark.parametrize(
    'spec_type',
    [
        ion.LineSpecification,
        ion.CylindricalSliceSpecification,
        ion.SphericalSliceSpecification,
        ion.SphericalHarmonicSpecification,
        ide.IntegroDifferentialEquationSpecification,
    ]
)
def test_dc_correct_electric_potential_replaces_given_potential_after_simulation_initialization(spec_type):
    pot = ion.SincPulse()

    spec = spec_type(
        'test',
        electric_potential = pot,
        electric_potential_dc_correction = True,
    )

    assert spec.electric_potential is pot

    sim = spec.to_simulation()

    assert sim.spec.electric_potential is not pot
    assert sim.spec.electric_potential[0] is pot
    assert isinstance(sim.spec.electric_potential[1], ion.Rectangle)
