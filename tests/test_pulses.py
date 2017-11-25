import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np

import ionization as ion
from simulacra.units import *

PULSE_TYPES = [
    ion.SincPulse,
    ion.GaussianPulse,
    ion.SechPulse,
]


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
