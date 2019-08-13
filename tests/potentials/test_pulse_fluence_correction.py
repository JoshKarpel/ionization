import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np

import simulacra.units as u

import ionization as ion


def test_rectangle_fluence_is_exactly_target_fluence_after_correction():
    target_fluence = 1 * u.Jcm2

    pulse = ion.potentials.Rectangle(
        start_time=0, end_time=100 * u.asec, amplitude=1 * u.atomic_electric_field
    )

    times = np.linspace(-10 * u.asec, 110 * u.asec, 1000)

    assert not np.allclose(pulse.get_fluence_numeric(times), target_fluence)

    corrected_pulse = ion.potentials.FluenceCorrector(
        electric_potential=pulse, times=times, target_fluence=target_fluence
    )

    assert np.allclose(corrected_pulse.get_fluence_numeric(times), target_fluence)


@hyp.settings(deadline=None)
@hyp.given(
    fluence=st.floats(min_value=0.001 * u.Jcm2, max_value=10 * u.Jcm2),
    pulse_width=st.floats(min_value=10 * u.asec, max_value=1000 * u.asec),
    phase=st.floats(min_value=0, max_value=u.twopi),
)
def test_sinc_pulse_fluence_is_exactly_target_fluence_after_correction(
    fluence, pulse_width, phase
):
    pulse = ion.potentials.SincPulse(
        pulse_width=pulse_width,
        fluence=fluence,
        phase=phase,
        window=ion.potentials.LogisticWindow(
            window_time=30 * pulse_width, window_width=0.2 * pulse_width
        ),
    )
    times = np.linspace(-35 * pulse_width, 35 * pulse_width, 10000)

    corrected_pulse = ion.potentials.FluenceCorrector(
        electric_potential=pulse, times=times, target_fluence=fluence
    )

    assert np.allclose(corrected_pulse.get_fluence_numeric(times), fluence)


@hyp.settings(deadline=None)
@hyp.given(
    fluence=st.floats(min_value=0.001 * u.Jcm2, max_value=10 * u.Jcm2),
    pulse_width=st.floats(min_value=10 * u.asec, max_value=1000 * u.asec),
    phase=st.floats(min_value=0, max_value=u.twopi),
)
def test_sinc_pulse_fluence_is_exactly_target_fluence_after_correction_and_dc_correction(
    fluence, pulse_width, phase
):
    pulse = ion.potentials.SincPulse(
        pulse_width=pulse_width,
        fluence=fluence,
        phase=phase,
        window=ion.potentials.LogisticWindow(
            window_time=30 * pulse_width, window_width=0.2 * pulse_width
        ),
    )
    times = np.linspace(-35 * pulse_width, 35 * pulse_width, 10000)

    pulse = ion.potentials.DC_correct_electric_potential(pulse, times)

    corrected_pulse = ion.potentials.FluenceCorrector(
        electric_potential=pulse, times=times, target_fluence=fluence
    )

    assert np.allclose(corrected_pulse.get_fluence_numeric(times), fluence)
