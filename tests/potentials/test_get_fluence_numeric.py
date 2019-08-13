import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np

import ionization as ion

import simulacra.units as u


@hyp.given(
    amplitude=st.floats(
        min_value=-1 * u.atomic_electric_field, max_value=1 * u.atomic_electric_field
    ),
    duration=st.floats(min_value=10 * u.asec, max_value=1000 * u.asec),
)
def test_fluence_for_rectangle(amplitude, duration):
    rect = ion.potentials.Rectangle(
        start_time=0, end_time=duration, amplitude=amplitude
    )
    times = np.linspace(-0.1 * duration, duration * 1.1, 100000)

    analytic_fluence = u.epsilon_0 * u.c * (amplitude ** 2) * duration

    assert np.allclose(rect.get_fluence_numeric(times), analytic_fluence)


@hyp.settings(deadline=None)
@hyp.given(
    amplitude_1=st.floats(
        min_value=-1 * u.atomic_electric_field, max_value=1 * u.atomic_electric_field
    ),
    amplitude_2=st.floats(
        min_value=-1 * u.atomic_electric_field, max_value=1 * u.atomic_electric_field
    ),
    duration_1=st.floats(min_value=10 * u.asec, max_value=1000 * u.asec),
    delay=st.floats(min_value=20 * u.asec, max_value=500 * u.asec),
    duration_2=st.floats(min_value=10 * u.asec, max_value=1000 * u.asec),
)
def test_fluence_for_disjoint_rectangles(
    amplitude_1, amplitude_2, duration_1, delay, duration_2
):
    pot = ion.potentials.Rectangle(
        start_time=0, end_time=duration_1, amplitude=amplitude_1
    )
    pot += ion.potentials.Rectangle(
        start_time=duration_1 + delay,
        end_time=duration_1 + delay + duration_2,
        amplitude=amplitude_2,
    )

    total_duration = duration_1 + delay + duration_2
    times = np.linspace(-0.1 * total_duration, 1.1 * total_duration, 1000000)

    analytic_fluence = u.epsilon_0 * u.c * (amplitude_1 ** 2) * duration_1
    analytic_fluence += u.epsilon_0 * u.c * (amplitude_2 ** 2) * duration_2

    assert np.allclose(pot.get_fluence_numeric(times), analytic_fluence, rtol=1e-3)


@hyp.settings(deadline=None)
@hyp.given(
    amplitude_1=st.floats(
        min_value=-1 * u.atomic_electric_field, max_value=1 * u.atomic_electric_field
    ),
    amplitude_2=st.floats(
        min_value=-1 * u.atomic_electric_field, max_value=1 * u.atomic_electric_field
    ),
    duration_1=st.floats(min_value=200 * u.asec, max_value=500 * u.asec),
    delay=st.floats(min_value=10 * u.asec, max_value=100 * u.asec),
    duration_2=st.floats(min_value=10 * u.asec, max_value=50 * u.asec),
)
def test_fluence_for_overlapping_rectangles(
    amplitude_1, amplitude_2, duration_1, delay, duration_2
):
    pot = ion.potentials.Rectangle(
        start_time=0, end_time=duration_1, amplitude=amplitude_1
    )
    pot += ion.potentials.Rectangle(
        start_time=delay, end_time=delay + duration_2, amplitude=amplitude_2
    )

    times = np.linspace(-0.1 * duration_1, 1.1 * duration_1, 1000000)

    analytic_fluence = (
        u.epsilon_0 * u.c * (amplitude_1 ** 2) * delay
    )  # first rectangle is on
    analytic_fluence += (
        u.epsilon_0 * u.c * ((amplitude_1 + amplitude_2) ** 2) * duration_2
    )  # both on
    analytic_fluence += (
        u.epsilon_0 * u.c * (amplitude_1 ** 2) * (duration_1 - duration_2 - delay)
    )  # only first on

    assert np.allclose(pot.get_fluence_numeric(times), analytic_fluence, rtol=1e-3)
