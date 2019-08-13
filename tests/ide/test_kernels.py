import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np
import scipy.interpolate as interp

import simulacra.units as u

import ionization as ion


@pytest.fixture(scope="module")
def electric_field():
    return ion.potentials.GaussianPulse.from_number_of_cycles(
        pulse_width=100 * u.asec, fluence=2 * u.Jcm2, phase=0, number_of_cycles=3
    )


NUM_TIMES = 1000


@pytest.fixture(scope="module")
def times(electric_field):
    return np.linspace(
        -5 * electric_field.pulse_width, 5 * electric_field.pulse_width, NUM_TIMES
    )


@pytest.fixture(scope="module")
def vector_potential(electric_field, times):
    return interp.CubicSpline(
        y=electric_field.get_vector_potential_amplitude_numeric_cumulative(times),
        x=times,
        bc_type="natural",
    )


@hyp.settings(deadline=None, max_examples=100)
@hyp.given(current_time_index=st.integers(min_value=1, max_value=NUM_TIMES - 1))
@pytest.mark.parametrize(
    "kernel",
    [
        ion.ide.LengthGaugeHydrogenKernel(),
        ion.ide.LengthGaugeHydrogenKernelWithContinuumContinuumInteraction(),
    ],
)
def test_hydrogen_kernel_at_time_difference_zero_is_correct(
    kernel, electric_field, vector_potential, times, current_time_index
):
    current_time = times[current_time_index]

    assert (
        kernel(current_time, np.array([current_time]), electric_field, vector_potential)
        == u.bohr_radius ** 2
    )


@hyp.settings(deadline=None, max_examples=100)
@hyp.given(current_time_index=st.integers(min_value=1, max_value=NUM_TIMES - 1))
def test_approximate_continuum_continuum_interaction_phase_is_zero_at_time_difference_zero(
    vector_potential, times, current_time_index
):
    kernel = ion.ide.LengthGaugeHydrogenKernelWithContinuumContinuumInteraction()

    current_time = times[current_time_index]
    previous_times = times[: current_time_index + 1]

    # the last entry is the one at time difference zero
    time_diff_zero_phase_factor = kernel._vector_potential_phase_factor(
        current_time, previous_times, vector_potential
    )[-1]

    assert time_diff_zero_phase_factor == np.exp(0)  # i.e., 1


@hyp.settings(deadline=None, max_examples=100)
@hyp.given(current_time_index=st.integers(min_value=1, max_value=NUM_TIMES - 1))
def test_integral_of_vector_potential_squared_in_phase_factor_is_never_negative(
    vector_potential, times, current_time_index
):
    kernel = ion.ide.LengthGaugeHydrogenKernelWithContinuumContinuumInteraction()

    previous_times = times[: current_time_index + 1]

    assert all(
        kernel._integrate_vector_potential_over_previous_times(
            previous_times, vector_potential, power=2
        )
        >= 0
    )
