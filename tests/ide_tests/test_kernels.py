import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np
import scipy.interpolate as interp

import ionization as ion
import ionization.ide as ide
import simulacra.units as u


@pytest.fixture(scope = 'module')
def electric_field():
    return ion.potentials.GaussianPulse.from_number_of_cycles(
        pulse_width = 100 * u.asec,
        fluence = 2 * u.Jcm2,
        phase = 0,
        number_of_cycles = 3,
    )


NUM_TIMES = 10000


@pytest.fixture(scope = 'module')
def times(electric_field):
    return np.linspace(-5 * electric_field.pulse_width, 5 * electric_field.pulse_width, NUM_TIMES)


@pytest.fixture(scope = 'module')
def vector_potential(electric_field, times):
    return interp.CubicSpline(
        y = electric_field.get_vector_potential_amplitude_numeric_cumulative(times),
        x = times,
        bc_type = 'natural',
    )


@hyp.settings(
    deadline = None,
)
@hyp.given(current_time_index = st.integers(min_value = 1, max_value = NUM_TIMES - 1))
@pytest.mark.parametrize(
    'kernel',
    [
        ide.LengthGaugeHydrogenKernel(),
        ide.ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction(),
    ]
)
def test_hydrogen_kernel_at_time_difference_zero(kernel, electric_field, vector_potential, times, current_time_index):
    current_time = times[current_time_index]

    assert kernel(current_time, np.array([current_time]), electric_field, vector_potential) == u.bohr_radius ** 2
