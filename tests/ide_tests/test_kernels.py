import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np
import scipy.integrate as integ
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


NUM_TIMES = 1000


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
    max_examples = 100,
)
@hyp.given(current_time_index = st.integers(min_value = 1, max_value = NUM_TIMES - 1))
@pytest.mark.parametrize(
    'kernel',
    [
        ide.LengthGaugeHydrogenKernel(),
        ide.ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction(),
    ]
)
def test_hydrogen_kernel_at_time_difference_zero_is_correct(kernel, electric_field, vector_potential, times, current_time_index):
    current_time = times[current_time_index]

    assert kernel(current_time, np.array([current_time]), electric_field, vector_potential) == u.bohr_radius ** 2


@hyp.settings(
    deadline = None,
    max_examples = 100,
)
@hyp.given(current_time_index = st.integers(min_value = 1, max_value = NUM_TIMES - 1))
def test_approximate_continuum_continuum_interaction_phase_is_zero_at_time_difference_zero(vector_potential, times, current_time_index):
    kernel = ide.ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction()

    current_time = times[current_time_index]
    previous_times = times[:current_time_index + 1]

    # the last entry is the one at time difference zero
    time_diff_zero_phase_factor = kernel._vector_potential_phase_factor(current_time, previous_times, vector_potential)[-1]

    assert time_diff_zero_phase_factor == np.exp(0)  # i.e., 1


@hyp.settings(
    deadline = None,
    max_examples = 100,
)
@hyp.given(current_time_index = st.integers(min_value = 1, max_value = NUM_TIMES - 1))
def test_integral_of_vector_potential_squared_in_phase_factor_is_never_negative(vector_potential, times, current_time_index):
    kernel = ide.ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction()

    previous_times = times[:current_time_index + 1]

    assert all(kernel._integrate_vector_potential_over_previous_times(previous_times, vector_potential, power = 2) >= 0)


@pytest.fixture(scope = 'module')
def p():
    return np.linspace(0, 50, 10000) * u.atomic_momentum


@pytest.fixture(scope = 'module')
def full_hydrogen_kernel():
    return ide.FullHydrogenKernel()


@pytest.fixture(scope = 'module')
def pulse():
    return ion.potentials.GaussianPulse.from_number_of_cycles(
        pulse_width = 200 * u.asec,
        number_of_cycles = 4,
        fluence = 1 * u.Jcm2,
        phase = 0,
    )


@pytest.fixture(scope = 'module')
def times(pulse):
    return np.linspace(-5 * pulse.pulse_width, 5 * pulse.pulse_width, 2000)


@pytest.fixture(scope = 'module')
def vector_potential(pulse, times):
    return ide.interpolate_vector_potential(pulse, times)


class TestFullHydrogenKernel:
    def test_phase_integrand_is_p2_if_previous_time_is_current_time(self, full_hydrogen_kernel, p, vector_potential, times):
        phase_integrand = full_hydrogen_kernel.phase_integrand(p, u.pi / 3, times[1000], times[1000], vector_potential)

        np.testing.assert_allclose(phase_integrand, p ** 2)

    def test_phase_integral_is_zero_if_previous_time_is_current_time(self, full_hydrogen_kernel, vector_potential, times):
        phase_integral = full_hydrogen_kernel.phase_integral(1 * u.atomic_momentum, u.pi / 3, times[1000], times[1000], vector_potential)

        np.testing.assert_allclose(phase_integral, 0)

    def test_phase_integral_is_one_if_previous_time_is_current_time(self, full_hydrogen_kernel, vector_potential, times):
        phase_factor = full_hydrogen_kernel.phase_factor(1 * u.atomic_momentum, u.pi / 3, times[1000], times[1000], vector_potential)

        np.testing.assert_allclose(phase_factor, 1)

    def test_z_dipole_matrix_element_per_momentum_is_zero_if_p_is_zero(self, full_hydrogen_kernel):
        matrix_element = full_hydrogen_kernel.z_dipole_matrix_element_per_momentum(0, u.pi / 3)

        assert matrix_element == 0

    def test_matrix_element_prefactor_magnitude_squared_is_right(self, full_hydrogen_kernel):
        prefactor2 = np.abs(full_hydrogen_kernel.matrix_element_prefactor) ** 2
        expected = 512 * 3 * (u.bohr_radius ** 7) / (12 * (u.pi ** 2) * (u.hbar ** 5))

        np.testing.assert_allclose(prefactor2, expected)

    def test_z_dipole_matrix_element_per_momentum_is_pure_imaginary(self, full_hydrogen_kernel, p):
        matrix_element = full_hydrogen_kernel.z_dipole_matrix_element_per_momentum(p, u.pi / 3)

        np.testing.assert_allclose(1j * np.imag(matrix_element), matrix_element)

    def test_integral_of_z_dipole_matrix_element_for_zero_time_difference_is_bohr_radius_squared(self, full_hydrogen_kernel, p):
        integrand = (p ** 2) * np.abs(full_hydrogen_kernel.z_dipole_matrix_element_per_momentum(p, 0)) ** 2
        integral = integ.simps(
            y = integrand,
            x = p
        )
        integral *= 4 * u.pi / 3  # angular part by hand

        np.testing.assert_allclose(integral, u.bohr_radius ** 2)

    @pytest.mark.slow
    def test_kernel_integrand_at_time_difference_zero(self, full_hydrogen_kernel, times, vector_potential):
        manual = (4 * u.pi / 3) * (full_hydrogen_kernel.p ** 2) * np.abs(full_hydrogen_kernel.z_dipole_matrix_element_per_momentum(full_hydrogen_kernel.p, 0)) ** 2
        method = full_hydrogen_kernel.integrand_vs_p(times[1000], times[1000], vector_potential)

        np.testing.assert_allclose(manual, method)
