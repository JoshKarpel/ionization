import pytest

import numpy as np

import simulacra.units as u

import ionization as ion

from .conftest import LOW_N_HYDROGEN_BOUND_STATES


@pytest.mark.parametrize(
    'initial_state',
    LOW_N_HYDROGEN_BOUND_STATES
)
def test_initial_wavefunction_is_normalized_for_spherical_harmonic_mesh_with_numeric_eigenstates(initial_state):
    sim = ion.mesh.SphericalHarmonicSpecification(
        'test',
        initial_state = initial_state,
        r_bound = 100 * u.bohr_radius,
        l_bound = 30,
        use_numeric_eigenstates = True,
        numeric_eigenstate_max_energy = 20 * u.eV,
        numeric_eigenstate_max_angular_momentum = 10,
    ).to_sim()

    np.testing.assert_allclose(sim.mesh.norm(), 1)


@pytest.mark.parametrize(
    'initial_state',
    LOW_N_HYDROGEN_BOUND_STATES
)
def test_with_no_potential_final_state_is_initial_state_for_spherical_harmonic_mesh_with_numeric_eigenstates_and_crank_nicholson_evolution(initial_state):
    sim = ion.mesh.SphericalHarmonicSpecification(
        'test',
        initial_state = initial_state,
        operators = ion.mesh.SphericalHarmonicLengthGaugeOperators(),
        evolution_method = ion.mesh.AlternatingDirectionImplicitCrankNicolson(),
        time_initial = 0,
        time_final = 100 * u.asec,
        time_step = 1 * u.asec,
        r_bound = 100 * u.bohr_radius,
        r_points = 500,
        l_bound = 30,
        use_numeric_eigenstates = True,
        numeric_eigenstate_max_energy = 20 * u.eV,
        numeric_eigenstate_max_angular_momentum = 10,
    ).to_sim()

    sim.run()

    initial_norm = sim.data.norm[0]
    final_norm = sim.data.norm[-1]
    initial_state_overlaps = np.array([overlap_vs_time[0] for overlap_vs_time in sim.data.state_overlaps.values()])
    final_state_overlaps = np.array([overlap_vs_time[-1] for overlap_vs_time in sim.data.state_overlaps.values()])

    np.testing.assert_allclose(initial_norm, final_norm, atol = 1e-14)
    np.testing.assert_allclose(initial_state_overlaps, final_state_overlaps, atol = 1e-14)


@pytest.mark.parametrize(
    'initial_state', LOW_N_HYDROGEN_BOUND_STATES,
)
@pytest.mark.parametrize(
    'operators', [ion.mesh.SphericalHarmonicLengthGaugeOperators(), ion.mesh.SphericalHarmonicVelocityGaugeOperators()],
)
def test_with_no_potential_final_state_is_initial_state_for_spherical_harmonic_mesh_with_numeric_eigenstates_and_split_operator_evolution(initial_state, operators):
    sim = ion.mesh.SphericalHarmonicSpecification(
        'test',
        initial_state = initial_state,
        operators = operators,
        evolution_method = ion.mesh.SphericalHarmonicSplitOperator(),
        time_initial = 0,
        time_final = 100 * u.asec,
        time_step = 1 * u.asec,
        r_bound = 100 * u.bohr_radius,
        r_points = 500,
        l_bound = 30,
        use_numeric_eigenstates = True,
        numeric_eigenstate_max_energy = 10 * u.eV,
        numeric_eigenstate_max_angular_momentum = 10,
    ).to_sim()

    sim.run()

    initial_norm = sim.data.norm[0]
    final_norm = sim.data.norm[-1]
    initial_state_overlaps = np.array([overlap_vs_time[0] for overlap_vs_time in sim.data.state_overlaps.values()])
    final_state_overlaps = np.array([overlap_vs_time[-1] for overlap_vs_time in sim.data.state_overlaps.values()])

    np.testing.assert_allclose(initial_norm, final_norm, atol = 1e-14)
    np.testing.assert_allclose(initial_state_overlaps, final_state_overlaps, atol = 1e-14)
