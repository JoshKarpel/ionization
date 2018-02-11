import itertools

import pytest

import numpy as np

import ionization as ion
import simulacra.units as u

from tests import testutils

SPEC_TYPES = [
    ion.mesh.LineSpecification,
    ion.mesh.CylindricalSliceSpecification,
    ion.mesh.SphericalSliceSpecification,
    ion.mesh.SphericalHarmonicSpecification,
]

THREE_DIMENSIONAL_SPEC_TYPES = [
    ion.mesh.CylindricalSliceSpecification,
    ion.mesh.SphericalSliceSpecification,
    ion.mesh.SphericalHarmonicSpecification,
]

SPEC_TYPES_WITH_NUMERIC_EIGENSTATES = [
    ion.mesh.LineSpecification,
    ion.mesh.SphericalHarmonicSpecification,
]

LOW_N_HYDROGEN_BOUND_STATES = [ion.mesh.states.HydrogenBoundState(n, l) for n in range(6) for l in range(n)]

EVOLUTION_GAUGES = ['LEN', 'VEL']


class TestMeshSavingAndLoading:
    @pytest.mark.parametrize(
        'spec_type',
        SPEC_TYPES
    )
    def test_loaded_sim_mesh_is_same_as_original(self, spec_type, tmpdir):
        sim_type = spec_type.simulation_type
        sim = spec_type('test').to_simulation()

        special_g = testutils.complex_random_sample(sim.mesh.g.shape)
        sim.mesh.g = special_g

        path = sim.save(tmpdir, save_mesh = True)
        loaded_sim = sim_type.load(path)

        np.testing.assert_equal(loaded_sim.mesh.g, sim.mesh.g)
        np.testing.assert_equal(loaded_sim.mesh.g, special_g)

    @pytest.mark.parametrize(
        'spec_type',
        SPEC_TYPES
    )
    def test_loaded_sim_mesh_is_none_if_not_saved(self, spec_type, tmpdir):
        sim_type = spec_type.simulation_type
        sim = spec_type('test').to_simulation()

        path = sim.save(tmpdir, save_mesh = False)
        loaded_sim = sim_type.load(path)

        assert loaded_sim.mesh is None


@pytest.mark.parametrize(
    'initial_state',
    LOW_N_HYDROGEN_BOUND_STATES
)
def test_initial_wavefunction_is_normalized_for_spherical_harmonic_mesh_with_numeric_eigenstates(initial_state):
    sim = ion.mesh.SphericalHarmonicSpecification(
        'test',
        initial_state = initial_state,
        evolution_gauge = 'LEN',
        evolution_method = 'CN',
        time_initial = 0,
        time_final = 100 * u.asec,
        time_step = 1 * u.asec,
        r_bound = 100 * u.bohr_radius,
        l_bound = 30,
        use_numeric_eigenstates = True,
        numeric_eigenstate_max_energy = 10 * u.eV,
        numeric_eigenstate_max_angular_momentum = 10,
    ).to_simulation()

    np.testing.assert_allclose(sim.mesh.norm(), 1)


@pytest.mark.parametrize(
    'initial_state',
    LOW_N_HYDROGEN_BOUND_STATES
)
def test_with_no_potential_final_state_is_initial_state_for_spherical_harmonic_mesh_with_numeric_eigenstates_and_crank_nicholson_evolution(initial_state):
    sim = ion.mesh.SphericalHarmonicSpecification(
        'test',
        initial_state = initial_state,
        evolution_gauge = 'LEN',
        evolution_method = 'CN',
        time_initial = 0,
        time_final = 100 * u.asec,
        time_step = 1 * u.asec,
        r_bound = 100 * u.bohr_radius,
        r_points = 500,
        l_bound = 30,
        use_numeric_eigenstates = True,
        numeric_eigenstate_max_energy = 10 * u.eV,
        numeric_eigenstate_max_angular_momentum = 10,
    ).to_simulation()

    sim.run_simulation()

    initial_norm = sim.norm_vs_time[0]
    final_norm = sim.norm_vs_time[-1]
    initial_state_overlaps = np.array([overlap_vs_time[0] for overlap_vs_time in sim.state_overlaps_vs_time.values()])
    final_state_overlaps = np.array([overlap_vs_time[-1] for overlap_vs_time in sim.state_overlaps_vs_time.values()])

    np.testing.assert_allclose(initial_norm, final_norm, atol = 1e-14)
    np.testing.assert_allclose(initial_state_overlaps, final_state_overlaps, atol = 1e-14)


@pytest.mark.parametrize(
    'initial_state, evolution_gauge',
    itertools.product(
        LOW_N_HYDROGEN_BOUND_STATES,
        EVOLUTION_GAUGES,
    )
)
def test_with_no_potential_final_state_is_initial_state_for_spherical_harmonic_mesh_with_numeric_eigenstates_and_split_operator_evolution(initial_state, evolution_gauge):
    sim = ion.mesh.SphericalHarmonicSpecification(
        'test',
        initial_state = initial_state,
        evolution_gauge = evolution_gauge,
        evolution_method = 'SO',
        time_initial = 0,
        time_final = 100 * u.asec,
        time_step = 1 * u.asec,
        r_bound = 100 * u.bohr_radius,
        r_points = 500,
        l_bound = 30,
        use_numeric_eigenstates = True,
        numeric_eigenstate_max_energy = 10 * u.eV,
        numeric_eigenstate_max_angular_momentum = 10,
    ).to_simulation()

    sim.run_simulation()

    initial_norm = sim.norm_vs_time[0]
    final_norm = sim.norm_vs_time[-1]
    initial_state_overlaps = np.array([overlap_vs_time[0] for overlap_vs_time in sim.state_overlaps_vs_time.values()])
    final_state_overlaps = np.array([overlap_vs_time[-1] for overlap_vs_time in sim.state_overlaps_vs_time.values()])

    np.testing.assert_allclose(initial_norm, final_norm, atol = 1e-14)
    np.testing.assert_allclose(initial_state_overlaps, final_state_overlaps, atol = 1e-14)
