import pytest

import numpy as np

import simulacra.units as u

import ionization as ion

from .conftest import LOW_N_HYDROGEN_BOUND_STATES


class TestSphericalHarmonicSimulation:
    @pytest.mark.parametrize("initial_state", LOW_N_HYDROGEN_BOUND_STATES)
    def test_initial_wavefunction_is_normalized_with_numeric_eigenstates(
        self, initial_state
    ):
        sim = ion.mesh.SphericalHarmonicSpecification(
            "test",
            initial_state=initial_state,
            r_bound=50 * u.bohr_radius,
            r_points=250,
            l_bound=30,
            use_numeric_eigenstates=True,
            numeric_eigenstate_max_energy=20 * u.eV,
            numeric_eigenstate_max_angular_momentum=10,
        ).to_sim()

        np.testing.assert_allclose(sim.mesh.norm(), 1)

    @pytest.mark.parametrize("initial_state", LOW_N_HYDROGEN_BOUND_STATES)
    @pytest.mark.parametrize(
        "operators, evolution_method",
        [
            (
                ion.mesh.SphericalHarmonicLengthGaugeOperators(),
                ion.mesh.AlternatingDirectionImplicit(),
            ),
            (
                ion.mesh.SphericalHarmonicLengthGaugeOperators(),
                ion.mesh.SplitInteractionOperator(),
            ),
            (
                ion.mesh.SphericalHarmonicVelocityGaugeOperators(),
                ion.mesh.SplitInteractionOperator(),
            ),
        ],
    )
    def test_with_no_potential_final_state_is_initial_state_with_numeric_eigenstates(
        self, initial_state, operators, evolution_method
    ):
        sim = ion.mesh.SphericalHarmonicSpecification(
            "test",
            initial_state=initial_state,
            operators=operators,
            evolution_method=evolution_method,
            time_initial=0,
            time_final=100 * u.asec,
            time_step=1 * u.asec,
            r_bound=50 * u.bohr_radius,
            r_points=250,
            l_bound=30,
            use_numeric_eigenstates=True,
            numeric_eigenstate_max_energy=10 * u.eV,
            numeric_eigenstate_max_angular_momentum=10,
        ).to_sim()

        sim.run()

        initial_norm = sim.data.norm[0]
        final_norm = sim.data.norm[-1]
        initial_state_overlaps = np.array(
            [overlap_vs_time[0] for overlap_vs_time in sim.data.state_overlaps.values()]
        )
        final_state_overlaps = np.array(
            [
                overlap_vs_time[-1]
                for overlap_vs_time in sim.data.state_overlaps.values()
            ]
        )

        np.testing.assert_allclose(initial_norm, final_norm, atol=1e-14)
        np.testing.assert_allclose(
            initial_state_overlaps, final_state_overlaps, atol=1e-14
        )
