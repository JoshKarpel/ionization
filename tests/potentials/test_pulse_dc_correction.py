import pytest

import numpy as np

import simulacra.units as u

import ionization as ion


@pytest.mark.parametrize(
    'spec_type',
    [
        ion.mesh.LineSpecification,
        ion.mesh.CylindricalSliceSpecification,
        ion.mesh.SphericalSliceSpecification,
        ion.mesh.SphericalHarmonicSpecification,
        ion.ide.IntegroDifferentialEquationSpecification,
    ]
)
def test_dc_correct_electric_potential_replaces_given_potential_after_simulation_initialization(spec_type):
    pot = ion.potentials.SincPulse()

    spec = spec_type(
        'test',
        electric_potential = pot,
        electric_potential_dc_correction = True,
    )

    assert spec.electric_potential is pot

    sim = spec.to_simulation()

    assert sim.spec.electric_potential is not pot
    assert sim.spec.electric_potential[0] is pot
    assert isinstance(sim.spec.electric_potential[1], ion.potentials.Rectangle)


def test_dc_corrected_pulse_has_zero_vector_potential_before_and_after_pulse():
    times = np.linspace(0 * u.asec, 100 * u.asec, 1000)
    pulse = ion.potentials.Rectangle(start_time = 25 * u.asec, end_time = 75 * u.asec)

    pulse_vp = pulse.get_vector_potential_amplitude_numeric_cumulative(times)
    assert pulse_vp[0] == 0
    assert pulse_vp[-1] != 0

    corrected_pulse = ion.potentials.DC_correct_electric_potential(pulse, times)

    corrected_pulse_vp = corrected_pulse.get_vector_potential_amplitude_numeric_cumulative(times)
    assert corrected_pulse_vp[0] == 0
    assert np.isclose(corrected_pulse_vp[-1], 0)
