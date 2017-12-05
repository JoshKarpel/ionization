import pytest

import ionization as ion


@pytest.mark.parametrize(
    'spec_type',
    [
        ion.LineSpecification,
        ion.CylindricalSliceSpecification,
        ion.SphericalSliceSpecification,
        ion.SphericalHarmonicSpecification,
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
