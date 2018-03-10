import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np
import scipy.interpolate as interp

import ionization as ion
import ionization.ide as ide
import simulacra.units as u

from .conftest import TUNNELING_MODEL_TYPES


@pytest.mark.parametrize('model_type', TUNNELING_MODEL_TYPES)
def test_no_tunneling_if_no_electric_field(model_type):
    sim = ion.tunneling.TunnelingSpecification(
        'test',
        time_initial = 0,
        time_step = .1 * u.fsec,
        time_final = 10 * u.fsec,
        tunneling_model = model_type(),
        electric_potential = ion.potentials.NoElectricPotential(),
    ).to_sim()

    sim.run()

    assert sim.b[-1] == sim.spec.b_initial
    assert sim.b2[-1] == np.abs(sim.spec.b_initial) ** 2
