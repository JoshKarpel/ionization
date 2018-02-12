import pytest

import numpy as np

from tests import testutils

from .test_sims import SPEC_TYPES


@pytest.mark.parametrize(
    'spec_type',
    SPEC_TYPES
)
def test_loaded_sim_mesh_is_same_as_original(spec_type, tmpdir):
    sim_type = spec_type.simulation_type
    sim = spec_type('test').to_sim()

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
def test_loaded_sim_mesh_is_none_if_not_saved(spec_type, tmpdir):
    sim_type = spec_type.simulation_type
    sim = spec_type('test').to_sim()

    path = sim.save(tmpdir, save_mesh = False)
    loaded_sim = sim_type.load(path)

    assert loaded_sim.mesh is None


@pytest.mark.parametrize(
    'spec_type',
    SPEC_TYPES
)
def test_can_save_sim_that_already_has_no_mesh(spec_type, tmpdir):
    sim_type = spec_type.simulation_type
    sim = spec_type('test').to_sim()

    sim.mesh = None

    path = sim.save(tmpdir, save_mesh = False)
    loaded_sim = sim_type.load(path)

    assert loaded_sim.mesh is None
