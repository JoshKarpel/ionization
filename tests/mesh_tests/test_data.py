import itertools

import pytest

import numpy as np

import simulacra.units as u

import ionization as ion

from .conftest import SPEC_TYPES

DATASTORE_TYPES_SAFE_FOR_ALL_MESHES = [
    ion.mesh.Fields,
    ion.mesh.Norm,
    ion.mesh.InnerProductsAndOverlaps,
    ion.mesh.RExpectationValue,
    ion.mesh.ZExpectationValue,
    ion.mesh.ZDipoleMomentExpectationValue,
    ion.mesh.InternalEnergyExpectationValue,
    ion.mesh.TotalEnergyExpectationValue,
]


@pytest.mark.parametrize(
    'spec_type', SPEC_TYPES
)
@pytest.mark.parametrize(
    'datastore_type', DATASTORE_TYPES_SAFE_FOR_ALL_MESHES,
)
def test_datastore_names_exist_for_single_datastore(spec_type, datastore_type):
    expected_names = ion.mesh.DATASTORE_TYPE_TO_DATA_NAMES[datastore_type]
    sim = spec_type(
        'test',
        datastore_types = [datastore_type],
    ).to_sim()

    for expected_name in expected_names:
        assert hasattr(sim.data, expected_name)
