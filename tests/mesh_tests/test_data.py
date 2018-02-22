import itertools

import pytest

import numpy as np

import simulacra.units as u

import ionization as ion

from .conftest import SPEC_TYPES

DATASTORE_TYPES_SAFE_FOR_ALL_MESHES = [
    ion.mesh.Fields,
    ion.mesh.Norm,
    ion.mesh.InnerProducts,
    ion.mesh.RExpectationValue,
    ion.mesh.ZExpectationValue,
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
        datastores = [datastore_type()],
    ).to_sim()

    for expected_name in expected_names:
        assert hasattr(sim.data, expected_name)


@pytest.mark.parametrize(
    'spec_type', SPEC_TYPES  # parametrize over these because why not
)
def test_bogus_data_attr_raises_exception(spec_type):
    sim = spec_type(
        'test',
        datastores = [],
    ).to_sim()

    with pytest.raises(ion.exceptions.UnknownData):
        sim.data.bogus


@pytest.mark.parametrize(
    'spec_type', SPEC_TYPES
)
@pytest.mark.parametrize(
    'datastore_type', DATASTORE_TYPES_SAFE_FOR_ALL_MESHES,
)
def test_accessing_missing_data_raises_exception(spec_type, datastore_type):
    expected_names = ion.mesh.DATASTORE_TYPE_TO_DATA_NAMES[datastore_type]
    sim = spec_type(
        'test',
        datastores = [],
    ).to_sim()

    for expected_name in expected_names:
        with pytest.raises(ion.exceptions.MissingDatastore):
            getattr(sim.data, expected_name)


@pytest.mark.parametrize(
    'spec_type', SPEC_TYPES
)
def test_spec_doesnt_have_reference_to_datastores_after_sim_init(spec_type):
    spec = spec_type(
        'test',
        datastores = [ion.mesh.Norm()],
    )

    assert hasattr(spec, 'datastores')
    assert hasattr(spec, 'datastore_types')

    sim = spec.to_sim()

    assert not hasattr(spec, 'datastores')
    assert hasattr(spec, 'datastore_types')
    assert hasattr(sim, 'datastores')
