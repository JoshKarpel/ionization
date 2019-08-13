import pytest

from .conftest import SPEC_TYPES

DATASTORE_TYPES_SAFE_FOR_ALL_MESHES = [
    mesh.Fields,
    mesh.Norm,
    mesh.InnerProducts,
    mesh.RExpectationValue,
    mesh.ZExpectationValue,
    mesh.InternalEnergyExpectationValue,
    mesh.TotalEnergyExpectationValue,
]


@pytest.mark.parametrize("spec_type", SPEC_TYPES)
@pytest.mark.parametrize("datastore_type", DATASTORE_TYPES_SAFE_FOR_ALL_MESHES)
def test_datastore_names_exist_for_single_datastore(spec_type, datastore_type):
    expected_names = mesh.DATASTORE_TYPE_TO_DATA_NAMES[datastore_type]
    sim = spec_type("test", datastores=[datastore_type()]).to_sim()

    for expected_name in expected_names:
        assert hasattr(sim.data, expected_name)


@pytest.mark.parametrize(
    "spec_type", SPEC_TYPES  # parametrize over these because why not
)
def test_bogus_data_attr_raises_exception(spec_type):
    sim = spec_type("test", datastores=[]).to_sim()

    with pytest.raises(exceptions.UnknownData):
        sim.data.bogus


@pytest.mark.parametrize("spec_type", SPEC_TYPES)
@pytest.mark.parametrize("datastore_type", DATASTORE_TYPES_SAFE_FOR_ALL_MESHES)
def test_accessing_missing_data_raises_exception(spec_type, datastore_type):
    expected_names = mesh.DATASTORE_TYPE_TO_DATA_NAMES[datastore_type]
    sim = spec_type("test", datastores=[]).to_sim()

    for expected_name in expected_names:
        with pytest.raises(exceptions.MissingDatastore):
            getattr(sim.data, expected_name)


@pytest.mark.parametrize("spec_type", SPEC_TYPES)
def test_cannot_duplicate_datastores(spec_type):
    with pytest.raises(exceptions.DuplicateDatastores):
        spec_type("test", datastores=[mesh.Norm(), mesh.Norm()])


@pytest.mark.parametrize("spec_type", SPEC_TYPES)
def test_to_sim_multiple_times_produces_new_datastores(spec_type):
    spec = spec_type("test")

    sim_one = spec.to_sim()
    sim_two = spec.to_sim()

    assert sim_one != sim_two
    for ds_type in spec.datastore_types:
        assert (
            sim_one.datastores_by_type[ds_type] != sim_two.datastores_by_type[ds_type]
        )
