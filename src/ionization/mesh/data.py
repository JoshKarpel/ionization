import sys
import abc
import collections

import numpy as np

import simulacra.units as u

from .. import exceptions
from . import sims


class Data:
    """A central point on a MeshSimulation for accessing the simulation's Datastores."""

    def __init__(self, sim):
        self.sim = sim
        self.times = sim.data_times

    def __repr__(self):
        return f'{self.__class__.__name__}(sim = {repr(self.sim)}'

    def __getattr__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError as e:
            if item.startswith('__') and item.endswith('__'):
                raise

            datastore = DATA_NAME_TO_DATASTORE_TYPE.get(item)

            if datastore is None:
                raise exceptions.UnknownData(f"Couldn't find any data named '{item}' on {self.sim}. Ensure that the corresponding datastore is correctly implemented.")
            else:
                raise exceptions.MissingDatastore(f"Couldn't get data '{item}' for {self.sim} because it does not include a {datastore.__name__} datastore.")


DATA_NAME_TO_DATASTORE_TYPE = {}


class Datastore(abc.ABC):
    """Handles storing and reporting a type of time-indexed data for a MeshSimulation."""

    @abc.abstractmethod
    def init(self, sim: 'sims.MeshSimulation'):
        """Initialize the data storage containers."""
        self.sim = sim
        self.spec = sim.spec

        self.attach()

    @abc.abstractmethod
    def store(self):
        """Store data for the current time step."""
        raise NotImplementedError

    @abc.abstractmethod
    def attach(self):
        """Attach references to the simulation's Data object."""
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}'


def link_property_to_data(datastore_type, method):
    def get_data_property(data):
        try:
            return getattr(data.sim.datastores[datastore_type.__name__], method.__name__)()
        except KeyError as e:
            raise exceptions.MissingDatastore(f"Couldn't get data {method.__name__} for {data.sim} because it does not include a {datastore_type.__name__} datastore.")

    return property(get_data_property)


class Fields(Datastore):
    """Stores the electric field and vector potential of the simulation's electric potential."""

    def init(self, sim: 'sims.MeshSimulation'):
        self.electric_field_amplitude = sim.get_blank_data()
        self.vector_potential_amplitude = sim.get_blank_data()

        super().init(sim)

    def store(self):
        idx = self.sim.data_time_index
        self.electric_field_amplitude[idx] = self.spec.electric_potential.get_electric_field_amplitude(t = self.sim.time)
        self.vector_potential_amplitude[idx] = self.spec.electric_potential.get_vector_potential_amplitude_numeric(times = self.sim.times_to_current)

    def attach(self):
        self.sim.data.electric_field_amplitude = self.electric_field_amplitude
        self.sim.data.vector_potential_amplitude = self.vector_potential_amplitude

    def __sizeof__(self):
        return self.electric_field_amplitude.nbytes + self.vector_potential_amplitude.nbytes + super().__sizeof__()


DATA_NAME_TO_DATASTORE_TYPE.update({
    'electric_field_amplitude': Fields,
    'vector_potential_amplitude': Fields,
})


class Norm(Datastore):
    """Stores the wavefunction norm as a function of time."""

    def init(self, sim: 'sims.MeshSimulation'):
        self.norm = sim.get_blank_data()

        super().init(sim)

    def store(self):
        self.norm[self.sim.data_time_index] = self.sim.mesh.norm()

    def attach(self):
        self.sim.data.norm = self.norm

    def __sizeof__(self):
        return self.norm.nbytes + super().__sizeof__()


DATA_NAME_TO_DATASTORE_TYPE.update({
    'norm': Norm,
})


class InnerProducts(Datastore):
    """Stores inner products between the wavefunction and the test states. Also handles converting inner products to state overlaps."""

    def init(self, sim: 'sims.MeshSimulation'):
        self.inner_products = {state: sim.get_blank_data(dtype = np.complex128) for state in sim.spec.test_states}

        super().init(sim)

    def store(self):
        for state in self.spec.test_states:
            self.inner_products[state][self.sim.data_time_index] = self.sim.mesh.inner_product(state)

    def state_overlaps(self):
        return {state: np.abs(inner_product) ** 2 for state, inner_product in self.inner_products.items()}

    def initial_state_overlap(self):
        return np.abs(self.inner_products[self.spec.initial_state]) ** 2

    def bound_state_overlap(self):
        return sum(overlap for state, overlap in self.state_overlaps().items() if state.bound)

    def attach(self):
        self.sim.data.inner_products = self.inner_products
        self.sim.data.initial_state_inner_product = self.inner_products[self.spec.initial_state]

    def __sizeof__(self):
        return sum(ip.nbytes for ip in self.inner_products.values()) + sys.getsizeof(self.inner_products) + super().__sizeof__()


Data.state_overlaps = link_property_to_data(InnerProducts, InnerProducts.state_overlaps)
Data.initial_state_overlap = link_property_to_data(InnerProducts, InnerProducts.initial_state_overlap)
Data.bound_state_overlap = link_property_to_data(InnerProducts, InnerProducts.bound_state_overlap)

DATA_NAME_TO_DATASTORE_TYPE.update({
    'inner_products': InnerProducts,
    'initial_state_inner_product': InnerProducts,
    'state_overlaps': InnerProducts,
    'initial_state_overlap': InnerProducts,
    'bound_state_overlap': InnerProducts,
})


class InternalEnergyExpectationValue(Datastore):
    """Stores the expectation value of the internal Hamiltonian."""

    def init(self, sim: 'sims.MeshSimulation'):
        self.internal_energy_expectation_value = sim.get_blank_data()

        super().init(sim)

    def store(self):
        self.internal_energy_expectation_value[self.sim.data_time_index] = self.sim.mesh.internal_energy_expectation_value()

    def attach(self):
        self.sim.data.internal_energy_expectation_value = self.internal_energy_expectation_value

    def __sizeof__(self):
        return self.internal_energy_expectation_value.nbytes + super().__sizeof__()


DATA_NAME_TO_DATASTORE_TYPE.update({
    'internal_energy_expectation_value': InternalEnergyExpectationValue,
})


class TotalEnergyExpectationValue(Datastore):
    """Stores the expectation value of the full Hamiltonian (internal + external)."""

    def init(self, sim: 'sims.MeshSimulation'):
        self.total_energy_expectation_value = sim.get_blank_data()

        super().init(sim)

    def store(self):
        self.total_energy_expectation_value[self.sim.data_time_index] = self.sim.mesh.total_energy_expectation_value()

    def attach(self):
        self.sim.data.total_energy_expectation_value = self.total_energy_expectation_value

    def __sizeof__(self):
        return self.total_energy_expectation_value.nbytes + super().__sizeof__()


DATA_NAME_TO_DATASTORE_TYPE.update({
    'total_energy_expectation_value': TotalEnergyExpectationValue,
})


class ZExpectationValue(Datastore):
    """Stores the expectation value of the z position."""

    def init(self, sim: 'sims.MeshSimulation'):
        self.z_expectation_value = sim.get_blank_data()

        super().init(sim)

    def store(self):
        self.z_expectation_value[self.sim.data_time_index] = self.sim.mesh.z_expectation_value()

    def attach(self):
        self.sim.data.z_expectation_value = self.z_expectation_value

    def z_dipole_moment_expectation_value(self):
        return self.spec.test_charge * self.z_expectation_value

    def __sizeof__(self):
        return self.z_expectation_value.nbytes + super().__sizeof__()


Data.z_dipole_moment_expectation_value = link_property_to_data(ZExpectationValue, ZExpectationValue.z_dipole_moment_expectation_value)

DATA_NAME_TO_DATASTORE_TYPE.update({
    'z_expectation_value': ZExpectationValue,
    'z_dipole_moment_expectation_value': ZExpectationValue,
})


class RExpectationValue(Datastore):
    """Stores the expectation value of the r position."""

    def init(self, sim: 'sims.MeshSimulation'):
        self.r_expectation_value = sim.get_blank_data()

        super().init(sim)

    def store(self):
        self.r_expectation_value[self.sim.data_time_index] = self.sim.mesh.r_expectation_value()

    def attach(self):
        self.sim.data.r_expectation_value = self.r_expectation_value

    def __sizeof__(self):
        return self.r_expectation_value.nbytes + super().__sizeof__()


DATA_NAME_TO_DATASTORE_TYPE.update({
    'r_expectation_value': RExpectationValue,
})


class NormWithinRadius(Datastore):
    def __init__(self, radii = ()):
        self.radii = tuple(sorted(radii))

    def init(self, sim: 'sims.MeshSimulation'):
        self.norm_within_radius = {r: sim.get_blank_data() for r in self.radii}

        super().init(sim)

    def store(self):
        for r in self.radii:
            m = np.where(
                self.sim.mesh.r_mesh <= r,
                self.sim.mesh.g,
                0,
            )
            self.norm_within_radius[r][self.sim.data_time_index] = self.sim.mesh.norm(m)

    def attach(self):
        self.sim.data.norm_within_radius = self.norm_within_radius

    def __sizeof__(self):
        return sum(ip.nbytes for ip in self.norm_within_radius.values()) + sys.getsizeof(self.norm_within_radius) + super().__sizeof__()


DATA_NAME_TO_DATASTORE_TYPE.update({
    'norm_within_radius': NormWithinRadius,
})


class NormByL(Datastore):
    """Stores the norm of the wavefunction in each spherical harmonic."""

    def init(self, sim: 'sims.MeshSimulation'):
        self.norm_by_l = {sph_harm: sim.get_blank_data() for sph_harm in self.spec.spherical_harmonics}

        super().init(sim)

    def store(self):
        norm_by_l = self.sim.mesh.norm_by_l
        for sph_harm, l_norm in zip(self.spec.spherical_harmonics, norm_by_l):
            self.norm_by_l[sph_harm][self.sim.data_time_index] = l_norm

    def attach(self):
        self.sim.data.norm_by_l = self.norm_by_l

    def __sizeof__(self):
        return sum(ip.nbytes for ip in self.norm_by_l.values()) + sys.getsizeof(self.norm_by_l) + super().__sizeof__()


DATA_NAME_TO_DATASTORE_TYPE.update({
    'norm_by_l': NormByL,
})


class DirectionalRadialProbabilityCurrent(Datastore):
    """
    Stores the radial probability current as a function of radius, separated into the positive and negative z directions.
    Only compatible with meshes where one of the coordinates is r, or at least where a consistent r mesh can be extracted (i.e., the number of unique radial coordinates is the same as one of the dimensions of the internal mesh dimensions).
    """

    def init(self, sim: 'sims.MeshSimulation'):
        self.radial_probability_current__pos_z = np.zeros((self.sim.data_time_steps, self.spec.r_points), dtype = np.float64) * np.NaN
        self.radial_probability_current__neg_z = np.zeros((self.sim.data_time_steps, self.spec.r_points), dtype = np.float64) * np.NaN

        theta = self.sim.mesh.theta_calc
        self.d_theta = np.abs(theta[1] - theta[0])
        self.sin_theta = np.sin(theta)
        self.mask = theta <= u.pi / 2

        super().init(sim)

    def store(self):
        radial_current_density = self.sim.mesh.get_radial_probability_current_density_mesh__spatial()

        integrand = radial_current_density * self.sin_theta * self.d_theta * u.twopi  # sin(theta) d_theta from theta integral, twopi from phi integral

        self.radial_probability_current__pos_z[self.sim.data_time_index] = np.sum(integrand[:, self.mask], axis = 1) * (self.sim.mesh.r ** 2)
        self.radial_probability_current__neg_z[self.sim.data_time_index] = np.sum(integrand[:, ~self.mask], axis = 1) * (self.sim.mesh.r ** 2)

    def attach(self):
        self.sim.data.radial_probability_current__pos_z = self.radial_probability_current__pos_z
        self.sim.data.radial_probability_current__neg_z = self.radial_probability_current__neg_z

    def radial_probability_current__total(self):
        return self.radial_probability_current__pos_z + self.radial_probability_current__neg_z

    def __sizeof__(self):
        return self.radial_probability_current__pos_z.nbytes + self.radial_probability_current__neg_z.nbytes + super().__sizeof__()


Data.radial_probability_current__total = link_property_to_data(DirectionalRadialProbabilityCurrent, DirectionalRadialProbabilityCurrent.radial_probability_current__total)

DATA_NAME_TO_DATASTORE_TYPE.update({
    'radial_probability_current__pos_z': DirectionalRadialProbabilityCurrent,
    'radial_probability_current__neg_z': DirectionalRadialProbabilityCurrent,
    'radial_probability_current__total': DirectionalRadialProbabilityCurrent,
})

DATASTORE_TYPE_TO_DATA_NAMES = collections.defaultdict(set)
for data_name, datastore_type in DATA_NAME_TO_DATASTORE_TYPE.items():
    DATASTORE_TYPE_TO_DATA_NAMES[datastore_type].add(data_name)

DEFAULT_DATASTORES = (
    Fields,
    Norm,
    InnerProducts,
)
