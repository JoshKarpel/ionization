import sys
import abc

import numpy as np

import simulacra.units as u

from . import sims


class Data:
    """A central point on a MeshSimulation for accessing the simulation's Datastores."""

    def __init__(self, sim):
        self.sim = sim
        self.times = sim.data_times

    def __repr__(self):
        return f'{self.__class__.__name__}(sim = {repr(self.sim)}'


class Datastore(abc.ABC):
    """Handles storing and reporting a type of time-indexed data for a MeshSimulation."""

    def __init__(self, sim: 'sims.MeshSimulation'):
        self.sim = sim
        self.spec = sim.spec

        self.init()
        self.attach()

    @abc.abstractmethod
    def init(self):
        """Initialize the data storage containers."""
        raise NotImplementedError

    @abc.abstractmethod
    def store(self):
        """Store data for the current time step."""
        raise NotImplementedError

    @abc.abstractmethod
    def attach(self):
        """Attach references to the simulation's Data object."""
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}(sim = {repr(self.sim)})'


class Norm(Datastore):
    """Stores the wavefunction norm as a function of time."""

    def init(self):
        self.norm = self.sim.get_blank_data()

    def store(self):
        self.norm[self.sim.data_time_index] = self.sim.mesh.norm()

    def attach(self):
        self.sim.data.norm = self.norm

    def __sizeof__(self):
        return self.norm.nbytes + super().__sizeof__()


class InnerProducts(Datastore):
    """Stores inner products between the wavefunction and the test states. Also handles converting inner products to state overlaps."""

    def init(self):
        self.inner_products = {state: self.sim.get_blank_data(dtype = np.complex128) for state in self.spec.test_states}

    def store(self):
        for state in self.spec.test_states:
            self.inner_products[state][self.sim.data_time_index] = self.sim.mesh.inner_product(state)

    def state_overlaps(self):
        return {state: np.abs(inner_product) ** 2 for state, inner_product in self.inner_products.items()}

    def initial_state_overlap(self):
        return np.abs(self.inner_products[self.spec.initial_state]) ** 2

    def attach(self):
        self.sim.data.inner_products = self.inner_products
        self.sim.data.initial_state_inner_product = self.inner_products[self.spec.initial_state]

    Data.state_overlaps = property(lambda data: data.sim.datastores['InnerProducts'].state_overlaps())
    Data.initial_state_overlap = property(lambda data: data.sim.datastores['InnerProducts'].initial_state_overlap())

    def __sizeof__(self):
        return sum(ip.nbytes for ip in self.inner_products.values()) + sys.getsizeof(self.inner_products) + super().__sizeof__()


class Fields(Datastore):
    """Stores the electric field and vector potential of the simulation's electric potential."""

    def init(self):
        self.electric_field_amplitude = self.sim.get_blank_data()
        self.vector_potential_amplitude = self.sim.get_blank_data()

    def store(self):
        idx = self.sim.data_time_index
        self.electric_field_amplitude[idx] = self.spec.electric_potential.get_electric_field_amplitude(t = self.sim.time)
        self.vector_potential_amplitude[idx] = self.spec.electric_potential.get_vector_potential_amplitude_numeric(times = self.sim.times_to_current)

    def attach(self):
        self.sim.data.electric_field_amplitude = self.electric_field_amplitude
        self.sim.data.vector_potential_amplitude = self.vector_potential_amplitude

    def __sizeof__(self):
        return self.electric_field_amplitude.nbytes + self.vector_potential_amplitude.nbytes + super().__sizeof__()


class RadialPositionExpectationValue(Datastore):
    """Stores the expectation value of the radial position."""

    def init(self):
        self.radial_position_expectation_value = self.sim.get_blank_data()

    def store(self):
        self.radial_position_expectation_value[self.sim.data_time_index] = self.sim.mesh.radial_position_expectation_value()

    def attach(self):
        self.sim.data.radial_position_expectation_value = self.radial_position_expectation_value

    def __sizeof__(self):
        return self.radial_position_expectation_value.nbytes + super().__sizeof__()


class InternalEnergyExpectationValue(Datastore):
    """Stores the expectation value of the internal Hamiltonian."""

    def init(self):
        self.internal_energy_expectation_value = self.sim.get_blank_data()

    def store(self):
        self.internal_energy_expectation_value[self.sim.data_time_index] = self.sim.mesh.energy_expectation_value(include_interaction = False)

    def attach(self):
        self.sim.data.internal_energy_expectation_value = self.internal_energy_expectation_value

    def __sizeof__(self):
        return self.internal_energy_expectation_value.nbytes + super().__sizeof__()


class TotalEnergyExpectationValue(Datastore):
    """Stores the expectation value of the full Hamiltonian (internal + external)."""

    def init(self):
        self.total_energy_expectation_value = self.sim.get_blank_data()

    def store(self):
        self.total_energy_expectation_value[self.sim.data_time_index] = self.sim.mesh.energy_expectation_value(include_interaction = True)

    def attach(self):
        self.sim.data.total_energy_expectation_value = self.total_energy_expectation_value

    def __sizeof__(self):
        return self.total_energy_expectation_value.nbytes + super().__sizeof__()


class ElectricDipoleMomentZExpectationValue(Datastore):
    """Stores the expectation value of the electric dipole moment in the z direction."""

    def init(self):
        self.z_dipole_moment_expectation_value = self.sim.get_blank_data(dtype = np.complex128)

    def store(self):
        self.z_dipole_moment_expectation_value[self.sim.data_time_index] = np.real(self.sim.mesh.z_dipole_moment_inner_product())

    def attach(self):
        self.sim.data.z_dipole_moment_expectation_value = self.z_dipole_moment_expectation_value

    def __sizeof__(self):
        return self.z_dipole_moment_expectation_value.nbytes + super().__sizeof__()


class NormByL(Datastore):
    """Stores the norm of the wavefunction in each spherical harmonic."""

    def init(self):
        self.norm_by_l = {sph_harm: self.sim.get_blank_data() for sph_harm in self.spec.spherical_harmonics}

    def store(self):
        norm_by_l = self.sim.mesh.norm_by_l
        for sph_harm, l_norm in zip(self.spec.spherical_harmonics, norm_by_l):
            self.norm_by_l[sph_harm][self.sim.data_time_index] = l_norm

    def attach(self):
        self.sim.data.norm_by_l = self.norm_by_l

    def __sizeof__(self):
        return sum(ip.nbytes for ip in self.norm_by_l.values()) + sys.getsizeof(self.norm_by_l) + super().__sizeof__()


class DirectionalRadialProbabilityCurrent(Datastore):
    """
    Stores the radial probability current as a function of radius, separated into the positive and negative z directions.
    Only compatible with meshes where one of the coordinates is r, or at least where a consistent r mesh can be extracted (i.e., the number of unique radial coordinates is the same as one of the dimensions of the internal mesh dimensions).
    """

    def init(self):
        self.radial_probability_current__pos_z = np.zeros((self.sim.data_time_steps, self.spec.r_points), dtype = np.float64) * np.NaN
        self.radial_probability_current__neg_z = np.zeros((self.sim.data_time_steps, self.spec.r_points), dtype = np.float64) * np.NaN

        theta = self.sim.mesh.theta_calc
        self.d_theta = np.abs(theta[1] - theta[0])
        self.sin_theta = np.sin(theta)
        self.mask = theta <= u.pi / 2

    def store(self):
        radial_current_density = self.sim.mesh.get_radial_probability_current_density_mesh__spatial()

        integrand = radial_current_density * self.sin_theta * self.d_theta * u.twopi  # sin(theta) d_theta from theta integral, twopi from phi integral

        self.radial_probability_current__pos_z[self.sim.data_time_index] = np.sum(integrand[:, self.mask], axis = 1) * (self.sim.mesh.r ** 2)
        self.radial_probability_current__neg_z[self.sim.data_time_index] = np.sum(integrand[:, ~self.mask], axis = 1) * (self.sim.mesh.r ** 2)

    def attach(self):
        self.sim.data.radial_probability_current__pos_z = self.radial_probability_current__pos_z
        self.sim.data.radial_probability_current__neg_z = self.radial_probability_current__neg_z

    def radial_probability_current_vs_time(self):
        return self.radial_probability_current__pos_z + self.radial_probability_current__neg_z

    Data.radial_probability_current_vs_time = property(lambda data: data.sim.datastores['DirectionalRadialProbabilityCurrent'].radial_probability_current_vs_time())

    def __sizeof__(self):
        return self.radial_probability_current__pos_z.nbytes + self.radial_probability_current__neg_z.nbytes + super().__sizeof__()


DEFAULT_DATASTORES = (Norm, InnerProducts, Fields)
