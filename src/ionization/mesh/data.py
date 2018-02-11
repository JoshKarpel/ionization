from abc import ABC, abstractmethod

import numpy as np

from . import sims


class Data:
    def __init__(self, sim):
        self.sim = sim


class TimeIndexedData(ABC):
    def __init__(self, sim: 'sims.MeshSimulation'):
        self.sim = sim
        self.spec = sim.spec

        self.init()
        self.attach()

    @abstractmethod
    def init(self):
        raise NotImplementedError

    @abstractmethod
    def store(self):
        raise NotImplementedError

    @abstractmethod
    def attach(self):
        raise NotImplementedError


class Norm(TimeIndexedData):
    def init(self):
        self.norm = self.sim.get_blank_data()

    def store(self):
        self.norm[self.sim.data_time_index] = self.sim.mesh.norm()

    def attach(self):
        self.sim.data.norm = self.norm


class InnerProducts(TimeIndexedData):
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
