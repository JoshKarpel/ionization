import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

import simulacra as si
from simulacra.units import *

from . import core, potentials, states

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def instantaneous_tunneling_rate(electric_field_amplitude, ionization_potential = -rydberg):
    amplitude_scaled = np.abs(electric_field_amplitude) / atomic_electric_field
    potential_scaled = np.abs(ionization_potential) / hartree

    f = amplitude_scaled / ((2 * potential_scaled) ** 1.5)

    return (4 / f) * np.exp(-2 / (3 * f)) / atomic_time


class TunnelingIonizationSimulation(si.Simulation):
    pass


class TunnelingIonizationSpecification(si.Specification):
    pass
