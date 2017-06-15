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
    # f = np.abs(electric_field_amplitude / atomic_electric_field)
    #
    # return (4 / f) * (electron_mass_reduced * (proton_charge ** 4) / (hbar ** 3)) * np.exp(-(2 / 3) / f)

    amplitude_scaled = np.abs(electric_field_amplitude / atomic_electric_field)
    potential_scaled = np.abs(ionization_potential / hartree)

    f = amplitude_scaled / ((2 * potential_scaled) ** 1.5)

    return (4 / f) * np.exp(-2 / (3 * f)) / atomic_time

    # e_a = (electron_mass_reduced ** 2) * (proton_charge ** 5) / (((4 * pi * epsilon_0) ** 3) * (hbar ** 4))
    # w_a = (electron_mass_reduced * (proton_charge ** 4)) / (((4 * pi * epsilon_0) ** 2) * (hbar ** 3))
    # f = e_a / np.abs(electric_field_amplitude)

    # return 4 * w_a * f * np.exp(-2 * f / 3)


class TunnelingIonizationSimulation(si.Simulation):
    pass


class TunnelingIonizationSpecification(si.Specification):
    pass
