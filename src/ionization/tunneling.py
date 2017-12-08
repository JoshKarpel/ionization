import logging

import numpy as np

import simulacra as si
import simulacra.units as u

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def instantaneous_tunneling_rate(electric_field_amplitude, ionization_potential = -u.rydberg):
    amplitude_scaled = np.abs(electric_field_amplitude) / u.atomic_electric_field
    potential_scaled = np.abs(ionization_potential) / u.hartree

    f = amplitude_scaled / ((2 * potential_scaled) ** 1.5)

    return (4 / f) * np.exp(-2 / (3 * f)) / u.atomic_time


class TunnelingIonizationSimulation(si.Simulation):
    pass


class TunnelingIonizationSpecification(si.Specification):
    pass
