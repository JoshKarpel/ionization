import logging
import warnings

import numpy as np

import simulacra.units as u

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def landau_tunneling_rate(electric_field_amplitude, ionization_potential = -u.rydberg):
    """The instantaneous tunneling rate calculated by Landau. See Landau & Lifshitz QM."""
    amplitude_scaled = np.abs(electric_field_amplitude) / u.atomic_electric_field
    potential_scaled = np.abs(ionization_potential) / u.hartree

    f = amplitude_scaled / ((2 * potential_scaled) ** 1.5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # ignore div by zero errors
        rate = (4 / f) * np.exp(-2 / (3 * f)) / u.atomic_time,

    return np.where(
        np.isclose(amplitude_scaled, 0),
        0,
        rate
    )
