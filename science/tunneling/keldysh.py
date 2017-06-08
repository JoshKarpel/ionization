import logging
import os

from tqdm import tqdm

import numpy as np

import simulacra as si
from simulacra.units import *
import ionization as ion


# import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)


PLT_KWARGS = dict(
        target_dir = OUT_DIR,
)


def monochromatic_keldysh_1d(photon_energy_min, photon_energy_max, electric_field_amplitudes, ionization_potential = -rydberg):
    energies = np.linspace(photon_energy_min, photon_energy_max, 1e5)
    gammas = []
    labels = []
    for amplitude in electric_field_amplitudes:
        efields = (ion.SineWave.from_photon_energy(photon_energy, amplitude = amplitude) for photon_energy in energies)
        gammas.append(np.array(list(efield.keldysh_parameter(ionization_potential) for efield in efields)))
        labels.append(fr'${ion.LATEX_EFIELD}_0 = {uround(amplitude, atomic_electric_field)} \; \mathrm{{a.u.}}$')

    si.vis.xy_plot(
            f'1d_keldysh__{uround(photon_energy_min, eV)}eV_to_{uround(photon_energy_max, eV)}eV',
            energies,
            *gammas,
            line_labels = labels,
            x_label = r'Photon Energy $E$', x_unit = 'eV',
            y_label = r'Keldysh Parameter $\gamma$', y_lower_limit = 0, y_upper_limit = 10, y_pad = 0,
            hlines = [1], hline_kwargs = [{'linestyle': '--', 'color': 'black'}],
            title = 'Keldysh Parameter vs. Photon Energy',
            **PLT_KWARGS,
    )


if __name__ == '__main__':
    with log as logger:
        monochromatic_keldysh_1d(.1 * eV, 50 * eV, atomic_electric_field * np.array([.01, .05, .1, .5, 1, 5]))
