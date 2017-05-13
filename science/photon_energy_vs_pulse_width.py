import logging
import os

import numpy as np
import simulacra as si
from simulacra.units import *

from src import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = si.utils.LogManager('compy', 'ionization', stdout_level = logging.INFO, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)

if __name__ == '__main__':
    with log as logger:
        pulse_widths = np.linspace(50, 1000, 1e5) * asec
        largest_photon_energy = []
        frequency_cutoff = []

        for pw in pulse_widths:
            largest_photon_energy.append(ion.SincPulse(pulse_width = pw).photon_energy_max)
            frequency_cutoff.append(ion.SincPulse(pulse_width = pw).frequency_max)

        largest_photon_energy = np.array(largest_photon_energy)
        frequency_cutoff = np.array(frequency_cutoff)

        ionization_energy = atomic_energy / 2
        one_to_two = 0.75 * ionization_energy

        si.plots.xy_plot('energy__vs__pulse_width', pulse_widths, largest_photon_energy,
                         target_dir = OUT_DIR,
                         x_unit = 'asec', y_unit = 'eV',
                         x_label = 'Pulse Width', y_label = 'Largest Photon Energy', title = 'Largest Photon Energy for Sinc Pulses',
                         hlines = (ionization_energy, one_to_two), )

        si.plots.xy_plot('energy__vs__pulse_width__log_x', pulse_widths, largest_photon_energy,
                         target_dir = OUT_DIR,
                         x_unit = 'asec', y_unit = 'eV',
                         x_label = 'Pulse Width', y_label = 'Largest Photon Energy', title = 'Largest Photon Energy for Sinc Pulses',
                         hlines = (ionization_energy, one_to_two),
                         x_log_axis = True)

        si.plots.xy_plot('frequency_cutoff__vs__pulse_width', pulse_widths, frequency_cutoff,
                         target_dir = OUT_DIR,
                         x_unit = 'asec', y_unit = 'THz',
                         x_label = 'Pulse Width', y_label = 'Frequency Cutoff', title = 'Frequency Cutoff for Sinc Pulses',
                         )

        si.plots.xy_plot('frequency_cutoff__vs__pulse_width__log_x', pulse_widths, frequency_cutoff,
                         target_dir = OUT_DIR,
                         x_unit = 'asec', y_unit = 'THz',
                         x_label = 'Pulse Width', y_label = 'Frequency Cutoff', title = 'Frequency Cutoff for Sinc Pulses',
                         x_log_axis = True)
