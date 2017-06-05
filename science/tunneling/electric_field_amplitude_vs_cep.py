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

if __name__ == '__main__':
    with log as logger:
        pw = 200 * asec
        flu = 1 * Jcm2

        ceps = np.linspace(0, twopi, 1e4)

        bound = 30
        times = np.linspace(-bound * pw, bound * pw, 1e6)
        len_times = len(times)

        amplitude_fraction_vs_cep = np.zeros(len(ceps))

        field_zero = ion.SincPulse(pulse_width = pw, fluence = flu, phase = 0).get_electric_field_amplitude(times)
        field_cut = np.nanmax(np.abs(field_zero)) / np.sqrt(2)

        for ii, cep in enumerate(tqdm(ceps)):
            field = ion.SincPulse(pulse_width = pw, fluence = flu, phase = cep).get_electric_field_amplitude(times)

            amplitude_fraction_vs_cep[ii] = len(field[np.abs(field) > field_cut]) / len_times

        print(amplitude_fraction_vs_cep)
        amplitude_fraction_vs_cep = amplitude_fraction_vs_cep / amplitude_fraction_vs_cep[0]  # normalize to cep = 0
        print(amplitude_fraction_vs_cep)

        plt_kwargs = dict(
                target_dir = OUT_DIR,
        )

        si.vis.xy_plot('amp_fraction_vs_cep',
                       ceps, amplitude_fraction_vs_cep,
                       x_unit = 'rad',
                       **plt_kwargs)
