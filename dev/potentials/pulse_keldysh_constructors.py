import logging
import os
import itertools

from tqdm import tqdm

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 5,
)

T_PLOT = 10
T_BOUND = 13

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.INFO):
        pulse_type = ion.GaussianPulse
        pulse_widths = np.array([50, 200, 800, 1600]) * asec

        print('\n\n\n\n####### CONSTANT KELDYSH CARRIER ########')
        for pw in pulse_widths:
            pulse = pulse_type.from_keldysh_parameter(pulse_width = pw, keldysh_omega_selector = 'carrier')

            print(pulse.info())

        print('\n\n\n\n####### CONSTANT KELDYSH BANDWIDTH ########')
        for pw in pulse_widths:
            pulse = pulse_type.from_keldysh_parameter(pulse_width = pw, keldysh_omega_selector = 'bandwidth')

            print(pulse.info())
