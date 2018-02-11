import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def cb(sim):
    if not hasattr(sim, 'foo'):
        sim.foo = np.empty_like(sim.data_times, dtype = np.complex128) * np.NaN

    sim.foo[sim.data_time_index] = sim.time_index


if __name__ == '__main__':
    with LOGMAN as logger:
        sim = ion.SphericalHarmonicSpecification(
            'callback_test',
            store_data_callbacks = [cb],
            store_data_every = 2,
        ).to_sim()

        sim.run()

        print(type(sim))
        print(sim.foo)
        print(len(sim.foo))
