#!/usr/bin/env python

import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

if __name__ == '__main__':
    with LOGMAN as logger:
        window = ion.potentials.SymmetricExponentialTimeWindow(
            window_time = 500 * u.asec,
            window_width = 50 * u.asec,
            window_center = 100 * u.asec,
        )
        times = np.linspace(-1, 1, 10000) * 1000 * u.asec

        si.vis.xy_plot(
            'symmetric_exponential_window',
            times,
            window(times),
            x_unit = 'asec',
            x_label = r'$t$',
            vlines = np.array([50, 500, 550, 600, 650, 700, 750]) * u.asec,
            **PLOT_KWARGS,
        )
