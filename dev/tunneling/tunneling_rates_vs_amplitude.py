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
        landau = ion.tunneling.LandauRate()

        amplitudes = np.linspace(0, 1 * u.atomic_electric_field, 1000)

        tunneling_rates = np.abs(landau.tunneling_rate_from_amplitude(amplitudes, -u.rydberg).squeeze())

        si.vis.xy_plot(
            'tunneling_rates',
            amplitudes,
            tunneling_rates,
            x_unit = 'atomic_electric_field',
            y_unit = 'per_asec',
            **PLOT_KWARGS
        )
