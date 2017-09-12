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
    fig_dpi_scale = 6,
)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.INFO):
        pulse = ion.CosSquaredPulse.from_period(
            amplitude = 1 * atomic_electric_field,
            number_of_cycles = 1,
        )

        times = np.linspace(-.5 * pulse.period, .5 * pulse.period, 1e3)

        si.vis.xy_plot(
            'sine_squared_pulse',
            times,
            pulse.get_electric_field_envelope(times) * pulse.amplitude,
            pulse.get_electric_field_amplitude(times),
            line_labels = [
                'Envelope',
                'Pulse',
            ],
            x_unit = 'asec',
            x_label = r'Time $t$',
            y_unit = 'atomic_electric_field',
            y_label = rf'${ion.LATEX_EFIELD}(t)$',
            **PLOT_KWARGS,
        )
