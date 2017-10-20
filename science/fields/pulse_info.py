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

log = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

if __name__ == '__main__':
    with log as logger:
        pulse_widths_sparse = np.array([50, 100, 200, 400, 800]) * asec
        fluences_sparse = np.array([.1, 1, 5, 10, 20]) * Jcm2

        pulse_widths_dense = np.linspace(50, 1000, 1e3) * asec
        fluences_dense = np.linspace(0, 20, 1e3) * Jcm2

        si.vis.xy_plot(
            'pulse_width_scan',
            pulse_widths_dense,
            *[[ion.SincPulse(pulse_width = pw, fluence = flu).amplitude for pw in pulse_widths_dense]
              for flu in fluences_sparse],
            line_labels = [rf'$ H = {uround(flu, Jcm2, 1)} \, \mathrm{{J/cm^2}} $' for flu in fluences_sparse],
            x_label = r'Pulse Width $ \tau $', x_unit = 'asec',
            y_label = rf'Field Prefactor $ {ion.LATEX_EFIELD}_0 $', y_unit = 'atomic_electric_field',
            y_lower_limit = 0, y_upper_limit = 3.5 * atomic_electric_field, y_pad = 0,
            title = 'Electric Field Prefactor vs. Pulse Width',
            font_size_legend = 10,
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            'fluence_scan',
            fluences_dense,
            *[[ion.SincPulse(pulse_width = pw, fluence = flu).amplitude for flu in fluences_dense]
              for pw in pulse_widths_sparse],
            line_labels = [rf'$ \tau = {uround(pw, asec, 1)} \, \mathrm{{as}} $' for pw in pulse_widths_sparse],
            x_label = r'Fluence $ H $', x_unit = 'Jcm2',
            y_label = rf'Field Prefactor $ {ion.LATEX_EFIELD}_0 $', y_unit = 'atomic_electric_field',
            y_lower_limit = 0, y_upper_limit = 3.5 * atomic_electric_field, y_pad = 0,
            title = 'Electric Field Prefactor vs. Fluence',
            font_size_legend = 10,
            **PLOT_KWARGS,
        )

        fluences_log = np.geomspace(.01, 20, 1e3) * Jcm2
        si.vis.xy_plot(
            'fluence_scan_log',
            fluences_log,
            *[[ion.SincPulse(pulse_width = pw, fluence = flu).amplitude for flu in fluences_log]
              for pw in pulse_widths_sparse],
            line_labels = [rf'$ \tau = {uround(pw, asec, 1)} \, \mathrm{{as}} $' for pw in pulse_widths_sparse],
            x_label = r'Fluence $ H $', x_unit = 'Jcm2',
            x_log_axis = True,
            y_label = rf'Field Prefactor $ {ion.LATEX_EFIELD}_0 $', y_unit = 'atomic_electric_field',
            y_lower_limit = 0, y_upper_limit = 3.5 * atomic_electric_field, y_pad = 0,
            title = 'Electric Field Prefactor vs. Fluence',
            font_size_legend = 10,
            **PLOT_KWARGS,
        )
