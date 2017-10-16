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
        for nc in [.1, 0.5, .75, 1.0, 1.1, 1.5, 1.9, 2.0, 3.0, 4.0]:
            pulse = ion.CosSquaredPulse.from_pulse_width(
                pulse_width = 200 * asec,
                amplitude = 1 * atomic_electric_field,
                number_of_cycles = nc,
            )

            times = np.linspace(-1 * pulse.pulse_width, 1 * pulse.pulse_width, 1e3)

            freq_lim = pulse.omega_carrier + (2 * pulse.sideband_offset)
            freqs = np.linspace(-freq_lim, freq_lim, 1e3)

            si.vis.xy_plot(
                f'cos_squared_pulse__field__Nc={nc}',
                times,
                pulse.get_electric_field_envelope(times) * pulse.amplitude / atomic_electric_field,
                pulse.get_electric_field_amplitude(times) / atomic_electric_field,
                pulse.get_vector_potential_amplitude_numeric_cumulative(times) * proton_charge / atomic_momentum,
                line_labels = [
                    r'$F_{\tau}(t)$',
                    r'$\mathcal{E}(t)$',
                    r'$q \, \mathcal{A}(t)$',
                ],
                x_unit = 'asec',
                x_label = r'Time $t$',
                y_label = r'Fields ($\mathrm{a.u.}$)',
                vlines = [
                    -pulse.pulse_width / 2,
                    pulse.pulse_width / 2,
                ],
                **PLOT_KWARGS,
            )

            si.vis.xy_plot(
                f'cos_squared_pulse__ft__Nc={nc}',
                freqs,
                vlines = [
                    pulse.omega_carrier,
                    -pulse.omega_carrier,
                    pulse.omega_carrier - pulse.sideband_offset,
                    pulse.omega_carrier + pulse.sideband_offset,
                    -pulse.omega_carrier - pulse.sideband_offset,
                    -pulse.omega_carrier + pulse.sideband_offset,
                ],
                vline_kwargs = [
                    {'ymax': .8, 'color': 'C0'},
                    {'ymax': .8, 'color': 'C1', 'linestyle': '--'},
                    {'ymax': .5, 'color': 'C0'},
                    {'ymax': .5, 'color': 'C0'},
                    {'ymax': .5, 'color': 'C1', 'linestyle': '--'},
                    {'ymax': .5, 'color': 'C1', 'linestyle': '--'},
                ],
                x_unit = twopi * THz,
                x_label = r'$f$ ($\mathrm{THz}$)',
                y_lower_limit = 0, y_upper_limit = 1, y_pad = 0,
                title = fr'$\tau = {uround(pulse.pulse_width, asec)} \, \mathrm{{as}}, \; N_c = {nc}$',
                **PLOT_KWARGS,
            )
