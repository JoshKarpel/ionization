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
        pw = 200 * u.asec

        cos = ion.potentials.SincPulse(pulse_width = pw)
        sin = ion.potentials.SincPulse(pulse_width = pw, phase = u.pi / 2)

        print(cos.info())
        print((1 / pw) / cos.omega_carrier)
        omega_c = 1 / pw
        print(omega_c / u.twopi / u.THz)

        times = np.linspace(-3 * pw, 3 * pw, 10000)
        si.vis.xy_plot(
            'compare',
            times,
            cos.get_electric_field_amplitude(times) / cos.amplitude,
            sin.get_electric_field_amplitude(times) / cos.amplitude,
            cos.get_electric_field_envelope(times),
            -cos.get_electric_field_envelope(times),
            line_labels = [
                r'$\varphi = 0$',
                r'$\varphi = \pi / 2$',
                'Envelope',
                None,
            ],
            line_kwargs = [
                None,
                None,
                {'linestyle': '--', 'color': 'black'},
                {'linestyle': '--', 'color': 'black'},
            ],
            hlines = [
                1 / np.sqrt(2),
            ],
            hline_kwargs = [
                {'linestyle': ':', 'color': 'black'},
            ],
            x_unit = 'asec',
            x_label = r'$t$',
            y_label = r'$\mathcal{E_{\varphi}}(t) / \mathcal{E_{\varphi = 0}}(t=0) $',
            **PLOT_KWARGS
        )
