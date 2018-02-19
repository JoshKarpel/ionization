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
        tunneling_models = [
            ion.tunneling.LandauRate(),
            ion.tunneling.KeldyshRate(),
            ion.tunneling.PosthumusRate(),
            ion.tunneling.MulserRate(),
            ion.tunneling.ADKRate(),
            # ion.tunneling.ADKExtendedToBSIRate(),
        ]

        amplitudes = np.geomspace(.04, 10, 1000) * u.atomic_electric_field

        tunneling_rates = [
            np.abs(model.tunneling_rate_from_amplitude(amplitudes, -u.rydberg).squeeze())
            for model in tunneling_models
        ]
        labels = [model.__class__.__name__.rstrip('Rate') for model in tunneling_models]

        si.vis.xy_plot(
            'tunneling_rates',
            amplitudes,
            *tunneling_rates,
            line_labels = labels,
            x_label = r'Electric Field Amplitude $ \mathcal{E} $',
            y_label = r'Tunneling Rate $ W $',
            x_unit = 'atomic_electric_field',
            y_unit = 'per_atomic_time',
            x_log_axis = True,
            y_log_axis = True,
            y_lower_limit = .001 / u.atomic_time,
            y_upper_limit = 10 / u.atomic_time,
            y_log_pad = 1,
            legend_on_right = True,
            **PLOT_KWARGS
        )
