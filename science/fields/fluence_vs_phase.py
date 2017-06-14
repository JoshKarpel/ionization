import logging
import os

from tqdm import tqdm

import numpy as np
import numpy.fft as fft
import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager(
    'simulacra', 'ionization',
    stdout_level = logging.INFO
)

PLT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 3,
)

PULSE_TYPES = (
    ion.SincPulse,
    ion.GaussianPulse,
    ion.SechPulse,
)

if __name__ == '__main__':
    with logman as logger:
        t_bound = 35
        p_bound = 30

        pw = 200 * asec
        flu = 1 * Jcm2
        phases = np.linspace(0, twopi, 1e4)

        times = np.linspace(-t_bound * pw, t_bound * pw, 2 ** 16)
        dt = np.abs(times[1] - times[0])

        window = window = ion.SymmetricExponentialTimeWindow(window_time = p_bound * pw, window_width = .2 * pw)
        # dummy = ion.SincPulse(pulse_width = pw, fluence = flu, phase = 0,
        #                       )
        # pulses = [
        #     dummy,
        #     ion.GaussianPulse(pulse_width = pw, fluence = flu, omega_carrier = dummy.omega_carrier, phase = dummy.phase,
        #                       window = dummy.window),
        #     ion.SechPulse(pulse_width = pw, fluence = flu, omega_carrier = dummy.omega_carrier, phase = dummy.phase,
        #                   window = dummy.window),
        # ]

        fluences = {pulse_type: np.empty_like(phases) for pulse_type in PULSE_TYPES}

        for pulse_type in PULSE_TYPES:
            for ii, phase in enumerate(tqdm(phases)):
                pulse = ion.SincPulse(pulse_width = pw, fluence = flu, phase = phase,
                                      window = window)
                if pulse_type != ion.SincPulse:
                    pulse = pulse_type(pulse_width = pw, fluence = flu, phase = phase, omega_carrier = pulse.omega_carrier,
                                       window = window)
                fluences[pulse_type][ii] = pulse.get_fluence_numeric(times)

        si.vis.xy_plot(
            'fluence_vs_phase',
            phases,
            *(flu_vs_phase for pulse, flu_vs_phase in fluences.items()),
            line_labels = (pulse.__name__ for pulse in fluences),
            x_label = r'$ \varphi $', x_unit = 'rad',
            y_label = r'$ H $', y_unit = 'Jcm2',
            **PLT_KWARGS,
        )

        si.vis.xy_plot(
            'fluence_vs_phase__rel_log',
            phases,
            *(np.abs(1 - (flu_vs_phase / flu_vs_phase[0])) for pulse, flu_vs_phase in fluences.items()),
            line_labels = (pulse.__name__ for pulse in fluences),
            x_label = r'$ \varphi $', x_unit = 'rad',
            y_label = r'$ H $', y_log_axis = True,
            **PLT_KWARGS,
        )