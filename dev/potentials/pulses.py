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

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG):
        pulse_types = [ion.SincPulse, ion.GaussianPulse, ion.SechPulse]
        pulse_widths = np.array([50, 100, 200, 400, 800, 1600]) * asec
        fluences = np.array([.1, 1]) * Jcm2
        phases = [0, pi / 4, pi / 2]
        omega_mins = np.array([300, 500, 800]) * twopi * THz

        t_bound = 10

        for pulse_type, pw, flu, phase, omega_min in tqdm(itertools.product(pulse_types, pulse_widths, fluences, phases, omega_mins)):
            name = f'{pulse_type.__name__}__PW={uround(pw, asec)}as__FLU={uround(flu, Jcm2)}jcm2__PHA={uround(phase, pi)}pi__OmegaMin=2pi_x_{uround(omega_min, twopi * THz)}THz'

            kwargs = dict(
                pulse_width = pw,
                fluence = flu,
                phase = phase,
                omega_min = omega_min,
            )

            pulse = pulse_type.from_omega_min(**kwargs)
            times = np.linspace(-t_bound * pw, t_bound * pw, 1000)

            si.vis.xy_plot(
                name,
                times,
                pulse.get_electric_field_amplitude(times),
                x_label = r'$ t $', x_unit = 'asec',
                y_label = fr'$ {ion.LATEX_EFIELD}(t) $', y_unit = 'atomic_electric_field',
                **PLOT_KWARGS,
            )

