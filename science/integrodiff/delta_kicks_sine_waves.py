import logging
import os
import functools
import collections

from tqdm import tqdm

import numpy as np
import scipy.integrate as integrate

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.integrodiff as ide

import delta_pulse_model as dpm

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 5,
)

time_field_unit = atomic_electric_field * atomic_time

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO) as logger:
        amp = .01 * atomic_electric_field

        test_width = 1 * bohr_radius
        test_charge = 1 * electron_charge
        test_mass = 1 * electron_mass
        # potential_depth = 36.831335 * eV

        prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge)
        tau_alpha = ide.gaussian_tau_alpha_LEN(test_width, test_mass)
        kernel = functools.partial(ide.gaussian_kernel_LEN, tau_alpha = tau_alpha)
        omega_alpha = 1 / (2 * tau_alpha)

        periods = np.linspace(.1, 10, 1e3) * tau_alpha

        results = []
        for period in tqdm(periods):
            efield = ion.SineWave.from_period(period = period, amplitude = amp)
            print(efield.info())
            t_bound = 10 * efield.period
            times = np.linspace(0, t_bound, 1e4)

            kicks = dpm.decompose_pulse_into_kicks__amplitude(efield, times)
            print(kicks)
            results.append(
                dpm.recursive_kicks(
                    kicks,
                    abs_prefactor = np.abs(prefactor),
                    kernel_func = kernel,
                    bound_state_frequency = omega_alpha
                )[-1])

        si.vis.xy_plot(
            'bound_state_amplitude_vs_sine_period',
            periods,
            np.abs(results) ** 2,
            x_label = r'Sine Wave Period $T$ ($ \tau_{\alpha} $)', x_unit = tau_alpha,
            y_label = 'Bound State Amplitude',
            **PLT_KWARGS,
        )
