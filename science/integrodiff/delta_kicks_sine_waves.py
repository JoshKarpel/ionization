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


def make_sine_wave_kicks(number_of_periods, period, eta):
    kicks = []
    for n in range(number_of_periods):
        kicks.append(dpm.kick(time = (2 * n) * period / 2, time_field_product = eta))
        kicks.append(dpm.kick(time = ((2 * n) + 1) * period / 2, time_field_product = -eta))

    return kicks


def make_eta_movie():
    length = 30
    etas = np.linspace(.1, 1, 30 * length) * time_field_unit

    number_of_periods = 10

    test_width = 1 * bohr_radius
    test_charge = 1 * electron_charge
    test_mass = 1 * electron_mass
    # potential_depth = 36.831335 * eV

    prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge)
    tau_alpha = ide.gaussian_tau_alpha_LEN(test_width, test_mass)
    kernel = functools.partial(ide.gaussian_kernel_LEN, tau_alpha = tau_alpha)
    omega_alpha = 1 / (2 * tau_alpha)

    periods = np.linspace(0, 10, 500) * tau_alpha
    periods = periods[1:]

    def curve(periods, eta):
        results = []
        for period in periods:
            kicks = make_sine_wave_kicks(number_of_periods, period, eta)

            results.append(
                dpm.recursive_kicks(
                    kicks,
                    abs_prefactor = np.abs(prefactor),
                    kernel_func = kernel,
                    bound_state_frequency = omega_alpha
                )[-1])

        return np.abs(results) ** 2

    si.vis.xyt_plot(
        f'bound_state_amplitude_vs_sine_period__eta_scan',
        periods,
        etas,
        curve,
        x_label = r'Sine Wave Period $T$ ($ \tau_{\alpha} $)', x_unit = tau_alpha,
        y_label = 'Bound State Amplitude',
        t_fmt_string = r'$\eta = {} \; \mathrm{{a.u.}}$',
        t_unit = dpm.time_field_unit,
        vlines = [tau_alpha / 2, tau_alpha], vline_kwargs = [{'linestyle': ':', 'color': 'black'}, {'linestyle': ':', 'color': 'black'}],
        y_log_axis = True, y_log_pad = 1,
        y_lower_limit = 1e-9, y_upper_limit = 1,
        length = length,
        **PLT_KWARGS,
    )


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO) as logger:
        # amp = .01 * atomic_electric_field

        # etas = np.array([.001, .005, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]) * dpm.time_field_unit
        #
        # for eta in etas:
        #     number_of_periods = 10
        #
        #     test_width = 1 * bohr_radius
        #     test_charge = 1 * electron_charge
        #     test_mass = 1 * electron_mass
        #     # potential_depth = 36.831335 * eV
        #
        #     prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge)
        #     tau_alpha = ide.gaussian_tau_alpha_LEN(test_width, test_mass)
        #     kernel = functools.partial(ide.gaussian_kernel_LEN, tau_alpha = tau_alpha)
        #     omega_alpha = 1 / (2 * tau_alpha)
        #
        #     periods = np.linspace(.1, 5, 1e3) * tau_alpha
        #
        #     results = []
        #     for period in tqdm(periods):
        #         kicks = make_sine_wave_kicks(number_of_periods, period, eta)
        #
        #         results.append(
        #             dpm.recursive_kicks(
        #                 kicks,
        #                 abs_prefactor = np.abs(prefactor),
        #                 kernel_func = kernel,
        #                 bound_state_frequency = omega_alpha
        #             )[-1])
        #
        #     si.vis.xy_plot(
        #         f'bound_state_amplitude_vs_sine_period__eta={uround(eta, dpm.time_field_unit)}',
        #         periods,
        #         np.abs(results) ** 2,
        #         x_label = r'Sine Wave Period $T$ ($ \tau_{\alpha} $)', x_unit = tau_alpha,
        #         y_label = 'Bound State Amplitude',
        #         vlines = [tau_alpha / 2, tau_alpha], vline_kwargs = [{'linestyle': ':', 'color': 'black'}, {'linestyle': ':', 'color': 'black'}],
        #         **PLT_KWARGS,
        #     )
        #

        make_eta_movie()
