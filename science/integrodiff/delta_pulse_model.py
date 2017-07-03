import logging
import os
import functools

import numpy as np
import scipy.integrate as integrate

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.integrodiff as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 5,
)


@si.utils.memoize
def t1(b):
    # return (1 - (2 * b)) ** 2
    return (1 - b) ** 4


def t2(pulse_delay, b, kernel_func):
    # return (b ** 2) * (1 - (b ** 2)) * (np.abs(kernel_func(pulse_delay)) ** 2)
    return (b ** 2) * ((1 - b) ** 2) * (np.abs(kernel_func(pulse_delay)) ** 2)


def t3(pulse_delay, b, kernel_func, tau_alpha):
    # prefactor = b * (1 - b) * (1 - (b ** 2))
    # return prefactor * 2 * np.real(np.exp(1j * pulse_delay / tau_alpha) * kernel_func(pulse_delay))
    prefactor = b * ((1 - b) ** 3)
    return prefactor * 2 * np.real(np.exp(1j * pulse_delay / tau_alpha) * kernel_func(pulse_delay))


def a_alpha(pulse_delay, *, b, tau_alpha, kernel_func):
    # t1 = (1 - (2 * b)) ** 2  # isolated pulses
    # t2 = (b ** 2) * (1 - (b ** 2)) * (np.abs(kernel_func(pulse_delay)) ** 2)  # first pulse at second time
    # t3 = 2 * (1 - (2 * b)) * b * (1 - b) * kernel_func(pulse_delay) * np.exp(-1j * pulse_delay / tau_alpha)

    # return t1 + t2 + t3

    return t1(b) + t2(pulse_delay, b, kernel_func) + t3(pulse_delay, b, kernel_func, tau_alpha)


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization') as logger:
        test_width = bohr_radius
        test_mass = electron_mass
        test_charge = electron_charge
        efield_tau_product = (1 * atomic_electric_field) * (10 * asec)
        strong_mult = 1.1
        # efield = .11 * atomic_electric_field
        # stronger_efield = 1.25 * efield

        # classical_orbit_time = twopi * bohr_radius / (alpha * c)
        # print('orbit time', uround(classical_orbit_time, asec))

        tau_alpha = ide.gaussian_tau_alpha_LEN(test_width, test_mass)
        kernel = functools.partial(ide.gaussian_kernel_LEN, tau_alpha = tau_alpha)

        b = np.sqrt(pi) * ((test_width * test_charge * efield_tau_product / hbar) ** 2)
        print('b', b)

        stronger_b = np.sqrt(pi) * ((test_width * test_charge * efield_tau_product * strong_mult / hbar) ** 2)
        print('strong b', stronger_b)

        print(f'tau alpha: {uround(tau_alpha, asec)} as')
        print(kernel)

        pulse_delays = np.linspace(0, 1000 * asec, 1e3)
        double_kick = functools.partial(a_alpha, b = b, tau_alpha = tau_alpha, kernel_func = kernel)
        print(double_kick)

        ex = np.exp(-1j * pulse_delays / tau_alpha)
        si.vis.xy_plot(
            'inteference',
            pulse_delays,
            np.real(ex),
            np.imag(ex),
            np.real(kernel(pulse_delays)),
            np.imag(kernel(pulse_delays)),
            2 * np.real(ex + kernel(pulse_delays)),
            line_labels = [
                r'exp real',
                r'exp imag',
                r'ker real',
                r'ker imag',
                r'2 * real(exp + ker)'
            ],
            line_kwargs = [
                {'color': 'C0', 'linestyle': '-'},
                {'color': 'C0', 'linestyle': '--'},
                {'color': 'C1', 'linestyle': '-'},
                {'color': 'C1', 'linestyle': '--'},
                {'color': 'C2', 'linestyle': '-'},
            ],
            legend_on_right = True,
            x_label = 'Half Pulse Delay', x_unit = 'asec',
            **PLT_KWARGS,
        )

        si.vis.xy_plot(
            'double_kick',
            2 * pulse_delays,
            double_kick(pulse_delays),
            np.ones_like(pulse_delays) * t1(b),
            t2(pulse_delays, b, kernel),
            t3(pulse_delays, b, kernel, tau_alpha),
            np.ones_like(pulse_delays) * ((1 - stronger_b) ** 2),
            line_labels = [
                r'Double Kick',
                r'Isolated Pulses',
                r'First pulse seen from second',
                r'Interference',
                r'Single Strong Kick',
            ],
            legend_on_right = True,
            x_label = 'Pulse Delay', x_unit = 'asec',
            # x_extra_ticks = [tau_alpha, classical_orbit_time],
            # x_extra_tick_labels = [r'$ \tau_{\alpha} $', r'$ \tau_{\mathrm{orb}} $'],
            **PLT_KWARGS,
        )
