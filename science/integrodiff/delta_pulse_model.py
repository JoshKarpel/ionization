import logging
import os
import functools
import collections

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

time_field_unit = atomic_electric_field * atomic_time


@si.utils.memoize
def t1(b):
    # return (1 - (2 * b)) ** 2
    return (1 - b) ** 4


def t2(pulse_delay, b, kernel_func):
    # return (b ** 2) * (1 - (b ** 2)) * (np.abs(kernel_func(pulse_delay)) ** 2)
    return (b ** 2) * ((1 - b) ** 2) * (np.abs(kernel_func(2 * pulse_delay)) ** 2)


def t3(pulse_delay, b, kernel_func, tau_alpha):
    # prefactor = b * (1 - b) * (1 - (b ** 2))
    # return prefactor * 2 * np.real(np.exp(1j * pulse_delay / tau_alpha) * kernel_func(pulse_delay))
    prefactor = b * ((1 - b) ** 3)
    return prefactor * 2 * np.real(np.exp(-1j * pulse_delay / tau_alpha) * kernel_func(2 * pulse_delay))


def a_alpha(pulse_delay, *, b, tau_alpha, kernel_func):
    # t1 = (1 - (2 * b)) ** 2  # isolated pulses
    # t2 = (b ** 2) * (1 - (b ** 2)) * (np.abs(kernel_func(pulse_delay)) ** 2)  # first pulse at second time
    # t3 = 2 * (1 - (2 * b)) * b * (1 - b) * kernel_func(pulse_delay) * np.exp(-1j * pulse_delay / tau_alpha)

    # return t1 + t2 + t3

    return t1(b) + t2(pulse_delay, b, kernel_func) + t3(pulse_delay, b, kernel_func, tau_alpha)


def compare_cosine_and_sine(cosine_product, sine_product):
    test_width = bohr_radius
    test_mass = electron_mass
    test_charge = electron_charge
    # efield_tau_product = 1200 * atomic_electric_field * atomic_time
    # efield = .11 * atomic_electric_field
    # stronger_efield = 1.25 * efield

    # cosine_product = 1 * atomic_electric_field * 10 * asec
    # sine_product = 1.25 * atomic_electric_field * 10 * asec

    # classical_orbit_time = twopi * bohr_radius / (alpha * c)
    # print('orbit time', uround(classical_orbit_time, asec))

    tau_alpha = ide.gaussian_tau_alpha_LEN(test_width, test_mass)
    kernel = functools.partial(ide.gaussian_kernel_LEN, tau_alpha = tau_alpha)

    b_sine = np.sqrt(pi) * ((test_width * test_charge * sine_product / hbar) ** 2)
    b_cosine = np.sqrt(pi) * ((test_width * test_charge * cosine_product / hbar) ** 2)

    print(b_sine, b_cosine)

    pulse_delays = np.linspace(0, 500 * asec, 1e3)
    double_kick = functools.partial(a_alpha, b = b_sine, tau_alpha = tau_alpha, kernel_func = kernel)

    ex = np.exp(-1j * pulse_delays / tau_alpha)
    kern = kernel(2 * pulse_delays)
    si.vis.xy_plot(
        'inteference',
        pulse_delays,
        np.real(ex),
        np.imag(ex),
        np.real(kern),
        np.imag(kern),
        2 * np.real(ex + kern),
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
        f'double_kick__cos={uround(cosine_product, time_field_unit)}__sin={uround(sine_product, time_field_unit)}',
        2 * pulse_delays,
        double_kick(pulse_delays),
        np.ones_like(pulse_delays) * t1(b_sine),
        t2(pulse_delays, b_sine, kernel),
        t3(pulse_delays, b_sine, kernel, tau_alpha),
        np.ones_like(pulse_delays) * ((1 - b_cosine) ** 2),
        line_labels = [
            r'Double Kick',
            r'Isolated Pulses',
            r'First pulse seen from second',
            r'Interference',
            r'Single Strong Kick',
        ],
        legend_on_right = True,
        x_label = 'Pulse Delay', x_unit = 'asec',
        vlines = [tau_alpha], vline_kwargs = [{'linestyle': ':'}],
        # x_extra_ticks = [tau_alpha, classical_orbit_time],
        # x_extra_tick_labels = [r'$ \tau_{\alpha} $', r'$ \tau_{\mathrm{orb}} $'],
        **PLT_KWARGS,
    )


kick = collections.namedtuple('kick', ['time', 'time_field_product'])


def decompose_pulse_into_kicks(electric_potential, times):
    efield_vs_time = electric_potential.get_electric_field_amplitude(times)
    signs = np.sign(efield_vs_time)

    # state machine
    kicks = []
    current_sign = signs[0]
    efield_accumulator = 0
    start_time = times[0]
    prev_time = times[0]
    for efield, sign, time in zip(efield_vs_time, signs, times):
        if sign == current_sign:
            # efield_accumulator += (efield ** 2) * (time - prev_time)
            efield_accumulator += efield * (time - prev_time)
        else:
            # time_diff = time - start_time
            kick_time = (time + start_time) / 2

            kicks.append(kick(time = kick_time, time_field_product = efield_accumulator))
            # kicks.append(kick(time = kick_time, time_field_product = current_sign * np.sqrt(efield_accumulator)))

            # reset
            current_sign = sign
            start_time = time
            efield_accumulator = 0
        prev_time = time

    return kicks


def decompose_sinc(sinc, times):
    kicks = decompose_pulse_into_kicks(sinc, times)

    for k in kicks:
        print(uround(k.time, asec), uround(k.time_field_product, time_field_unit))

    si.vis.xy_plot(
        f'pulse__{sinc}',
        times,
        sinc.get_electric_field_amplitude(times),
        x_label = '$t$', x_unit = 'asec',
        y_label = r'$ \mathcal{E}(t) $', y_unit = 'atomic_electric_field',
        **PLT_KWARGS,
    )

    kick_times = [k.time for k in kicks]
    kick_products = [k.time_field_product for k in kicks]

    si.vis.xy_plot(
        f'pulse__{sinc}__decomposed',
        kick_times,
        kick_products,
        line_kwargs = [{'linestyle': ':', 'marker': 'o'}],
        x_label = 'Kick Times', x_unit = 'asec',
        y_label = r'Amplitudes (au * au)', y_unit = time_field_unit,
        **PLT_KWARGS,
    )


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization') as logger:
        decompose_sinc(ion.SincPulse(pulse_width = 200 * asec, fluence = .2 * Jcm2, phase = 0), np.linspace(-1000 * asec, 1000 * asec, 1e4))
        decompose_sinc(ion.SincPulse(pulse_width = 200 * asec, fluence = .2 * Jcm2, phase = pi / 2), np.linspace(-1000 * asec, 1000 * asec, 1e4))
        compare_cosine_and_sine(cosine_product = .496 * time_field_unit, sine_product = .372 * time_field_unit)
        compare_cosine_and_sine(cosine_product = .702 * time_field_unit, sine_product = .526 * time_field_unit)

        # times = np.linspace(-1000 * asec, 1000 * asec, 1e5)
        # sinc = ion.SincPulse(pulse_width = 200 * asec, fluence = .1 * Jcm2, phase = 0)
        # field = sinc.get_electric_field_amplitude(times)
