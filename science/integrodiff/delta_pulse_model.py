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

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLOT_KWARGS = dict(
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

    # print(b_sine, b_cosine)

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
        **PLOT_KWARGS,
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
        **PLOT_KWARGS,
    )


kick = collections.namedtuple('kick', ['time', 'time_field_product'])


def decompose_pulse_into_kicks__amplitude(electric_potential, times):
    efield_vs_time = electric_potential.get_electric_field_amplitude(times)
    signs = np.sign(efield_vs_time)

    # state machine
    kicks = []
    current_sign = signs[0]
    efield_accumulator = 0
    start_time = times[0]
    prev_time = times[0]
    max_field = 0
    max_field_time = 0
    for efield, sign, time in zip(efield_vs_time, signs, times):
        if sign == current_sign:
            # efield_accumulator += (efield ** 2) * (time - prev_time)
            efield_accumulator += efield * (time - prev_time)
            if max_field < np.abs(efield):
                max_field = np.abs(efield)
                max_field_time = time
        else:
            # time_diff = time - start_time
            # kick_time = (time + start_time) / 2

            kicks.append(kick(time = max_field_time, time_field_product = efield_accumulator))
            # kicks.append(kick(time = kick_time, time_field_product = current_sign * np.sqrt(efield_accumulator)))

            # reset
            current_sign = sign
            start_time = time
            efield_accumulator = 0
            max_field = 0
        prev_time = time

    return kicks


def decompose_pulse_into_kicks__fluence(electric_potential, times):
    efield_vs_time = electric_potential.get_electric_field_amplitude(times)
    signs = np.sign(efield_vs_time)

    # state machine
    kicks = []
    current_sign = signs[0]
    fluence_accumulator = 0
    start_time = times[0]
    prev_time = times[0]
    last_time = times[-1]
    for efield, sign, time in zip(efield_vs_time, signs, times):
        if sign == current_sign and time != last_time:
            # efield_accumulator += (efield ** 2) * (time - prev_time)
            fluence_accumulator += (np.abs(efield) ** 2) * (time - prev_time)
        else:
            time_diff = time - start_time
            kick_time = (time + start_time) / 2

            kicks.append(kick(time = kick_time, time_field_product = current_sign * np.sqrt(fluence_accumulator * time_diff)))

            # reset
            current_sign = sign
            start_time = time
            fluence_accumulator = 0
        prev_time = time

    return kicks


def plot_pulse_decomposition(pulse, times, selector = 'amplitude'):
    # kicks = locals()[f'decompose_pulse_into_kicks__{selector}'](sinc, times)
    if selector == 'amplitude':
        kicks = decompose_pulse_into_kicks__amplitude(pulse, times)
    elif selector == 'power':
        kicks = decompose_pulse_into_kicks__fluence(pulse, times)

    # for k in kicks:
    #     print(uround(k.time, asec), uround(k.time_field_product, time_field_unit))

    try:
        name = f'pulse__{uround(pulse.pulse_width, asec)}as_{uround(pulse.fluence, Jcm2)}jcm2_{uround(pulse.phase, pi)}pi'
    except AttributeError:
        name = f'sinewave__{uround(pulse.period, asec)}as_{uround(pulse.amplitude, atomic_electric_field)}aef'

    si.vis.xy_plot(
        name,
        times,
        pulse.get_electric_field_amplitude(times),
        x_label = '$t$', x_unit = 'asec',
        y_label = r'$ \mathcal{E}(t) $', y_unit = 'atomic_electric_field',
        **PLOT_KWARGS,
    )

    kick_times = [k.time for k in kicks]
    kick_products = [k.time_field_product for k in kicks]

    si.vis.xy_plot(
        f'{name}__decomposed__{selector}',
        kick_times,
        kick_products,
        line_kwargs = [{'linestyle': ':', 'marker': 'o'}],
        x_label = 'Kick Times', x_unit = 'asec',
        y_label = r'Amplitudes (au * au)', y_unit = time_field_unit,
        y_lower_limit = -1 * time_field_unit, y_upper_limit = 1 * time_field_unit,
        **PLOT_KWARGS,
    )

    return kicks


def recursive_kicks(kicks, *, abs_prefactor, kernel_func, bound_state_frequency):
    abs_prefactor = np.abs(abs_prefactor)
    bound_state_frequency = np.abs(bound_state_frequency)

    @si.utils.memoize
    def time_diff(i, j):
        return kicks[i].time - kicks[j].time

    @si.utils.memoize
    def b(i, j):
        return abs_prefactor * kicks[i].time_field_product * kicks[j].time_field_product

    @si.utils.memoize
    def a(n):
        # print(f'calling a({n})')
        if n < 0:
            return 1
        else:
            first_term = np.exp(-1j * np.abs(bound_state_frequency) * time_diff(n, n - 1)) * a(n - 1) * (1 - b(n, n))
            second_term = sum(a(i) * b(n, i) * kernel_func(time_diff(n, i)) for i in range(n))  # all but current kick
            # print(second_term)

            return first_term - second_term

    return np.array(list(a(i) for i in range(len(kicks))))


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO) as logger:
        # for selector in ['amplitude', 'power']:
        #     decompose_sinc(ion.SincPulse(pulse_width = 200 * asec, fluence = .1 * Jcm2, phase = 0), np.linspace(-1000 * asec, 1000 * asec, 1e4), selector = selector)
        #     decompose_sinc(ion.SincPulse(pulse_width = 200 * asec, fluence = .1 * Jcm2, phase = pi / 2), np.linspace(-1000 * asec, 1000 * asec, 1e4), selector = selector)
        # compare_cosine_and_sine(cosine_product = .496 * time_field_unit, sine_product = .372 * time_field_unit)
        # compare_cosine_and_sine(cosine_product = .702 * time_field_unit, sine_product = .526 * time_field_unit)

        # COMPARE TO FULL IDE
        pulse_width = 200 * asec
        fluence = .2 * Jcm2
        phase = pi / 2

        test_width = 1 * bohr_radius
        test_charge = 1 * electron_charge
        test_mass = 1 * electron_mass
        potential_depth = 36.831335 * eV

        internal_potential = ion.FiniteSquareWell(potential_depth = potential_depth, width = test_width)
        bound_state = ion.FiniteSquareWellState.from_potential(internal_potential, mass = electron_mass)
        # pulse = ion.GaussianPulse(pulse_width = pulse_width, fluence = fluence, phase = phase)
        pulse = ion.SincPulse(pulse_width = pulse_width, fluence = fluence, phase = phase)

        t_bound = 10 * pulse_width

        prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge)
        tau_alpha = ide.gaussian_tau_alpha_LEN(test_width, test_mass)
        kernel = functools.partial(ide.gaussian_kernel_LEN, tau_alpha = tau_alpha)
        omega_alpha = 1 / (2 * tau_alpha)

        sim = ide.IntegroDifferentialEquationSpecification(
            'compare',
            test_width = test_width,
            test_charge = test_charge,
            test_mass = test_mass,
            potential_depth = potential_depth,
            electric_potential = pulse,
            time_initial = -t_bound,
            time_final = t_bound,
            time_step = 1 * asec,
            electric_potential_dc_correction = True,
            prefactor = prefactor,
            kernel = ide.gaussian_kernel_LEN,
            kernel_kwargs = {'tau_alpha': ide.gaussian_tau_alpha_LEN(test_width, test_mass)},
            test_energy = hbar * omega_alpha,
            # test_energy = bound_state.energy,
            evolution_gauge = 'LEN',
            evolution_method = 'RK4',
            # time_step_minimum = 1 * asec,
            # time_step_maximum = 10 * asec,
            # error_on = 'da/dt',
            # epsilon = 1e-6,
            store_data_every = 1,
        ).to_simulation()

        sim.run_simulation()
        sim.plot_wavefunction_vs_time(
            show_vector_potential = False,
            **PLOT_KWARGS
        )

        kicks_amp = decompose_pulse_into_kicks__amplitude(
            sim.spec.electric_potential,
            sim.times
        )
        kick_a_vs_time__amp = recursive_kicks(
            kicks_amp,
            abs_prefactor = np.abs(prefactor),
            kernel_func = kernel,
            bound_state_frequency = omega_alpha
        )

        kicks_flu = decompose_pulse_into_kicks__fluence(
            sim.spec.electric_potential,
            sim.times
        )
        kick_a_vs_time__flu = recursive_kicks(
            kicks_flu,
            abs_prefactor = np.abs(prefactor),
            kernel_func = kernel,
            bound_state_frequency = omega_alpha
        )

        si.vis.xxyy_plot(
            'comparison_to_full_simulation',
            [
                sim.times,
                [k.time for k in kicks_amp],
                [k.time for k in kicks_flu],
            ],
            [
                sim.a2,
                np.abs(kick_a_vs_time__amp) ** 2,
                np.abs(kick_a_vs_time__flu) ** 2,
            ],
            line_labels = ['IDE Simulation', r'$\delta$-kick Model (Amp.)', r'$\delta$-kick Model (Flu.)'],
            x_label = r'$ t $', x_unit = 'asec',
            y_label = 'Bound State Overlap',
            **PLOT_KWARGS
        )


        # vs pulse width plot

        def get_final_a2(pulse, times):
            kicks = decompose_pulse_into_kicks__amplitude(pulse, times)

            r = recursive_kicks(
                kicks,
                abs_prefactor = prefactor,
                kernel_func = kernel,
                bound_state_frequency = omega_alpha
            )

            return np.abs(r[-1]) ** 2


        pulse_widths = np.linspace(50, 2000, 1e3) * asec
        # cosine_pulses = list(ion.GaussianPulse(pulse_width = pw, fluence = fluence, phase = 0) for pw in pulse_widths)
        cosine_pulses = list(ion.SincPulse(pulse_width = pw, fluence = fluence, phase = 0) for pw in pulse_widths)
        # sine_pulses = list(ion.GaussianPulse(pulse_width = pw, fluence = fluence, phase = pi / 2) for pw in pulse_widths)
        sine_pulses = list(ion.SincPulse(pulse_width = pw, fluence = fluence, phase = pi / 2) for pw in pulse_widths)

        cosine_results = np.array(list(get_final_a2(pulse, sim.times) for pulse in tqdm(cosine_pulses)))
        sine_results = np.array(list(get_final_a2(pulse, sim.times) for pulse in tqdm(sine_pulses)))

        si.vis.xy_plot(
            'recursive_vs_pulse_width',
            pulse_widths,
            cosine_results,
            sine_results,
            line_labels = ['cos', 'sin'],
            x_label = r'$ \tau $', x_unit = 'asec',
            y_label = 'Bound State Overlap',
            **PLOT_KWARGS,
        )

        # consistency

        # decomposition = decompose_pulse_into_kicks__amplitude(pulse, sim.times)
        # for k in decomposition:
        #     print(k.time / asec, k.time_field_product / time_field_unit)
        #
        # test_kicks_cosine = [kick(0, .496 * time_field_unit)]
        # test_kicks_sine = [kick(-83.408 * asec, .372 * time_field_unit), kick(83.408 * asec, .372 * time_field_unit)]
        # pulse_delay = 2 * 83.408
        # print(pulse_delay)
        # kicks_a_vs_time_cosine = recursive_kicks(test_kicks_cosine,
        #                                          abs_prefactor = np.abs(prefactor),
        #                                          kernel_func = kernel,
        #                                          bound_state_frequency = omega_alpha)
        # kicks_a_vs_time_sine = recursive_kicks(test_kicks_sine,
        #                                        abs_prefactor = np.abs(prefactor),
        #                                        kernel_func = kernel,
        #                                        bound_state_frequency = omega_alpha)
        # print(np.abs(kicks_a_vs_time_cosine) ** 2)
        # print(np.abs(kicks_a_vs_time_sine) ** 2)
