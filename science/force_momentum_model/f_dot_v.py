import logging
import os
import functools

import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp

import simulacra as si
import simulacra.units as u

import ionization as ion
import ionization.ide as ide

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

CHARGE = u.electron_charge
MASS = u.electron_mass_reduced


def pulse_identifier(pulse):
    return f'{pulse.__class__.__name__}_pw={u.uround(pulse.pulse_width, u.asec)}as_cep={u.uround(pulse.phase, u.pi)}pi'


def get_power(pulse, times, lookback = None):
    force = CHARGE * pulse.get_electric_field_amplitude(times)
    velocity = -CHARGE * pulse.get_vector_potential_amplitude_numeric_cumulative(times) / MASS

    if lookback is not None:
        dt = times[1] - times[0]
        window = int(lookback / dt)

        velocity[window:] = velocity[window:] - velocity[:-window]

    return force * velocity


def plot_f_and_v(cos_pulse, sin_pulse, times):
    pulses = [cos_pulse, sin_pulse]

    powers = [get_power(pulse, times) for pulse in pulses]

    si.vis.xy_plot(
        'F_AND_V___' + '__'.join(pulse_identifier(pulse[0]) for pulse in pulses),
        times,
        *[power / u.atomic_power for power in powers],
        line_labels = [
            r'$ F(t) \cdot v(t), \, \varphi = 0$',
            r'$ F(t) \cdot v(t), \, \varphi = \pi / 2$',
        ],
        x_unit = 'asec',
        x_label = r'$ t $',
        # legend_on_right = True,
        title = '\n'.join(pulse_identifier(pulse[0]) for pulse in pulses),
        **PLOT_KWARGS,
    )


def solve_power_model(pulse, times, lookback = None):
    power = get_power(pulse, times, lookback = lookback)

    rate = -power / u.hartree

    rate_integral = integ.cumtrapz(rate, times, initial = 0)

    return np.exp(rate_integral)


def plot_power_model(pulses, times):
    bs = [solve_power_model(pulse, times) for pulse in pulses]
    abs_b_squared = [np.abs(b) ** 2 for b in bs]

    bs_with_window = [solve_power_model(pulse, times, lookback = 150 * u.asec) for pulse in pulses]
    abs_b_squared_with_lookback = [np.abs(b) ** 2 for b in bs_with_window]

    si.vis.xy_plot(
        'B___' + '__'.join(pulse_identifier(pulse[0]) for pulse in pulses),
        times,
        *abs_b_squared,
        *abs_b_squared_with_lookback,
        line_labels = [
            r'$ \varphi = 0 $',
            r'$ \varphi = 0 $, w/LB',
            r'$ \varphi = \pi / 2 $',
            r'$ \varphi = \pi / 2 $, w/LB',
        ],
        line_kwargs = [
            {'color': 'C0', 'linestyle': '--'},
            {'color': 'C0', 'linestyle': '-'},
            {'color': 'C1', 'linestyle': '--'},
            {'color': 'C1', 'linestyle': '-'},
        ],
        x_unit = 'asec',
        x_label = r'$ t $',
        y_label = r'$ \left| b(t) \right|^2 $',
        # legend_on_right = True,
        title = '\n'.join(pulse_identifier(pulse[0]) for pulse in pulses),
        **PLOT_KWARGS
    )


if __name__ == '__main__':
    with LOGMAN as logger:
        pulses = [
            ion.potentials.SincPulse(phase = 0),
            ion.potentials.SincPulse(phase = u.pi / 2),
        ]
        times = np.linspace(-20 * pulses[0].pulse_width, 20 * pulses[0].pulse_width, 10000)
        pulses = [ion.potentials.DC_correct_electric_potential(pulse, times) for pulse in pulses]
        plot_f_and_v(*pulses, times)
        plot_power_model(pulses, times)

        pulses = [
            ion.potentials.GaussianPulse.from_number_of_cycles(number_of_cycles = 3, phase = 0),
            ion.potentials.GaussianPulse.from_number_of_cycles(number_of_cycles = 3, phase = u.pi / 2),
        ]
        times = np.linspace(-5 * pulses[0].pulse_width, 5 * pulses[0].pulse_width, 10000)
        pulses = [ion.potentials.DC_correct_electric_potential(pulse, times) for pulse in pulses]
        plot_f_and_v(*pulses, times)
        plot_power_model(pulses, times)
