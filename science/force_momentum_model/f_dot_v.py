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


def plot_f_and_v(pulse, times):
    efield = pulse.get_electric_field_amplitude(times)
    afield = pulse.get_vector_potential_amplitude_numeric_cumulative(times)

    force = CHARGE * efield
    velocity = -CHARGE * afield / MASS
    power = force * velocity
    energy = integ.cumtrapz(y = power, x = times, initial = 0)

    # https://stackoverflow.com/questions/12709853/python-running-cumulative-sum-with-a-given-window
    dt = times[1] - times[0]
    window = int(150 * u.asec / dt)
    energy_lookback = np.cumsum(power)
    energy_lookback[window:] = energy_lookback[window:] - energy_lookback[:-window]
    energy_lookback *= dt

    si.vis.xy_plot(
        pulse_identifier(pulse[0]),
        times,
        # force / u.atomic_force,
        # velocity / u.atomic_velocity,
        power / u.atomic_power,
        energy / u.hartree,
        energy_lookback / u.hartree,
        line_labels = [
            # r'$ F(t) $',
            # r'$ v(t) $',
            r'$ F(t) \cdot v(t) $',
            r"$ \int_{-\infty}^t F(t') \cdot v(t') \, dt' $",
            r"$ \int_{-T_{\mathrm{cl}}}^t F(t') \cdot v(t') \, dt' $",
        ],
        x_unit = 'asec',
        x_label = r'$ t $',
        legend_on_right = True,
        title = pulse_identifier(pulse[0]),
        **PLOT_KWARGS,
    )


if __name__ == '__main__':
    with LOGMAN as logger:
        pulses = [
            ion.potentials.SincPulse(phase = 0),
            ion.potentials.SincPulse(phase = u.pi / 2),
        ]
        for pulse in pulses:
            times = np.linspace(-35 * pulse.pulse_width, 35 * pulse.pulse_width, 10000)
            pulse = ion.potentials.DC_correct_electric_potential(pulse, times)
            print(pulse.info())
            plot_f_and_v(pulse, times)

        pulses = [
            ion.potentials.GaussianPulse.from_number_of_cycles(phase = 0, number_of_cycles = 3),
            ion.potentials.GaussianPulse.from_number_of_cycles(phase = u.pi / 2, number_of_cycles = 3),
        ]
        for pulse in pulses:
            times = np.linspace(-5 * pulse.pulse_width, 5 * pulse.pulse_width, 10000)
            pulse = ion.potentials.DC_correct_electric_potential(pulse, times)
            print(pulse.info())
            plot_f_and_v(pulse, times)
