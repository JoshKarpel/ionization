import logging
import os

from tqdm import tqdm
from mpmath import mpf

import numpy as np
import scipy.integrate as integ

import simulacra as si
from simulacra.units import *
import ionization as ion

# import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, 'simlib')

logman = si.utils.LogManager('simulacra', 'ionization',
                             stdout_level = logging.INFO)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    # img_format = 'png',
    # fig_dpi_scale = 6,
)


def instantaneous_tunneling_rate(electric_field_amplitude, ionization_potential = -rydberg):
    amplitude_scaled = np.abs(electric_field_amplitude / atomic_electric_field)
    potential_scaled = np.abs(ionization_potential / hartree)

    f = amplitude_scaled / ((2 * potential_scaled) ** 1.5)

    return np.where(np.not_equal(electric_field_amplitude, 0), (4 / f) * np.exp(-2 / (3 * f)) / atomic_time, 0)


def averaged_tunneling_rate(electric_field_amplitude, ionization_potential = -rydberg):
    amplitude_scaled = np.abs(electric_field_amplitude / atomic_electric_field)
    potential_scaled = np.abs(ionization_potential / hartree)

    f = amplitude_scaled / ((2 * potential_scaled) ** 1.5)

    return np.where(np.not_equal(electric_field_amplitude, 0), 4 * np.sqrt(3 / (pi * f)) * np.exp(-2 / (3 * f)) / atomic_time, 0)


def evolve(tunneling_rate_vs_time, times):
    wavefunction_remaining = 1
    for ii, tunneling_rate in enumerate(tunneling_rate_vs_time[:-1]):
        # wavefunction_remaining = wavefunction_remaining * (1 - (tunneling_rate * np.abs(times[ii + 1] - times[ii])))
        wavefunction_remaining *= np.exp(-tunneling_rate * np.abs(times[ii + 1] - times[ii]))

    return wavefunction_remaining


def run(pulse, times):
    tunneling_rate_vs_time = instantaneous_tunneling_rate(pulse.get_electric_field_amplitude(times))
    return evolve(tunneling_rate_vs_time, times)


if __name__ == '__main__':
    with logman as logger:
        # for flu in np.array([20, 30]) * Jcm2:
        for flu in np.array([.01, .1, 1, 2, 5, 10, 20, 30]) * Jcm2:
            for pulse_type in (ion.SincPulse, ion.GaussianPulse, ion.SechPulse):
                # pulse_type = ion.GaussianPulse
                pw = 200 * asec
                # flu = flu * Jcm2
                ceps = np.linspace(0, pi, 100)

                if pulse_type == ion.SincPulse:
                    t_bound_in_pw = 30
                else:
                    t_bound_in_pw = 10

                times = np.linspace(-t_bound_in_pw * pw, t_bound_in_pw * pw, 1e4)
                print(f'dt = {uround(np.abs(times[1] - times[0]), asec)}as')

                dummy = ion.SincPulse(pulse_width = pw)

                pulses = []
                for ii, cep in enumerate(ceps):
                    if pulse_type == ion.SincPulse:
                        pulse = pulse_type(pulse_width = pw, fluence = flu, phase = cep)
                    else:
                        pulse = pulse_type(pulse_width = pw, fluence = flu, phase = cep, omega_carrier = dummy.omega_carrier)

                    pulse.window = ion.SymmetricExponentialTimeWindow(window_time = (t_bound_in_pw - 3) * pw, window_width = .2 * pw)
                    pulse = ion.DC_correct_electric_potential(pulse, times)

                    pulses.append(pulse)

                wavefunction_vs_time = {pulse: run(pulse, times) for pulse in tqdm(pulses)}

                si.vis.xy_plot(
                    f'{pulse_type.__name__}__pw={uround(pw, asec)}as_flu={uround(flu, Jcm2)}jcm2',
                    ceps,
                    list(v for v in wavefunction_vs_time.values()),
                    x_label = r'$\varphi$', x_unit = 'rad',
                    y_label = 'Remaining Wavefunction',
                    y_log_axis = True, y_log_pad = 2,
                    **PLOT_KWARGS,
                )

                cosine = list(wavefunction_vs_time.values())[0]
                si.vis.xy_plot(
                    f'{pulse_type.__name__}__rel__pw={uround(pw, asec)}as_flu={uround(flu, Jcm2)}jcm2',
                    ceps,
                    list(v / cosine  for v in wavefunction_vs_time.values()),
                    x_label = r'$\varphi$', x_unit = 'rad',
                    y_label = r'Remaining Wavefunction (Rel. to $\varphi = 0$)',
                    y_log_axis = True, y_log_pad = 2,
                    **PLOT_KWARGS,
                )
