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

log = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)

PLT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 3,
)


def instantaneous_tunneling_rate(electric_field_amplitude, ionization_potential = -rydberg):
    # f = np.abs(electric_field_amplitude / atomic_electric_field)
    #
    # return (4 / f) * (electron_mass_reduced * (proton_charge ** 4) / (hbar ** 3)) * np.exp(-(2 / 3) / f)

    amplitude_scaled = np.abs(electric_field_amplitude / atomic_electric_field)
    potential_scaled = np.abs(ionization_potential / hartree)

    f = amplitude_scaled / ((2 * potential_scaled) ** 1.5)

    return (4 / f) * np.exp(-2 / (3 * f)) / atomic_time

    # e_a = (electron_mass_reduced ** 2) * (proton_charge ** 5) / (((4 * pi * epsilon_0) ** 3) * (hbar ** 4))
    # w_a = (electron_mass_reduced * (proton_charge ** 4)) / (((4 * pi * epsilon_0) ** 2) * (hbar ** 3))
    # f = e_a / np.abs(electric_field_amplitude)

    # return 4 * w_a * f * np.exp(-2 * f / 3)


def tunneling_rate_plot(electric_field_amplitude_min, electric_field_amplitude_max, ionization_potential = -rydberg):
    amplitudes = np.linspace(electric_field_amplitude_min, electric_field_amplitude_max, 1e4)
    tunneling_rates = instantaneous_tunneling_rate(amplitudes, ionization_potential)

    si.vis.xy_plot(
        f'tunneling_rate_vs_amplitude__amp={uround(electric_field_amplitude_min, atomic_electric_field)}aef_to_amp={uround(electric_field_amplitude_max, atomic_electric_field)}aef',
        amplitudes,
        tunneling_rates * asec,
        x_label = fr'Electric Field Amplitude ${ion.LATEX_EFIELD}$', x_unit = 'atomic_electric_field',
        y_label = r'Tunneling Rate ($\mathrm{as}^{-1}$)',
        title = 'Tunneling Rate in a Static Electric Field',
        **PLT_KWARGS,
    )


if __name__ == '__main__':
    with log as logger:
        tunneling_rate_plot(0 * atomic_electric_field, 10 * atomic_electric_field)
        tunneling_rate_plot(0 * atomic_electric_field, 5 * atomic_electric_field)
        tunneling_rate_plot(0 * atomic_electric_field, 1 * atomic_electric_field)
        tunneling_rate_plot(0 * atomic_electric_field, .2 * atomic_electric_field)

        #############################################

        # pw = 100 * asec
        # t_bound = 10 * pw
        # flu = .01 * Jcm2
        # efield = ion.SincPulse(pulse_width = pw, fluence = flu,
        #                       window = ion.SymmetricExponentialTimeWindow(window_time = (t_bound - (2 * pw)), window_width = .2 * pw))

        # amp = 0.03 * atomic_electric_field
        # t_bound = 20 * fsec
        # frac = .6
        #
        # title = f'rect_amp={uround(amp, atomic_electric_field)}aef_tb={uround(t_bound, asec)}_frac={frac}'
        # efield = ion.Rectangle(start_time = -t_bound, end_time = t_bound, amplitude = amp,
        #                        window = ion.SymmetricExponentialTimeWindow(window_time = frac * t_bound, window_width = .05 * t_bound))

        energy = 1.0 * eV
        amp = .1 * atomic_electric_field
        frac = 0.7
        bound_mult = 3
        efield = ion.SineWave.from_photon_energy(energy, amplitude = amp)
        t_bound = bound_mult * efield.period
        efield.window = window = ion.SymmetricExponentialTimeWindow(window_time = frac * t_bound, window_width = .05 * t_bound)
        title = f'sine_energy={uround(energy, eV)}eV_amp={uround(amp, atomic_electric_field)}aef_tb={bound_mult}pw_frac={frac}'

        r_bound = 200
        dt = 2 * asec

        sim = ion.SphericalHarmonicSpecification(
            title + f'__tdse__dt={uround(dt, asec)}as',
            r_bound = r_bound * bohr_radius,
            r_points = r_bound * 4,
            # evolution_gauge = 'VEL', l_bound = 200,
            evolution_gauge = 'LEN', l_bound = 200,
            time_initial = -t_bound, time_final = t_bound, time_step = dt,
            electric_potential = efield,
            # electric_potential_dc_correction = True,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 50 * eV,
            numeric_eigenstate_max_angular_momentum = 10,
            mask = ion.RadialCosineMask(inner_radius = .8 * r_bound * bohr_radius, outer_radius = r_bound * bohr_radius),
            store_data_every = 50,
        ).to_simulation()

        logger.info(sim.info())
        sim.run_simulation(progress_bar = True)
        logger.info(sim.info())

        sim.plot_wavefunction_vs_time(
            show_vector_potential = False,
            **PLT_KWARGS
        )

        pot = sim.spec.electric_potential
        times = np.linspace(-t_bound, t_bound, 1e3)

        tunneling_rate_vs_time = instantaneous_tunneling_rate(pot.get_electric_field_amplitude(times), -rydberg)

        wavefunction_remaining = np.empty_like(times)
        wavefunction_remaining[0] = 1
        for ii, tunneling_rate in enumerate(tunneling_rate_vs_time[:-1]):
            wavefunction_remaining[ii + 1] = wavefunction_remaining[ii] * (1 - (tunneling_rate * np.abs(times[ii + 1] - times[ii])))

        si.vis.xy_plot(
            title + '__tunneling_rate_vs_time',
            times,
            tunneling_rate_vs_time * asec,
            x_label = r'Time $t$', x_unit = 'asec',
            y_label = r'Tunneling Rate ($\mathrm{as}^{-1}$)',
            **PLT_KWARGS,
        )

        si.vis.xxyy_plot(
            title + '__comparison',
            (
                sim.data_times,
                sim.data_times,
                sim.data_times,
                times,
            ),
            (
                sim.norm_vs_time,
                sim.total_bound_state_overlap_vs_time,
                sim.state_overlaps_vs_time[sim.spec.initial_state],
                wavefunction_remaining,
            ),
            line_labels = ['TDSE Norm', 'TDSE All Bound States', 'TDSE Initial State', 'Tunneling Ionization'],
            x_label = r'Time $t$', x_unit = 'asec',
            y_label = 'Wavefunction Remaining',
            # y_lower_limit = 0, y_upper_limit = 1, y_pad = 0,
            # y_log_axis = True, y_lower_limit = .98, y_upper_limit = 1, y_log_pad = 1,
            **PLT_KWARGS,
        )
