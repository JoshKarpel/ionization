import logging
import os
import functools
import datetime

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
SIM_LIB = os.path.join(OUT_DIR, 'SIMLIB')

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

CHARGE = u.electron_charge
MASS = u.electron_mass_reduced


def run(spec):
    with LOGMAN as logger:
        sim = si.utils.find_or_init_sim(spec, search_dir = SIM_LIB)

        if not sim.status == si.Status.FINISHED:
            sim.run_simulation()
            sim.save(target_dir = SIM_LIB)

        return sim


def pulse_identifier(pulse):
    return f'{pulse.__class__.__name__}_pw={u.uround(pulse.pulse_width, u.asec)}as_cep={u.uround(pulse.phase, u.pi)}pi_flu={u.uround(pulse.fluence, u.Jcm2)}jcm2'


def get_power(pulse, times, lookback = None):
    force = CHARGE * pulse.get_electric_field_amplitude(times)
    velocity = -CHARGE * pulse.get_vector_potential_amplitude_numeric_cumulative(times) / MASS

    if lookback is not None:
        dt = times[1] - times[0]
        window = int(lookback / dt)

        velocity[window:] = velocity[window:] - velocity[:-window]

    return force * velocity


def plot_power(cos_pulse, sin_pulse, times, lookback = None):
    pulses = [cos_pulse, sin_pulse]

    powers = [get_power(pulse, times, lookback = lookback) for pulse in pulses]

    si.vis.xy_plot(
        'F_AND_V___' + '__'.join(pulse_identifier(pulse[0]) for pulse in pulses) + (f'__{u.uround(lookback, u.asec)}' if lookback is not None else ''),
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


def get_tdse_spec(pulse, times):
    spec = ion.SphericalHarmonicSpecification(
        pulse_identifier(pulse),
        r_bound = 100 * u.bohr_radius,
        r_points = 500,
        l_bound = 200,
        time_initial = times[0],
        time_final = times[-1],
        electric_potential = pulse,
        store_energy_expectation_value = True,
        store_electric_dipole_moment_expectation_value = True,
        use_numeric_eigenstates = True,
        numeric_eigenstate_max_energy = 20 * u.eV,
        numeric_eigenstate_max_angular_momentum = 3,
        electric_potential_dc_correction = True,
        checkpoints = True,
        checkpoint_every = datetime.timedelta(minutes = 1),
        checkpoint_dir = SIM_LIB,
    )

    return spec


def plot_tdse_expectation_values(sims):
    plot_tdse_initial_state_overlap(sims)
    plot_tdse_energy_and_power_expectation_values(sims)
    plot_tdse_z_and_vz_expectation_values(sims)


def plot_tdse_initial_state_overlap(sims):
    si.vis.xxyy_plot(
        'tdse_ionization',
        [s.data_times for s in sims],
        [s.state_overlaps_vs_time[s.spec.initial_state] for s in sims],
        x_label = r'$ t $',
        x_unit = 'asec',
        y_label = r'$ \left\langle \psi_{100}(t) | \Psi(t) \right\rangle $',
        line_labels = [rf'$ \varphi = {u.uround(s.spec.electric_potential[0].phase, u.pi)} \pi $' for s in sims],
        **PLOT_KWARGS
    )

    ionization_rate = [np.gradient(s.state_overlaps_vs_time[s.spec.initial_state], s.data_times) for s in sims]

    si.vis.xxyy_plot(
        'tdse_ionization_rate',
        [s.data_times for s in sims],
        ionization_rate,
        x_label = r'$ t $',
        x_unit = 'asec',
        y_label = r'$ \partial_t \left\langle \psi_{100}(t) | \Psi(t)  \right\rangle $',
        y_unit = 'per_asec',
        line_labels = [rf'$ \varphi = {u.uround(s.spec.electric_potential[0].phase, u.pi)} \pi $' for s in sims],
        **PLOT_KWARGS
    )


def plot_tdse_energy_and_power_expectation_values(sims):
    si.vis.xxyy_plot(
        'tdse_energy',
        [s.data_times for s in sims] * 2,
        [
            *[s.total_energy_expectation_value_vs_time for s in sims],
            *[s.internal_energy_expectation_value_vs_time for s in sims],
        ],
        line_kwargs = [
            {'color': 'C0', 'linestyle': '-'},
            {'color': 'C1', 'linestyle': '-'},
            {'color': 'C0', 'linestyle': '--'},
            {'color': 'C1', 'linestyle': '--'},
        ],
        x_label = r'$ t $',
        x_unit = 'asec',
        y_label = r'$ \left\langle H(t) \right\rangle $',
        y_unit = 'hartree',
        line_labels = [rf'$ \varphi = {u.uround(s.spec.electric_potential[0].phase, u.pi)} \pi $' for s in sims],
        **PLOT_KWARGS
    )

    total_powers = [np.gradient(s.total_energy_expectation_value_vs_time, s.data_times) for s in sims]
    internal_powers = [np.gradient(s.internal_energy_expectation_value_vs_time, s.data_times) for s in sims]

    si.vis.xxyy_plot(
        'tdse_power',
        [s.data_times for s in sims] * 2,
        [
            *total_powers,
            *internal_powers,
        ],
        line_kwargs = [
            {'color': 'C0', 'linestyle': '-'},
            {'color': 'C1', 'linestyle': '-'},
            {'color': 'C0', 'linestyle': '--'},
            {'color': 'C1', 'linestyle': '--'},
        ],
        x_label = r'$ t $',
        x_unit = 'asec',
        y_label = r'$ \partial_t \left\langle H(t) \right\rangle $',
        y_unit = 'atomic_power',
        line_labels = [rf'$ \varphi = {u.uround(s.spec.electric_potential[0].phase, u.pi)} \pi $' for s in sims],
        **PLOT_KWARGS
    )


def plot_tdse_z_and_vz_expectation_values(sims):
    z = [s.electric_dipole_moment_expectation_value_vs_time / s.spec.test_charge for s in sims]

    si.vis.xxyy_plot(
        'tdse_z',
        [s.data_times for s in sims],
        z,
        x_label = r'$ t $',
        x_unit = 'asec',
        y_label = r'$ \left\langle z(t) \right\rangle $',
        y_unit = 'bohr_radius',
        line_labels = [rf'$ \varphi = {u.uround(s.spec.electric_potential[0].phase, u.pi)} \pi $' for s in sims],
        **PLOT_KWARGS
    )

    v_z = [np.gradient(s.electric_dipole_moment_expectation_value_vs_time / s.spec.test_charge, s.data_times) for s in sims]

    si.vis.xxyy_plot(
        'tdse_vz',
        [s.data_times for s in sims],
        v_z,
        x_label = r'$ t $',
        x_unit = 'asec',
        y_label = r'$ \partial_t \left\langle z(t) \right\rangle $',
        y_unit = 'atomic_velocity',
        line_labels = [rf'$ \varphi = {u.uround(s.spec.electric_potential[0].phase, u.pi)} \pi $' for s in sims],
        **PLOT_KWARGS
    )

    force_times_v_z = [sim.spec.test_charge * sim.spec.electric_potential.get_electric_field_amplitude(sim.data_times) * v
                       for v, sim in zip(v_z, sims)]

    si.vis.xxyy_plot(
        'tdse_force_times_vz',
        [s.data_times for s in sims],
        force_times_v_z,
        x_label = r'$ t $',
        x_unit = 'asec',
        y_label = r'$ q \, \mathcal{E}(t) \, \partial_t \left\langle v_z(t) \right\rangle $',
        y_unit = 'atomic_power',
        line_labels = [rf'$ \varphi = {u.uround(s.spec.electric_potential[0].phase, u.pi)} \pi $' for s in sims],
        **PLOT_KWARGS
    )


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
            r'$ \varphi = \pi / 2 $',
            r'$ \varphi = 0 $, w/LB',
            r'$ \varphi = \pi / 2 $, w/LB',
        ],
        line_kwargs = [
            {'color': 'C0', 'linestyle': '--'},
            {'color': 'C1', 'linestyle': '--'},
            {'color': 'C0', 'linestyle': '-'},
            {'color': 'C1', 'linestyle': '-'},
        ],
        x_unit = 'asec',
        x_label = r'$ t $',
        y_label = r'$ \left| b(t) \right|^2 $',
        # legend_on_right = True,
        title = '\n'.join(pulse_identifier(pulse[0]) for pulse in pulses),
        **PLOT_KWARGS
    )


def peak_power_vs_cep():
    pw = 200 * u.asec
    ceps = np.linspace(0, u.pi, 1000)

    pulses = [ion.potentials.SincPulse(pulse_width = pw, phase = cep) for cep in ceps]
    times = np.linspace(-50 * pw, 50 * pw, 20000)

    powers = [get_power(pulse, times) for pulse in pulses]
    peak_power = [np.max(np.abs(power)) for power in powers]

    si.vis.xy_plot(
        'peak_power_vs_cep',
        ceps,
        peak_power,
        x_label = r'$ \varphi $',
        y_label = r'Peak Power',
        x_unit = 'rad',
        y_unit = 'atomic_power',
        **PLOT_KWARGS
    )


def power_vs_cep_movie():
    pw = 200 * u.asec
    ceps = np.linspace(0, u.pi, 1000)

    times = np.linspace(-50 * pw, 50 * pw, 20000)

    def y(times, cep):
        pulse = ion.potentials.SincPulse(pulse_width = pw, phase = cep,)
        return get_power(pulse, times)

    si.vis.xyt_plot(
        'power_vs_cep_movie',
        times,
        ceps,
        y,
        length = 10,
        t_fmt_string = r'$ \varphi = {} $',
        x_label = r'$ t $',
        y_label = r'$ F(t) \cdot v_{\mathrm{cl}}(t) $',
        y_unit = 'atomic_power',
        x_unit = 'asec',
        x_lower_limit = -5 * pw,
        x_upper_limit = 5 * pw,
        progress_bar = True,
        target_dir = OUT_DIR,
        fig_dpi_scale = 5,
    )


if __name__ == '__main__':
    with LOGMAN as logger:
        # peak_power_vs_cep()
        # power_vs_cep_movie()

        pulses = [
            ion.potentials.SincPulse(pulse_width = 20 * u.asec, fluence = 2 * u.Jcm2, phase = 0),
            ion.potentials.SincPulse(pulse_width = 20 * u.asec, fluence = 2 * u.Jcm2, phase = u.pi / 2),
            # ion.potentials.SincPulse(phase = u.pi / 3),
        ]
        window = ion.potentials.SymmetricExponentialTimeWindow(
            window_time = 30 * pulses[0].pulse_width,
            window_width = .2 * pulses[0].pulse_width,
        )
        for pulse in pulses:
            pulse.window = window
        times = np.linspace(-35 * pulses[0].pulse_width, 35 * pulses[0].pulse_width, 10000)

        corrected_pulses = [ion.potentials.DC_correct_electric_potential(pulse, times) for pulse in pulses]
        plot_power(*corrected_pulses, times, lookback = None)
        plot_power(*corrected_pulses, times, lookback = 5 * u.asec)
        plot_power(*corrected_pulses, times, lookback = 20 * u.asec)
        plot_power(*corrected_pulses, times, lookback = 50 * u.asec)
        plot_power(*corrected_pulses, times, lookback = 100 * u.asec)
        plot_power(*corrected_pulses, times, lookback = 150 * u.asec)
        plot_power(*corrected_pulses, times, lookback = 200 * u.asec)
        plot_power(*corrected_pulses, times, lookback = 500 * u.asec)
        # plot_power_model(corrected_pulses, times)

        # tdse_specs = [get_tdse_spec(pulse, times) for pulse in pulses]
        # tdse_sims = si.utils.multi_map(run, tdse_specs, processes = 2)
        # plot_tdse_expectation_values(tdse_sims)

        # pulses = [
        #     ion.potentials.GaussianPulse.from_number_of_cycles(number_of_cycles = 3, phase = 0),
        #     ion.potentials.GaussianPulse.from_number_of_cycles(number_of_cycles = 3, phase = u.pi / 2),
        # ]
        # times = np.linspace(-5 * pulses[0].pulse_width, 5 * pulses[0].pulse_width, 10000)
        # pulses = [ion.potentials.DC_correct_electric_potential(pulse, times) for pulse in pulses]
        # plot_f_and_v(*pulses, times)
        # plot_power_model(pulses, times)
