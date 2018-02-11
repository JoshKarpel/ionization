import logging
import os

import numpy as np
import scipy.linalg as linalg

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def run_hyd_ide_sim(pulse, tb):
    sim = ide.IntegroDifferentialEquationSpecification(
        'idesim',
        electric_potential = pulse,
        kernel = ide.hydrogen_kernel_LEN,
        integral_prefactor = -(electron_charge / hbar) ** 2,
        time_initial = -pulse.pulse_width * tb,
        time_final = pulse.pulse_width * tb,
        time_step = 1 * asec,
    ).to_simulation()

    sim.run_simulation()

    return sim


def run_dk_sim(pulse, tb):
    sim = ide.DeltaKickSpecification(
        'dksim',
        electric_potential = pulse,
        kernel = ide.hydrogen_kernel_LEN,
        integral_prefactor = -(electron_charge / hbar) ** 2,
        time_initial = -pulse.pulse_width * tb,
        time_final = pulse.pulse_width * tb,
    ).to_simulation()

    sim.run()

    return sim


def solve_ide_implicit_from_pulse(pulse, tb):
    times = np.linspace(-pulse.pulse_width * tb, pulse.pulse_width * tb, 1e5)

    kicks = ide.decompose_potential_into_kicks(pulse, times)

    return solve_ide_implicit_from_kicks(kicks)


def solve_ide_implicit_from_kicks(kicks):
    A = np.zeros((len(kicks), len(kicks)), dtype = np.complex128)

    omega = ion.HydrogenBoundState(1, 0).energy / hbar
    prefactor = -(electron_charge / hbar) ** 2
    kernel = lambda td: ide.hydrogen_kernel_LEN(td, omega_b = ion.HydrogenBoundState(1, 0).energy / hbar)

    def element(n, m):
        rv = prefactor * kicks[n].amplitude * kicks[m].amplitude * kernel(kicks[n].time - kicks[m].time)
        if m == n:
            rv -= 1
        elif m == n - 1:  # first subdiagonal
            rv += 1

        return rv

    for n in range(len(kicks)):
        for m in range(n + 1):
            A[n, m] = element(n, m)

    b = np.zeros(len(kicks))
    b[0] = 1

    a = linalg.solve_triangular(A, b, lower = True)

    return a, kicks


def compare_ide_to_matrix(pulse, tb):
    times = np.linspace(-pulse.pulse_width * tb, pulse.pulse_width * tb, 1e5)

    a, kicks = solve_ide_implicit_from_pulse(pulse, tb)
    sim = run_hyd_ide_sim(pulse, tb)
    sim_dk = run_dk_sim(pulse, tb)

    fm = si.vis.xy_plot(
        'pulse_decomposition',
        times,
        pulse.get_electric_field_amplitude(times),
        line_kwargs = [{'color': ion.COLOR_ELECTRIC_FIELD}],
        vlines = [
            k.time for k in kicks
        ],
        vline_kwargs = [{'linestyle': '--', 'linewidth': .5, 'alpha': 0.5} for _ in kicks],
        x_unit = 'asec',
        x_label = r'$ t $',
        y_unit = 'atomic_electric_field',
        y_label = r'$ \mathcal{E}(t) $',
        title = 'Delta-Kick IDE Solution Comparison',
        save_on_exit = False,
        close_after_exit = False,
        **PLOT_KWARGS,
    )

    fig = fm.fig

    ax_field = fig.axes[0]

    kt = np.repeat([k.time for k in kicks], 2)[1:]
    kt[0] = times[0]
    kt[-1] = times[-1]

    ax_a2 = ax_field.twinx()
    ax_a2.plot(
        kt / asec,
        np.repeat(np.abs(a) ** 2, 2)[:-1],
        color = 'black',
        linestyle = '--',
    )
    ax_a2.plot(
        sim.times / asec,
        sim.b2,
        color = 'black',
    )
    ax_a2.plot(
        sim_dk.data_times / asec,
        sim_dk.b2,
        color = 'purple',
        linestyle = ':',
        marker = 'o',
    )
    ax_a2.set_ylabel(r'$ \left| a(t) \right|^2 $', fontsize = 16)

    ax_field.set_xlim(times[0] / asec, times[-1] / asec)

    fm.save()
    fm.cleanup()

    sim_dk.plot_b2_vs_time(y_lower_limit = .95, **PLOT_KWARGS)


def pw_scan(fluence):
    pulse_widths = np.linspace(1, 800, 100) * asec
    ceps = [0, pi / 4, pi / 2]
    tb = 32

    cep_to_pulses = {cep: [ion.SincPulse(pulse_width = pw, fluence = fluence, phase = cep,
                                         window = ion.SymmetricExponentialTimeWindow(window_time = pw * (tb - 2), window_width = .2 * pw))
                           for pw in pulse_widths]
                     for cep in ceps}
    cep_to_solns = {cep: [solve_ide_implicit_from_pulse(pulse, tb) for pulse in pulses] for cep, pulses in cep_to_pulses.items()}

    si.vis.xy_plot(
        f'pulse_width_scan__flu={uround(fluence, Jcm2)}jcm2',
        pulse_widths,
        *[[np.abs(soln[0][-1]) ** 2 for soln in solns] for cep, solns in cep_to_solns.items()],
        line_labels = [rf'$\varphi = {uround(cep, pi)}\pi$' for cep in ceps],
        x_label = r'$ \tau $',
        x_unit = 'asec',
        y_label = r'$ \left| a(t_f) \right|^2 $',
        title = rf'Delta-Kick Model Pulse Width Scan, $H = {uround(fluence, Jcm2)} \, \mathrm{{J/cm^2}}$',
        x_lower_limit = 0,
        **PLOT_KWARGS,
    )


def kick_delay_scan(eta = .1 * atomic_time * atomic_electric_field):
    delays = np.linspace(1, 1000, 100) * asec

    kickss = [(ide.delta_kick(0, eta), ide.delta_kick(delay, -eta)) for delay in delays]
    solns = [solve_ide_implicit_from_kicks(kicks) for kicks in kickss]

    alpha = ((electron_charge / hbar) ** 2) * (eta ** 2)
    omega = ion.HydrogenBoundState(1, 0).energy / hbar
    kernel = lambda td: ide.hydrogen_kernel_LEN(td) * np.exp(1j * omega * td)

    analytic = (1 + (alpha * kernel(delays))) / ((1 + (alpha * kernel(0))) ** 2)

    ime_limit = np.abs(solns[-1][0][-1]) ** 2
    ana_limit = np.abs(1 / ((1 + (alpha * kernel(0))) ** 2)) ** 2

    print(ime_limit)
    print(ana_limit)
    print()

    si.vis.xy_plot(
        f'kick_delay_scan__eta={uround(eta, atomic_time * atomic_electric_field)}au',
        delays,
        [np.abs(soln[0][-1]) ** 2 for soln in solns],
        np.abs(analytic) ** 2,
        line_labels = ['IME', 'Analytic'],
        line_kwargs = [None, {'linestyle': '--'}],
        x_label = r'Kick Delay $ \Delta $',
        x_unit = 'asec',
        y_label = r'$ \left| a(t_f) \right|^2 $',
        title = f'Delta-Kick Model Kick Delay Scan, $\eta = {uround(eta, atomic_time * atomic_electric_field)} \, \mathrm{{a.u.}}$',
        vlines = [
            93 * asec,
            150 * asec,
        ],
        hlines = [
            ime_limit,
            ana_limit,
        ],
        x_lower_limit = 0,
        **PLOT_KWARGS,
    )


if __name__ == '__main__':
    with LOGMAN as logger:
        pulse = ion.GaussianPulse.from_number_of_cycles(pulse_width = 50 * asec, fluence = .1 * Jcm2, phase = 0, number_of_cycles = 2)
        # pulse = ion.SincPulse(pulse_width = 50 * asec, fluence = .1 * Jcm2, phase = pi / 2)
        compare_ide_to_matrix(pulse, tb = 4)

        # etas = np.array([.01, .05, .1, .2, .3, .4, .5,]) * atomic_time * atomic_electric_field
        # etas = np.geomspace(.01, 2, 10) * atomic_time * atomic_electric_field
        # si.utils.multi_map(kick_delay_scan, etas, processes = 3)

        # fluences = np.array([.01, .1, 1, 5, 10]) * Jcm2
        # fluences = np.geomspace(.01, 20, 10) * Jcm2
        # si.utils.multi_map(pw_scan, fluences, processes = 3)
