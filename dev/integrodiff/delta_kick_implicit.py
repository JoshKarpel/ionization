import logging
import os

import numpy as np
import scipy.linalg as linalg

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.integrodiff as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def run_hyd_ide_sim(pulse):
    sim = ide.IntegroDifferentialEquationSpecification(
        'idesim',
        electric_potential = pulse,
        kernel = ide.hydrogen_kernel_LEN,
        prefactor = ide.hydrogen_prefactor_LEN(electron_charge),
        time_initial = -pulse.pulse_width * 5,
        time_final = pulse.pulse_width * 5,
        time_step = .5 * asec,
    ).to_simulation()

    sim.run_simulation()

    return sim


def solve_ide_implicit(pulse):
    times = np.linspace(-pulse.pulse_width * 5, pulse.pulse_width * 5, 1e5)

    kicks = ide.decompose_potential_into_kicks__amplitude(pulse, times)[1:-1]  # slice off small kicks
    A = np.zeros((len(kicks), len(kicks)), dtype = np.complex128)

    omega = ion.HydrogenBoundState(1, 0).energy / hbar
    prefactor = ide.hydrogen_prefactor_LEN(electron_charge)
    kernel = lambda td: ide.hydrogen_kernel_LEN(td) * np.exp(1j * omega * td)

    def element(n, m):
        print(n, m)
        rv = prefactor * kicks[n].amplitude * kicks[m].amplitude * kernel(kicks[n].time - kicks[m].time)
        if m == n:
            rv -= 1
        elif m == n - 1:  # first subdiagonal
            rv += 1

        return rv

    for n in range(len(kicks)):
        for m in range(n + 1):
            A[n, m] = element(n, m)
    # print(A)
    # print()

    b = np.zeros(len(kicks))
    b[0] = 1

    a = linalg.solve(A, b, lower = True)

    return a, kicks


def compare_ide_to_matrix(pulse):
    times = np.linspace(-pulse.pulse_width * 5, pulse.pulse_width * 5, 1e5)

    a, kicks = solve_ide_implicit(pulse)
    sim = run_hyd_ide_sim(pulse)

    fm = si.vis.xy_plot(
        'pulse_decomposition',
        times,
        pulse.get_electric_field_amplitude(times),
        line_kwargs = [{'color': ion.COLOR_ELECTRIC_FIELD}],
        vlines = [
            k.time for k in kicks
        ],
        vline_kwargs = [{'linestyle': ':', 'linewidth': 1, 'alpha': 0.5} for _ in kicks],
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
        sim.a2,
        color = 'black',
    )
    ax_a2.set_ylabel(r'$ \left| a(t) \right|^2 $', fontsize = 16)

    ax_field.set_xlim(times[0] / asec, times[-1] / asec)

    fm.save()
    fm.cleanup()


if __name__ == '__main__':
    with LOGMAN as logger:
        pulse = ion.GaussianPulse.from_number_of_cycles(pulse_width = 50 * asec, fluence = .1 * Jcm2, phase = pi / 2, number_of_cycles = 2)
        compare_ide_to_matrix(pulse)
