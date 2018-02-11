import logging
import os
import functools

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.ide as ide

from idesolver import CIDESolver

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', 'idesolver', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def make_comparison_plot(identifier, rk4solvers, idesolvers):
    labels = [
        *[fr'RK4 $\Delta t = {uround(solver.time_step, asec)} \, \mathrm{{as}}$ ({solver.elapsed_time.total_seconds()} s)' for solver in rk4solvers],
        *[fr'IDE $\epsilon = {solver.global_error_tolerance}$ ({solver.elapsed_time.total_seconds()} s)' for solver in idesolvers],
    ]
    line_kwargs = [
        *[{} for solver in rk4solvers],
        *[{'linestyle': '--'} for solver in idesolvers]
    ]

    si.vis.xxyy_plot(
        f'{identifier}_comparison',
        [
            *[solver.times for solver in rk4solvers],
            *[solver.x for solver in idesolvers],
        ],
        [
            *[solver.b2 for solver in rk4solvers],
            *[np.abs(solver.y) ** 2 for solver in idesolvers],
        ],
        line_labels = labels,
        line_kwargs = line_kwargs,
        x_unit = 'asec',
        x_label = r'$t$',
        y_label = r'$\left| a(t) \right|^2$',
        title = identifier,
        legend_on_right = True,
        **PLOT_KWARGS,
    )


def run(solver):
    with LOGMAN as logger:
        if isinstance(solver, CIDESolver):
            with si.utils.BlockTimer() as timer:
                solver.solve()
            solver.elapsed_time = timer.wall_time_elapsed
            return solver
        else:
            sim = solver.to_simulation()

            logger.debug(sim.info())
            sim.run()
            logger.debug(sim.info())

            # sim.plot_a2_vs_time(**PLOT_KWARGS)

            return sim


def rk4(pulse, tb = 3, dts = (1 * asec, .1 * asec), processes = 2):
    specs = [
        ide.IntegroDifferentialEquationSpecification(
            f'rk4solver_dt={uround(dt, asec)}as__pw={uround(pulse.pulse_width, asec)}as',
            time_initial = -pulse.pulse_width * tb,
            time_final = pulse.pulse_width * tb,
            time_step = dt,
            electric_potential = pulse,
            electric_potential_dc_correction = False,
            kernel = ide.hydrogen_kernel_LEN,
            integral_prefactor = ide.hydrogen_prefactor_LEN(electron_charge),
        ) for dt in dts]

    return si.utils.multi_map(run, specs, processes = processes)


def c(x, y, omega = None):
    return -1j * omega * y


def d(x, pulse = None):
    return -((electron_charge / hbar) ** 2) * pulse.get_electric_field_amplitude(x)


def k(x, s, pulse = None):
    return pulse.get_electric_field_amplitude(s) * ide.hydrogen_kernel_LEN(x - s)


def f(y):
    return y


def lower(x, t = None):
    return t[0]


def upper(x):
    return x


def iterative(pulse, tb = 3, t_pts = 200, tols = (1e-3, 1e-6), maxiter = 100, processes = 2):
    omega = ion.HydrogenBoundState(1, 0).energy / hbar
    t = np.linspace(-pulse.pulse_width * tb, pulse.pulse_width * tb, t_pts)

    solvers = [CIDESolver(
        y_initial = 1,
        x = t,
        c = functools.partial(c, omega = omega),
        d = functools.partial(d, pulse = pulse),
        k = functools.partial(k, pulse = pulse),
        lower_bound = functools.partial(lower, t = t),
        upper_bound = upper,
        f = f,
        global_error_tolerance = tol,
        max_iterations = maxiter,
    ) for tol in tols]

    return si.utils.multi_map(run, solvers, processes = processes)


if __name__ == '__main__':
    with LOGMAN as logger:
        for pw in [50, 100, 150, 200, 250, 300, 350, 400]:
            pw = pw * asec
            flu = 1 * Jcm2
            phase = 0

            pulse = ion.GaussianPulse.from_number_of_cycles(pulse_width = pw, fluence = flu, phase = phase)

            tb = 3
            dts = [5 * asec, 1 * asec, .5 * asec, .1 * asec]
            tols = [1e-4, 1e-5, 1e-6]
            maxiter = 400
            pts = 100

            ident = f'pw={uround(pw, asec)}as__pts={pts}'

            rk4solvers = rk4(pulse, tb = tb, dts = dts)
            idesolvers = iterative(pulse, tb = tb, t_pts = pts, tols = tols, maxiter = maxiter)

            make_comparison_plot(ident, rk4solvers, idesolvers)
