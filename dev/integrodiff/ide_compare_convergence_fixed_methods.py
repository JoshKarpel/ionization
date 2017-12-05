import logging
import os
import itertools
import datetime

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, 'SIMLIB')

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def run(spec):
    with LOGMAN as logger:
        sim = spec.to_simulation()

        sim.run_simulation()

        return sim


def run_from_lib(spec):
    with LOGMAN as logger:
        sim = si.utils.find_or_init_sim(spec, search_dir = SIM_LIB)

        if sim.status != si.Status.FINISHED:
            sim.run_simulation()
            sim.save(target_dir = SIM_LIB)

        return sim


if __name__ == '__main__':
    with LOGMAN as logger:
        # dts = np.array([5, 2, 1, .5]) * asec
        # dts = np.geomspace(.1, 10, 30) * asec
        dts = np.geomspace(.1, 20, 30) * asec
        methods = ['FE', 'BE', 'TRAP', 'RK4']

        pw = 50 * asec
        flu = .1 * Jcm2
        cep = 0
        tb = 3.5
        pulse = ion.GaussianPulse.from_number_of_cycles(
            pulse_width = pw,
            fluence = flu,
            phase = cep,
            number_of_cycles = 3,
        )

        pulse_ident = f'{pulse.__class__.__name__}_pw={uround(pw, asec, 3)}as_flu={uround(flu, Jcm2, 3)}jcm2_cep={uround(cep, pi)}pi_tb={tb}'

        shared_spec_kwargs = dict(
            time_initial = -tb * pulse.pulse_width,
            time_final = tb * pulse.pulse_width,
            electric_potential = pulse,
            kernel = ide.hydrogen_kernel_LEN,
            kernel_kwargs = {'omega_b': ion.HydrogenBoundState(1, 0).energy / hbar},
            # checkpoints = True,
            # checkpoint_dir = SIM_LIB,
            # checkpoint_every = datetime.timedelta(minutes = 1),
        )

        specs = []
        for method, dt in itertools.product(methods, dts):
            spec = ide.IntegroDifferentialEquationSpecification(
                f'{method}_dt={uround(dt, asec)}as__{pulse_ident}',
                evolution_method = method,
                time_step = dt,
                **shared_spec_kwargs
            )

            specs.append(spec)

        results = si.utils.multi_map(run, specs, processes = 3)
        results_by_method = {method: sorted([r for r in results if r.spec.evolution_method == method], key = lambda x: x.spec.time_step) for method in methods}

        b2_final_by_method = {method: np.array([r.b2[-1] for r in results]) for method, results in results_by_method.items()}

        best_b2 = b2_final_by_method['RK4'][0]
        error_by_method = {method: np.abs(best_b2 - b2_final) for method, b2_final in b2_final_by_method.items()}

        colors = [f'C{x}' for x in range(len(methods))]

        fm = si.vis.xy_plot(
            f'convergence__{pulse_ident}',
            dts,
            *[error for method, error in error_by_method.items()],
            line_labels = methods,
            line_kwargs = [{'color': color} for color in colors],
            x_label = r'$ \Delta t $',
            x_unit = 'asec',
            x_log_axis = True,
            y_label = r'$ \left| \left|b_{\mathrm{best}}(t_f)\right|^2 - \left|b(t_f)\right|^2 \right| $ [solid]',
            y_log_axis = True,
            save_on_exit = False,
            close_after_exit = False,
            **PLOT_KWARGS
        )

        fig = fm.fig
        ax_error = fm.fig.axes[0]

        ax_time = ax_error.twinx()

        for method, color in zip(methods, colors):
            ax_time.plot(
                dts / asec,
                [r.running_time.total_seconds() for r in results_by_method[method]],
                color = color,
                linestyle = '--',
            )

        ax_time.set_ylabel('Runtime (s) [dashed]')
        ax_time.set_xlim(np.min(dts) / asec, np.max(dts) / asec)

        fm.save()
        fm.cleanup()

        col_1_len = max(len(method) for method in methods)

        # for dt in dts:
        #     print()
        #     for method in methods:
        #         r = results[method, dt]
        #         print(f' {r.spec.evolution_method.ljust(col_1_len)} │ {uround(r.spec.time_step, asec, 6):.3f} │ {r.b2[-1]:.12f} │ {r.running_time}')
