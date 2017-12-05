import logging
import os
import itertools

import numpy as np

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


def run(spec):
    with LOGMAN as logger:
        sim = spec.to_simulation()

        sim.run_simulation()

        sim.plot_b2_vs_time(**PLOT_KWARGS)

        return sim


if __name__ == '__main__':
    with LOGMAN as logger:
        max_dts = np.array([5, 2, 1, .5, .2]) * asec
        min_dt = .1 * asec
        methods = ['ARK4']

        pw = 100 * asec
        flu = 10 * Jcm2
        tb = 4
        pulse = ion.GaussianPulse.from_number_of_cycles(
            pulse_width = pw,
            fluence = flu,
        )
        # pulse = ion.SincPulse(
        #     pulse_width = pw,
        #     fluence = flu,
        #     window = ion.SymmetricExponentialTimeWindow(
        #         window_time = tb * pw,
        #         window_width = .2 * pw
        #     )
        # )

        shared_spec_kwargs = dict(
            time_initial = -tb * pulse.pulse_width,
            time_final = tb * pulse.pulse_width,
            electric_potential = pulse,
            kernel = ide.hydrogen_kernel_LEN,
            kernel_kwargs = {'omega_b': ion.HydrogenBoundState(1, 0).energy / hbar},
            time_step_min = min_dt,
        )

        specs = []
        for method, dt in itertools.product(methods, max_dts):
            spec = ide.IntegroDifferentialEquationSpecification(
                f'{method}_dt={uround(dt, asec)}as',
                evolution_method = method,
                time_step = dt,
                time_step_max = dt,
                **shared_spec_kwargs
            )

            specs.append(spec)

        results_raw = si.utils.multi_map(run, specs, processes = 2)
        results = {(r.spec.evolution_method, r.spec.time_step): r for r in results_raw}

        si.vis.xxyy_plot(
            'compare',
            [*[results[method, dt].data_times for method in methods for dt in max_dts]],
            [*[results[method, dt].b2 for method in methods for dt in max_dts]],
            line_labels = [results[method, dt].name for method in methods for dt in max_dts],
            x_unit = 'asec',
            **PLOT_KWARGS
        )

        col_1_len = max(len(method) for method in methods)

        for dt in max_dts:
            print()
            for method in methods:
                r = results[method, dt]
                print(f' {r.spec.evolution_method.ljust(col_1_len)} │ {uround(r.spec.time_step_min, asec, 6):.3f} | {uround(r.spec.time_step_max, asec, 6):.3f} │ {r.b2[-1]:.12f} │ {r.running_time}')
