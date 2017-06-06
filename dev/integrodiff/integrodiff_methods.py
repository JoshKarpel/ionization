import logging
import os
import itertools

import numpy as np

import simulacra as si
from simulacra.units import *
import ionization as ion
from ionization import integrodiff as ide


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG)

PLT_KWARGS = (
    dict(
            target_dir = OUT_DIR,
            img_format = 'png',
            fig_dpi_scale = 3,
    ),
    dict(
            target_dir = OUT_DIR,
    )
)


def run(spec):
    with logman as logger:
        sim = spec.to_simulation()

        logger.debug(sim.info())
        sim.run_simulation()
        logger.debug(sim.info())

        for kwargs in PLT_KWARGS:
            sim.plot_a2_vs_time(**kwargs)

        return sim


if __name__ == '__main__':
    with logman as logger:
        dt = .1 * asec

        pw = 100 * asec
        flu = .1 * Jcm2

        t_bound = 30
        electric_pot = ion.SincPulse(pulse_width = pw, fluence = flu,
                                     window = ion.SymmetricExponentialTimeWindow((t_bound - 2) * pw, .2 * pw))

        # t_bound = 3
        # electric_pot = ion.Rectangle(start_time = -5 * pw, end_time = 5 * pw, amplitude = 1 * atomic_electric_field,
        #                              window = ion.SymmetricExponentialTimeWindow(pw, .1 * pw))

        q = electron_charge
        m = electron_mass_reduced
        L = bohr_radius

        tau_alpha = 4 * m * (L ** 2) / hbar
        prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

        int_methods = ['simpson']
        # int_methods = ['trapezoid', 'simpson']
        # evol_methods = ['ARK4', ]
        evol_methods = ['FE', 'BE', 'TRAP', 'RK4', 'ARK4']
        # evol_methods = ['FE', 'BE', 'TRAP']

        spec_kwargs = dict(
                time_initial = -t_bound * pw, time_final = t_bound * pw, time_step = dt,
                prefactor = prefactor,
                electric_potential = electric_pot,
                kernel = ide.gaussian_kernel, kernel_kwargs = dict(tau_alpha = tau_alpha),
                # electric_potential_dc_correction = True,
        )

        specs = []

        for evol_method, int_method in itertools.product(evol_methods, int_methods):
            specs.append(ide.IntegroDifferentialEquationSpecification(
                    f'{evol_method}',
                    # 'int={}_evol={}'.format(int_method, evol_method),
                    evolution_method = evol_method,
                    integration_method = int_method,
                    # maximum_time_step = dt,
                    **spec_kwargs
            ))

        results = si.utils.multi_map(run, specs)

        for kwargs in PLT_KWARGS:
            si.vis.xxyy_plot(
                    f'NEW_method_comparison__pw={uround(pw, asec, 3)}_dt={uround(dt, asec, 3)}',
                    (r.times for r in results),
                    (r.a2 for r in results),
                    line_labels = (r.name for r in results),
                    x_label = r'Time $t$', x_unit = 'asec',
                    y_label = r'$\left| a(t) \right|^2$', y_lower_limit = 0, y_upper_limit = 1,
                    title = fr'Method Comparison at $\tau = {uround(pw, asec, 3)}$ as, $\Delta t = {uround(dt, asec, 3)}$ as',
                    **kwargs,
            )

        for r in results:
            print(r.info())

        print()

        for r in results:
            print(r.name.ljust(5), np.abs(r.a[-1]))
