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
        dt = 2 * asec

        pw = 100 * asec
        flu = .1 * Jcm2

        # t_bound = 30
        # electric_pot = ion.SincPulse(pulse_width = pw, fluence = flu,
        #                              window = ion.SymmetricExponentialTimeWindow((t_bound - 2) * pw, .2 * pw))

        t_bound = 3
        electric_pot = ion.Rectangle(start_time = -5 * pw, end_time = 5 * pw, amplitude = 1 * atomic_electric_field,
                                     window = ion.SymmetricExponentialTimeWindow(pw, .1 * pw))

        q = electron_charge
        m = electron_mass_reduced
        L = bohr_radius

        tau_alpha = 4 * m * (L ** 2) / hbar
        prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

        int_methods = ['simpson']
        # int_methods = ['trapezoid', 'simpson']
        evol_methods = ['FE', 'BE', 'RK4', 'TRAP']
        # evol_methods = ['FE', 'BE', 'TRAP']
        #
        # ark4 = ide.AdaptiveIntegroDifferentialEquationSpecification('int={}__evol={}'.format('simpson', 'ARK4'),
        #                                                             time_initial = -t_bound * asec, time_final = t_bound * asec, time_step = 1 * asec,
        #                                                             prefactor = prefactor,
        #                                                             electric_potential = electric_pot,
        #                                                             kernel = ide.gaussian_kernel, kernel_kwargs = dict(tau_alpha = tau_alpha),
        #                                                             evolution_method = 'ARK4',
        #                                                             integration_method = 'simpson',
        #                                                             ).to_simulation()
        # ark4.run_simulation()
        #
        # si.vis.xy_plot('time_step',
        #                ark4.times,
        #                ark4.time_steps_list,
        #                x_axis_label = r'Time $t$', x_unit = 'asec',
        #                y_axis_label = r'Time Step $\Delta t$', y_unit = 'asec',
        #                y_log_axis = True,
        #                target_dir = OUT_DIR,
        #                )

        specs = []

        for evol_method, int_method in itertools.product(evol_methods, int_methods):
            specs.append(ide.IntegroDifferentialEquationSpecification(
                    f'{evol_method}',
                    # 'int={}_evol={}'.format(int_method, evol_method),
                    time_initial = -t_bound * pw, time_final = t_bound * pw, time_step = dt,
                    prefactor = prefactor,
                    electric_potential = electric_pot,
                    kernel = ide.gaussian_kernel, kernel_kwargs = dict(tau_alpha = tau_alpha),
                    evolution_method = evol_method,
                    integration_method = int_method,
                    # electric_potential_dc_correction = True,
            ))

        results = si.utils.multi_map(run, specs)

        for r in results:
            print(r.times)

        for kwargs in PLT_KWARGS:
            si.vis.xxyy_plot(
                    f'NEW_method_comparison__pw={uround(pw, asec, 3)}_dt={uround(dt, asec, 3)}',
                    (r.times for r in results),
                    (r.a2 for r in results),
                    line_labels = (r.name for r in results),
                    x_label = r'Time $t$', x_unit = 'asec',
                    y_label = r'$\left| a(t) \right|^2$',
                    title = fr'Method Comparison at $\tau = {uround(pw, asec, 3)}$ as, $\Delta t = {uround(dt, asec, 3)}$ as',
                    **kwargs,
            )
