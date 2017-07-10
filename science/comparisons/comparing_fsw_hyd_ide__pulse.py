import logging
import os
import itertools

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.integrodiff as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, 'simlib')

logman = si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 3,
)


def run(spec):
    with logman as logger:
        sim = si.utils.find_or_init_sim(spec, search_dir = SIM_LIB)
        sim.spec.hydrogen_zero_angular_momentum_correction = True

        logger.info(sim.info())
        if not sim.status == si.STATUS_FIN:
            sim.run_simulation()
            sim.save(target_dir = SIM_LIB)
            logger.info(sim.info())

        sim.plot_wavefunction_vs_time(**PLOT_KWARGS)

        return sim


if __name__ == '__main__':
    with logman as logger:
        pulse_widths = np.array([100, 200, 400, 800]) * asec
        fluences = np.array([.001, .01, 0.1, 1, 10, 20]) * Jcm2
        # fluences = np.array([.1, 1, 10, 20]) * Jcm2
        phases = [0, pi / 2]

        for pw, flu, phase in itertools.product(pulse_widths, fluences, phases):
            t_bound = 32

            efield = ion.SincPulse(pulse_width = pw, fluence = flu, phase = phase,
                                   window = ion.SymmetricExponentialTimeWindow(window_time = (t_bound - 2) * pw, window_width = .2 * pw))

            test_width = 1 * bohr_radius
            test_charge = 1 * electron_charge
            test_mass = 1 * electron_mass
            potential_depth = 36.831335 * eV

            internal_potential = ion.FiniteSquareWell(potential_depth = potential_depth, width = test_width)

            shared_kwargs = dict(
                test_width = test_width,
                test_charge = test_charge,
                test_mass = test_mass,
                potential_depth = potential_depth,
                electric_potential = efield,
                time_initial = -t_bound * pw,
                time_final = t_bound * pw,
                time_step = 1 * asec,
                electric_potential_dc_correction = True,
                x_bound = 200 * bohr_radius,
                x_points = 2 ** 12,
                r_bound = 200 * bohr_radius,
                r_points = 800,
                l_bound = 400,
                mask = ion.RadialCosineMask(inner_radius = 175 * bohr_radius, outer_radius = 200 * bohr_radius),
                use_numeric_eigenstates = True,
                numeric_eigenstate_max_energy = 10 * eV,
                numeric_eigenstate_max_angular_momentum = 10,
                time_step_minimum = 1 * asec,
                time_step_maximum = 10 * asec,
                error_on = 'da/dt',
                epsilon = 1e-6,
                analytic_eigenstate_type = ion.FiniteSquareWellState,
                checkpoints = True,
                checkpoint_dir = SIM_LIB,
                store_data_every = 1,
            )

            prefix = f'pw={uround(pw, asec, 2)}as_flu={uround(flu, Jcm2, 4)}jcm2_phase={uround(phase, pi, 3)}pi'

            fsw_initial_state = ion.FiniteSquareWellState.from_potential(internal_potential, mass = test_mass)

            specs = [
                ion.LineSpecification(
                    prefix + '__fsw_len',
                    internal_potential = internal_potential,
                    initial_state = fsw_initial_state,
                    evolution_gauge = 'LEN',
                    **shared_kwargs,
                ),
                # ion.LineSpecification(
                #     prefix + '__line_vel',
                #     internal_potential = internal_potential,
                #     initial_state = ion.FiniteSquareWellState.from_potential(internal_potential, mass = test_mass),
                #     evolution_gauge = 'VEL',
                #     **shared_kwargs,
                # ),
                ion.SphericalHarmonicSpecification(
                    prefix + '__hyd_len',
                    evolution_gauge = 'LEN',
                    **shared_kwargs,
                ),
                # ion.SphericalHarmonicSpecification(
                #     prefix + '__hyd_vel',
                #     evolution_gauge = 'VEL',
                #     **shared_kwargs,
                # ),
                ide.IntegroDifferentialEquationSpecification(
                    prefix + '__ide_len',
                    prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge),
                    kernel = ide.gaussian_kernel_LEN,
                    kernel_kwargs = {'tau_alpha': ide.gaussian_tau_alpha_LEN(test_width, test_mass)},
                    test_energy = fsw_initial_state.energy,
                    evolution_gauge = 'LEN',
                    evolution_method = 'ARK4',
                    **shared_kwargs,
                ),
                ide.IntegroDifferentialEquationSpecification(
                    prefix + '__ide_vel',
                    prefactor = ide.gaussian_prefactor_VEL(test_width, test_charge, test_mass),
                    kernel = ide.gaussian_kernel_VEL,
                    kernel_kwargs = {'tau_alpha': ide.gaussian_tau_alpha_VEL(test_width, test_mass),
                                     'width': test_width},
                    test_energy = fsw_initial_state.energy,
                    evolution_gauge = 'VEL',
                    evolution_method = 'ARK4',
                    **shared_kwargs,
                )
            ]

            results = si.utils.multi_map(run, specs, processes = 4)

            final_initial_state_overlaps = []
            with open(os.path.join(OUT_DIR, f'results__{prefix}.txt'), mode = 'w') as file:
                for r in results:
                    try:
                        final_initial_state_overlap = r.state_overlaps_vs_time[r.spec.initial_state][-1]
                        file.write(f'{final_initial_state_overlap} : {r.name}\n')
                    except AttributeError:  # ide simulation
                        final_initial_state_overlap = r.a2[-1]
                        file.write(f'{final_initial_state_overlap} : {r.name}\n')
                    final_initial_state_overlaps.append(final_initial_state_overlap)

                file.write('\n\n\n')

                for r in results:
                    file.write(str(r.info()) + '\n')

            styles = {'len': '-', 'vel': '--'}
            colors = {'hyd': 'C0', 'fsw': 'C1', 'ide': 'C2'}


            def get_style_and_color_keys(name):
                for key in styles:
                    if key in name:
                        key_style = key
                for key in colors:
                    if key in name:
                        color_style = key

                return key_style, color_style


            y_data = []
            y_kwargs = []
            for r in results:
                key_style, key_color = get_style_and_color_keys(r.name)
                y_kwargs.append({'linestyle': styles[key_style], 'color': colors[key_color]})
                try:
                    y_data.append(r.state_overlaps_vs_time[r.spec.initial_state])
                except AttributeError:  # ide simulation
                    y_data.append(r.a2)

            si.vis.xxyy_plot(
                f'comparison__{prefix}',
                list(r.data_times for r in results),
                y_data,
                line_labels = (r.name[-7:] if 'line' not in r.name else r.name[-8:] for r in results),
                line_kwargs = y_kwargs,
                x_label = r'$t$', x_unit = 'asec',
                y_label = 'Initial State Population',
                **PLOT_KWARGS,
            )

            y_lower_limit, y_upper_limit = si.vis.get_axis_limits([1 - x for x in final_initial_state_overlaps], log = True, log_pad = 2)
            si.vis.xxyy_plot(
                f'comparison__{prefix}__log',
                list(r.data_times for r in results),
                [1 - y for y in y_data],
                line_labels = (r.name[-7:] if 'line' not in r.name else r.name[-8:] for r in results),
                line_kwargs = y_kwargs,
                x_label = r'$t$', x_unit = 'asec',
                y_label = '1 - Initial State Population',
                y_log_axis = True,
                y_lower_limit = y_lower_limit, y_upper_limit = y_upper_limit, y_log_pad = 1,
                **PLOT_KWARGS,
            )
