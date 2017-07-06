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

        logger.info(sim.info())
        if not sim.status == si.STATUS_FIN:
            sim.run_simulation()
            sim.save(target_dir = SIM_LIB)
            logger.info(sim.info())

        sim.plot_wavefunction_vs_time(show_vector_potential = 'vel' in sim.name, **PLOT_KWARGS)

        return sim


if __name__ == '__main__':
    with logman as logger:
        photon_energies = np.array([1, 10, 15, 20, 30]) * eV
        amplitudes = np.array([.025, .05, .1, .5, 1, 3]) * atomic_electric_field

        front_periods = 1
        plat_periods = 2
        end_periods = plat_periods + (2 * front_periods) + .1

        for photon_energy, amplitude in itertools.product(photon_energies, amplitudes):
            efield = ion.SineWave.from_photon_energy(photon_energy, amplitude = amplitude)
            efield.window = ion.SmoothedTrapezoidalWindow(time_front = front_periods * efield.period, time_plateau = plat_periods * efield.period)

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
                time_initial = -.1 * efield.period,
                time_final = end_periods * efield.period,
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
                time_step_minimum = .05 * asec,
                time_step_maximum = 10 * asec,
                error_on = 'da/dt',
                epsilon = 1e-6,
                analytic_eigenstate_type = ion.FiniteSquareWellState,
                checkpoints = True,
                checkpoint_dir = SIM_LIB,
                store_data_every = 20,
            )

            prefix = f'E={uround(photon_energy, eV, 3)}eV_amp={uround(amplitude, atomic_electric_field, 3)}aef'

            specs = [
                ion.LineSpecification(
                    prefix + '__fsw_len',
                    internal_potential = internal_potential,
                    initial_state = ion.FiniteSquareWellState.from_potential(internal_potential, mass = test_mass),
                    evolution_gauge = 'LEN',
                    **shared_kwargs,
                ),
                # ion.LineSpecification(
                #     prefix + '__fsw_vel',
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
                    evolution_gauge = 'LEN',
                    evolution_method = 'ARK4',
                    **shared_kwargs,
                ),
                # ide.IntegroDifferentialEquationSpecification(
                #     prefix + '__ide_vel',
                #     prefactor = ide.gaussian_prefactor_VEL(test_width, test_charge, test_mass),
                #     kernel = ide.gaussian_kernel_VEL,
                #     kernel_kwargs = {'tau_alpha': ide.gaussian_tau_alpha_VEL(test_width, test_mass),
                #                      'width': test_width},
                #     evolution_gauge = 'VEL',
                #     evolution_method = 'ARK4',
                #     **shared_kwargs,
                # )
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
                y_data,
                line_labels = (r.name[-7:] if 'line' not in r.name else r.name[-8:] for r in results),
                line_kwargs = y_kwargs,
                x_label = r'$t$', x_unit = 'asec',
                y_label = 'Initial State Population',
                y_log_axis = True,
                y_lower_limit = y_lower_limit, y_upper_limit = y_upper_limit, y_log_pad = 1,
                **PLOT_KWARGS,
            )
