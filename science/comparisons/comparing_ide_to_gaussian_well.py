import os
import logging

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
    fig_dpi_scale = 5,
)


def run(spec):
    with LOGMAN as logger:
        sim = spec.to_simulation()

        sim.info().log()
        sim.run()
        sim.info().log()

        if 'msh' in sim.name:
            sim.plot_state_overlaps_vs_time(
                states = list(sim.bound_states)[:8],
                show_vector_potential = 'vel' in sim.name,
                **PLOT_KWARGS)
        elif 'ide' in sim.name:
            sim.plot_wavefunction_vs_time(
                show_vector_potential = 'vel' in sim.name,
                **PLOT_KWARGS,
            )

        return sim


if __name__ == '__main__':
    with LOGMAN as logger:
        well_depth = -10 * eV
        well_width = 5 * bohr_radius

        gaussian_well = ion.GaussianPotential(potential_extrema = -10 * eV, width = 5 * bohr_radius)
        variational_ground_state = ion.GaussianWellState.from_potential(gaussian_well, electron_mass)

        test_mass = electron_mass
        test_charge = electron_charge
        test_width = variational_ground_state.width

        pulse_width = 200 * asec
        fluence = 1 * Jcm2
        phase = 0
        pulse = ion.SincPulse(pulse_width = pulse_width, fluence = fluence, phase = phase,
                              window = ion.SymmetricExponentialTimeWindow(window_time = 10 * pulse_width, window_width = .2 * pulse_width))

        identifier = ''

        # PLOT GAUSSIAN WELL POTENTIAL
        x = np.linspace(-5 * well_width, 5 * well_width, 1000)
        si.vis.xy_plot(
            'potential',
            x,
            gaussian_well(distance = x),
            x_label = r'$x$', x_unit = 'bohr_radius',
            y_label = r'$V(x)$', y_unit = 'eV',
            **PLOT_KWARGS,
        )

        # SET UP SPECS FOR COMPARISON
        shared_kwargs = dict(
            test_mass = test_mass,
            test_charge = test_charge,
            test_width = test_width,
            electric_potential = pulse,
            electric_potential_dc_correction = True,
            time_initial = -12 * pulse_width,
            time_final = 12 * pulse_width,
        )

        ide_shared_kwargs = dict(
            evolution_method = 'RK4',
            time_step = 1 * asec,
        )

        msh_spec = ion.LineSpecification(
            'msh',
            time_step = 1 * asec,
            x_bound = 200 * bohr_radius,
            x_points = 2 ** 10,
            internal_potential = gaussian_well,
            initial_state = variational_ground_state,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 20 * eV,
            analytic_eigenstate_type = ion.GaussianWellState,
            mask = ion.RadialCosineMask(inner_radius = 175 * bohr_radius, outer_radius = 200 * bohr_radius),
            animators = [
                animation.animators.RectangleSplitLowerAnimator(
                    postfix = '__g2',
                    axman_wavefunction = animation.animators.LineMeshAxis(),
                    axman_lower_left = animation.animators.ElectricPotentialPlotAxis(show_vector_potential = False),
                    axman_lower_right = animation.animators.WavefunctionStackplotAxis(states = [variational_ground_state]),
                    fig_dpi_scale = 1,
                    target_dir = OUT_DIR,
                ),
            ],
            **shared_kwargs,
        )
        len_spec = ide.IntegroDifferentialEquationSpecification(
            'ide_len',
            evolution_gauge = 'LEN',
            kernel = ide.gaussian_kernel_LEN,
            kernel_kwargs = {'tau_alpha': ide.gaussian_tau_alpha_LEN(test_width, test_mass)},
            integral_prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge),
            **shared_kwargs,
            **ide_shared_kwargs,
        )
        vel_spec = ide.IntegroDifferentialEquationSpecification(
            'ide_vel',
            evolution_gauge = 'VEL',
            kernel = ide.gaussian_kernel_VEL,
            kernel_kwargs = {'tau_alpha': ide.gaussian_tau_alpha_VEL(test_width, test_mass), 'width': test_width},
            integral_prefactor = ide.gaussian_prefactor_VEL(test_width, test_charge, test_mass),
            **shared_kwargs,
            **ide_shared_kwargs,
        )

        results = si.utils.multi_map(run, [msh_spec, len_spec, vel_spec], processes = 2)

        final_initial_state_overlaps = []
        with open(os.path.join(OUT_DIR, f'results__{identifier}.txt'), mode = 'w') as file:
            for r in results:
                try:
                    final_initial_state_overlap = r.state_overlaps_vs_time[r.spec.initial_state][-1]
                except AttributeError:  # ide simulation
                    final_initial_state_overlap = r.b2[-1]
                file.write(f'{final_initial_state_overlap} : {r.name}\n')
                final_initial_state_overlaps.append(final_initial_state_overlap)

            file.write('\n\n\n')

            for r in results:
                file.write(str(r.info()) + '\n')

        y_data = []
        y_kwargs = []
        for r in results:
            try:
                y_data.append(r.state_overlaps_vs_time[r.spec.initial_state])
            except AttributeError:  # ide simulation
                y_data.append(r.b2)

        si.vis.xxyy_plot(
            f'comparison__{identifier}',
            [r.data_times for r in results],
            y_data,
            line_labels = [r.name for r in results],
            x_label = r'$t$', x_unit = 'asec',
            y_label = r'Initial State Overlap',
            **PLOT_KWARGS,
        )
