import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 3,
)
ANIM_KWARGS = dict(
    length = 30,
    target_dir = OUT_DIR,
)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO) as logger:
        mass = electron_mass
        truncated_qho_pot = ion.HarmonicOscillator.from_energy_spacing_and_mass(energy_spacing = 10 * eV, mass = mass, cutoff_distance = 2 * bohr_radius)

        # init = ion.Superposition({ion.QHOState(omega = pot.omega(mass), mass = mass, n = 0): 1,
        #                           ion.QHOState(omega = pot.omega(mass), mass = mass, n = 1): 1})
        init = ion.QHOState.from_potential(truncated_qho_pot, mass, n = 0)

        efield = ion.SineWave.from_photon_energy(1 * eV, amplitude = .01 * atomic_electric_field)

        animators = [
            animation.animators.RectangleSplitLowerAnimator(
                postfix = 'psi2_split',
                axman_wavefunction = animation.animators.LineMeshAxis(
                    which = 'psi2'
                ),
                axman_lower_left = animation.animators.ElectricPotentialPlotAxis(),
                axman_lower_right = animation.animators.WavefunctionStackplotAxis(
                    states = (ion.QHOState.from_potential(truncated_qho_pot, mass = electron_mass, n = n) for n in range(4)),
                    show_norm = False,
                    legend_kwargs = {'fontsize': 12},
                ),
                **ANIM_KWARGS,
            ),
        ]

        sim = ion.LineSpecification('truncated_qho',
                                    x_bound = 50 * bohr_radius, x_points = 2 ** 13,
                                    internal_potential = truncated_qho_pot,
                                    electric_potential = efield,
                                    test_mass = mass,
                                    # use_numeric_eigenstates = True,
                                    # numeric_eigenstate_max_energy = 10 * eV,
                                    # analytic_eigenstate_type = ion.QHOState,
                                    initial_state = init,
                                    time_initial = 0, time_final = efield.period_carrier * 10, time_step = 10 * asec,
                                    animators = animators,
                                    ).to_simulation()

        sim.info().log()

        si.vis.xy_plot(
            'truncated_qho_potential',
            sim.mesh.x_mesh,
            truncated_qho_pot(distance = sim.mesh.x_mesh),
            x_label = r'$x$', x_unit = 'bohr_radius',
            y_label = r'$V(x)$', y_unit = 'eV',
            **PLOT_KWARGS,
        )

        # for state in sim.spec.test_states:
        #     si.vis.xy_plot(
        #         f'state__{state}',
        #         sim.mesh.x_mesh,
        #         np.abs(state.analytic_state(sim.mesh.x_mesh)) ** 2,
        #         np.abs(state(sim.mesh.x_mesh)) ** 2,
        #         line_labels = ['analytic', 'numeric'],
        #         line_kwargs = [None, {'linestyle': '--'}],
        #         x_label = r'$x$', x_unit = 'bohr_radius',
        #         **PLOT_KWARGS,
        #     )

        sim.run(progress_bar = True)

        sim.plot_state_overlaps_vs_time(time_unit = 'fsec', show_vector_potential = False, **PLOT_KWARGS)

        sim.info().log()
