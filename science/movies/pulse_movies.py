import logging
import os
from copy import deepcopy
import itertools

import simulacra as si
from simulacra.units import *

import ionization as ion

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)


def run(spec):
    with logman as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())


if __name__ == '__main__':
    with logman as logger:
        r_bound = 100
        points_per_r = 5
        r_points = r_bound * points_per_r
        l_bound = 400
        dt = 1
        t_bound = 10
        n_max = 3

        # initial_states = [ion.HydrogenBoundState(1, 0), ion.HydrogenBoundState(2, 0), ion.HydrogenBoundState(2, 1)]
        initial_states = [ion.HydrogenBoundState(1, 0)]

        # test_states = tuple(ion.HydrogenBoundState(n, l) for n in range(n_max + 1) for l in range(n))
        # pulse_types = [ion.SincPulse]
        pulse_types = [ion.SincPulse]
        # pulse_types = [ion.SincPulse, ion.GaussianPulse]
        pulse_widths = [70, 80, 90, 100]
        # pulse_widths = [50, 100, 200, 400, 800]
        # fluences = [1]
        fluences = [.1, 1, 5, 10]
        # phases = [0]
        # phases = [0, pi / 4, pi / 2]
        phases = [0, pi / 2]

        # used by all sims
        mask = ion.RadialCosineMask(inner_radius = (r_bound - 25) * bohr_radius, outer_radius = r_bound * bohr_radius)
        out_dir_mod = os.path.join(OUT_DIR, f'R={r_bound}br_PPR={points_per_r}_L={l_bound}_dt={dt}as_T={t_bound}pw')

        # used by all animators
        animator_kwargs = dict(
            target_dir = out_dir_mod,
            fig_dpi_scale = 2,
            length = 30,
            fps = 30,
        )

        epot_axman = ion.animators.ElectricPotentialPlotAxis(
            show_electric_field = True,
            show_vector_potential = False,
            show_y_label = False,
            show_ticks_right = True,
        )

        specs = []
        for initial_state, pulse_type, pulse_width, fluence, phase in itertools.product(initial_states, pulse_types, pulse_widths, fluences, phases):
            name = f'{pulse_type.__name__}__{initial_state.n}_{initial_state.l}__pw={pulse_width}as_flu={fluence}jcm2_cep={uround(phase, pi)}pi'

            pw = pulse_width * asec
            flu = fluence * Jcm2

            efield = pulse_type.from_omega_min(pulse_width = pw, fluence = flu, phase = phase,
                                               window = ion.SymmetricExponentialTimeWindow(window_time = (t_bound - 2) * pw, window_width = .2 * pw))

            wavefunction_axman = ion.animators.WavefunctionStackplotAxis(
                states = [initial_state],
                legend_kwargs = {'fontsize': 20, 'borderaxespad': .1},
            )

            animators = [
                ion.animators.PolarAnimator(
                    postfix = '__g_wavefunction_wide',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                        which = 'g',
                        colormap = plt.get_cmap('richardson'),
                        norm = si.vis.RichardsonNormalization(),
                    ),
                    axman_lower_right = deepcopy(epot_axman),
                    axman_upper_right = ion.animators.WavefunctionStackplotAxis(states = [initial_state]),
                    axman_colorbar = None,
                    **animator_kwargs,
                ),
                ion.animators.PolarAnimator(
                    postfix = '__g_wavefunction',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                        which = 'g',
                        colormap = plt.get_cmap('richardson'),
                        norm = si.vis.RichardsonNormalization(),
                        plot_limit = 30 * bohr_radius,
                    ),
                    axman_lower_right = deepcopy(epot_axman),
                    axman_upper_right = ion.animators.WavefunctionStackplotAxis(states = [initial_state]),
                    axman_colorbar = None,
                    **animator_kwargs,
                ),
                ion.animators.PolarAnimator(
                    postfix = '__g_angular_momentum',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                        which = 'g',
                        colormap = plt.get_cmap('richardson'),
                        norm = si.vis.RichardsonNormalization(),
                        plot_limit = 30 * bohr_radius,
                    ),
                    axman_lower_right = deepcopy(epot_axman),
                    axman_upper_right = ion.animators.AngularMomentumDecompositionAxis(maximum_l = 10),
                    axman_colorbar = None,
                    **animator_kwargs,
                ),
                ion.animators.PolarAnimator(
                    postfix = '__g2_wavefunction',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                        which = 'g2',
                        plot_limit = 30 * bohr_radius,
                    ),
                    axman_lower_right = deepcopy(epot_axman),
                    axman_upper_right = ion.animators.WavefunctionStackplotAxis(states = [initial_state]),
                    **animator_kwargs,
                ),
                ion.animators.PolarAnimator(
                    postfix = 'g2_wavefunction_wide',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                        which = 'g2',
                    ),
                    axman_lower_right = deepcopy(epot_axman),
                    axman_upper_right = ion.animators.WavefunctionStackplotAxis(states = [initial_state]),
                    **animator_kwargs,
                ),
                ion.animators.PolarAnimator(
                    postfix = 'g2_angular_momentum',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                        which = 'g2',
                        plot_limit = 30 * bohr_radius,
                    ),
                    axman_lower_right = deepcopy(epot_axman),
                    axman_upper_right = ion.animators.AngularMomentumDecompositionAxis(maximum_l = 10),
                    **animator_kwargs,
                ),
            ]

            specs.append(
                ion.SphericalHarmonicSpecification(
                    name,
                    electric_potential = efield,
                    time_initial = -t_bound * pw,
                    time_final = t_bound * pw,
                    time_step = dt * asec,
                    r_points = r_points,
                    r_bound = r_bound * bohr_radius,
                    l_bound = l_bound,
                    initial_state = initial_state,
                    mask = mask,
                    use_numeric_eigenstates = True,
                    numeric_eigenstate_max_energy = 50 * eV,
                    numeric_eigenstate_max_angular_momentum = 20,
                    animators = deepcopy(animators),
                    store_data_every = 5,
                )
            )

        si.utils.multi_map(run, specs, processes = 4)
