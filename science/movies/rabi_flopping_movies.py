import logging
import os
import itertools
from copy import deepcopy

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion

import matplotlib.pyplot as plt


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG, )


def run(spec):
    with logman as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())


if __name__ == '__main__':
    with logman as logger:
        state_a = ion.HydrogenBoundState(1, 0)
        state_b = ion.HydrogenBoundState(2, 1)

        amplitudes = [.005, .01, .1]
        cycles = [1, 2, 3]

        dt = 1
        bound = 100
        ppbr = 8

        inner = 20
        outer = 50

        animator_kwargs = dict(
                target_dir = OUT_DIR,
                length = 60,
                fps = 30,
        )

        lower_right_axman = ion.animators.ElectricPotentialPlotAxis(
                show_electric_field = True,
                show_vector_potential = False,
                show_y_label = False,
                show_ticks_right = True,
        )
        upper_right_axman = ion.animators.WavefunctionStackplotAxis(
                states = [state_a, state_b],
                show_norm = False,
                legend_kwargs = {'fontsize': 12}
        )

        animators = [
            ion.animators.PolarAnimator(
                    postfix = f'g2_{outer}',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                            which = 'g2',
                            plot_limit = outer * bohr_radius
                    ),
                    axman_lower_right = deepcopy(lower_right_axman),
                    axman_upper_right = deepcopy(upper_right_axman),
                    **animator_kwargs,
            ),
            ion.animators.PolarAnimator(
                    postfix = f'g2_{inner}',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                            which = 'g2',
                            plot_limit = inner * bohr_radius,
                    ),
                    axman_lower_right = deepcopy(lower_right_axman),
                    axman_upper_right = deepcopy(upper_right_axman),
                    **animator_kwargs,
            ),
            ion.animators.PolarAnimator(
                    postfix = f'g_{outer}',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                            which = 'g',
                            colormap = plt.get_cmap('richardson'),
                            norm = si.plots.RichardsonNormalization(),
                            plot_limit = outer * bohr_radius,
                    ),
                    axman_lower_right = deepcopy(lower_right_axman),
                    axman_upper_right = deepcopy(upper_right_axman),
                    **animator_kwargs,
            ),
            ion.animators.PolarAnimator(
                    postfix = f'g_{inner}',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                            which = 'g',
                            colormap = plt.get_cmap('richardson'),
                            norm = si.plots.RichardsonNormalization(),
                            plot_limit = inner * bohr_radius,
                    ),
                    axman_lower_right = deepcopy(lower_right_axman),
                    axman_upper_right = deepcopy(upper_right_axman),
                    **animator_kwargs,
            ),
        ]

        spec_kwargs = dict(
                r_bound = bound * bohr_radius,
                r_points = bound * ppbr,
                l_bound = 50,
                initial_state = state_a,
                test_states = (state_a, state_b),
                time_initial = 0 * asec,
                time_step = dt * asec,
                mask = ion.RadialCosineMask(inner_radius = .8 * bound * bohr_radius, outer_radius = bound * bohr_radius),
                use_numeric_eigenstates = True,
                numeric_eigenstate_max_energy = 20 * eV,
                numeric_eigenstate_max_angular_momentum = 10,
                animators = animators,
                store_data_every = -1,
                out_dir = OUT_DIR
        )

        dummy = ion.SphericalHarmonicSpecification('dummy', **spec_kwargs).to_simulation()
        matrix_element = np.abs(dummy.mesh.dipole_moment_expectation_value(mesh_a = dummy.mesh.get_g_for_state(state_b)))

        specs = []

        for amplitude, cycle in itertools.product(amplitudes, cycles):
            electric_field = ion.SineWave.from_photon_energy(np.abs(state_a.energy - state_b.energy), amplitude = amplitude * atomic_electric_field)

            rabi_frequency = amplitude * atomic_electric_field * matrix_element / hbar / twopi
            rabi_time = 1 / rabi_frequency

            specs.append(ion.SphericalHarmonicSpecification(
                    f'rabi__{state_a.n}_{state_a.l}_to_{state_b.n}_{state_b.l}__amp={amplitude}aef_{cycle}cycles__len={animator_kwargs["length"]}_fps={animator_kwargs["fps"]}',
                    time_final = cycle * rabi_time,
                    electric_potential = electric_field,
                    rabi_frequency = rabi_frequency,
                    rabi_time = rabi_time,
                    **spec_kwargs
            ))

        si.utils.multi_map(run, specs)
