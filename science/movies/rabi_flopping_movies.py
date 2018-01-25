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

logman = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)


def run(spec):
    with logman as logger:
        sim = spec.to_simulation()

        sim.info().log()
        sim.run_simulation()
        sim.info().log()


if __name__ == '__main__':
    with logman as logger:
        state_a = ion.HydrogenBoundState(1, 0)
        state_b = ion.HydrogenBoundState(3, 1)

        amplitudes = [.001]
        cycles = [1]
        gauges = ['LEN']

        dt = 1
        bound = 100
        ppbr = 10

        plot_radius = 50

        shading = 'flat'

        animator_kwargs = dict(
            target_dir = OUT_DIR,
            length = 30,
            fps = 30,
            fig_dpi_scale = 2,
        )

        axman_lower_right = animation.animators.ElectricPotentialPlotAxis(
            show_electric_field = True,
            show_vector_potential = False,
            show_y_label = False,
            show_ticks_right = True,
            legend_kwargs = {'fontsize': 30},
            grid_kwargs = {'linewidth': 1},
        )
        axman_upper_right = animation.animators.WavefunctionStackplotAxis(
            states = [state_a, state_b],
            show_norm = False,
            legend_kwargs = {'fontsize': 30, 'borderaxespad': .15},
            grid_kwargs = {'linewidth': 1},
        )

        animators = [
            animation.animators.PolarAnimator(
                postfix = f'__g_{plot_radius}',
                axman_wavefunction = animation.animators.SphericalHarmonicPhiSliceMeshAxis(
                    which = 'g',
                    colormap = plt.get_cmap('richardson'),
                    norm = si.vis.RichardsonNormalization(),
                    plot_limit = plot_radius * bohr_radius,
                    shading = shading,
                ),
                axman_lower_right = deepcopy(axman_lower_right),
                axman_upper_right = deepcopy(axman_upper_right),
                axman_colorbar = None,
                **animator_kwargs,
            ),
            animation.animators.PolarAnimator(
                postfix = f'__g2_{plot_radius}',
                axman_wavefunction = animation.animators.SphericalHarmonicPhiSliceMeshAxis(
                    which = 'g2',
                    plot_limit = plot_radius * bohr_radius,
                    shading = shading,
                ),
                axman_lower_right = deepcopy(axman_lower_right),
                axman_upper_right = deepcopy(axman_upper_right),
                **animator_kwargs,
            ),
        ]

        spec_kwargs = dict(
            r_bound = bound * bohr_radius,
            r_points = bound * ppbr,
            l_bound = 30,
            initial_state = state_a,
            time_initial = 0 * asec,
            time_step = dt * asec,
            mask = ion.RadialCosineMask(inner_radius = .8 * bound * bohr_radius, outer_radius = bound * bohr_radius),
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 20 * eV,
            numeric_eigenstate_max_angular_momentum = 10,
            store_data_every = 10,
        )

        dummy = ion.SphericalHarmonicSpecification('dummy', **spec_kwargs).to_simulation()

        state_a = dummy.mesh.analytic_to_numeric[state_a]
        state_b = dummy.mesh.analytic_to_numeric[state_b]

        dipole_moment = np.abs(dummy.mesh.dipole_moment_inner_product(b = state_b))

        # print(f'calculated dipole moment between {state_a} and {state_b} is {dipole_moment / (proton_charge * bohr_radius)}')
        # print(np.sqrt(2) * (2 ** 7) / (3 ** 5))

        specs = []

        for amplitude, cycle, gauge in itertools.product(amplitudes, cycles, gauges):
            electric_field = ion.SineWave.from_photon_energy(np.abs(state_a.energy - state_b.energy), amplitude = amplitude * atomic_electric_field)

            rabi_frequency = amplitude * atomic_electric_field * dipole_moment / hbar / twopi
            rabi_time = 1 / rabi_frequency

            specs.append(ion.SphericalHarmonicSpecification(
                f'rabi__{state_a.n}_{state_a.l}_to_{state_b.n}_{state_b.l}__amp={amplitude}aef_{cycle}cycles__gauge={gauge}',
                time_final = cycle * rabi_time,
                electric_potential = electric_field,
                evolution_gauge = gauge,
                animators = deepcopy(animators),
                dipole_moment = dipole_moment,
                rabi_frequency = rabi_frequency,
                rabi_time = rabi_time,
                **spec_kwargs
            ))

        if len(specs) > 1:
            si.utils.multi_map(run, specs, processes = 1)
        else:
            with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO) as logger:
                sim = specs[0].to_simulation()
                sim.info().log()
                sim.run_simulation(progress_bar = True)
                sim.info().log()
