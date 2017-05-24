"""
This script tests all of the evolution methods on each mesh.
"""

import itertools as it
import logging
import os

import numpy as np
import scipy.sparse as sparse

import simulacra as si
from simulacra.units import *

import ionization as ion

import matplotlib.pyplot as plt


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization',
                             stdout_level = logging.DEBUG)


def run_sim(spec):
    with logman as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        # sim.plot_test_state_overlaps_vs_time(target_dir = OUT_DIR)
        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR,
                                      img_format = 'png',
                                      fig_dpi_scale = 3,)



if __name__ == '__main__':
    with logman as logger:
        efield = ion.SineWave(twopi / (300 * asec), amplitude = .1 * atomic_electric_field,
                              window = ion.RectangularTimeWindow(on_time = -300 * asec, off_time = 300 * asec))

        # efield = ion.Rectangle(-100 * asec, -10 * asec, .1 * atomic_electric_field)
        # efield += ion.Rectangle(10 * asec, 100 * asec, -.1 * atomic_electric_field)

        spec_kwargs = dict(
            r_bound = 50 * bohr_radius,
            r_points = 200, l_bound = 30,
            # test_states = tuple(ion.HydrogenBoundState(n, l) for n in range(5) for l in range(n)),
            initial_state = ion.HydrogenBoundState(1, 0),
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_angular_momentum = 10,
            numeric_eigenstate_max_energy = 10 * eV,
            electric_potential = efield,
            time_initial = -400 * asec, time_final = 400 * asec,
            electric_potential_dc_correction = True,
            animators = [
                ion.animators.PolarAnimator(
                    postfix = 'g2',
                    length = 30,
                    fps = 30,
                    target_dir = OUT_DIR,
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                        which = 'g2'
                    )
                ),
                ion.animators.PolarAnimator(
                    postfix = 'g',
                    length = 30,
                    fps = 30,
                    target_dir = OUT_DIR,
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                        which = 'g',
                        colormap = plt.get_cmap('richardson'),
                        norm = si.plots.RichardsonNormalization(),
                    )
                )
            ]
        )

        specs = []

        # for gauge in ('VEL',):
        for gauge in ('LEN', 'VEL'):
            specs.append(ion.SphericalHarmonicSpecification(gauge,
                                                            evolution_gauge = gauge,
                                                            **spec_kwargs,
                                                            store_internal_energy_expectation_value = True,
                                                            ))

        si.utils.multi_map(run_sim, specs)

        # for r in results:
        #     logger.info(r.info())
        #     print()
        #     # print(r)
        #     # print(r.norm_vs_time[-1])
        #     # print(r.energy_expectation_value_vs_time_internal[-1] / eV)
        #     # print(r.mesh.g)
        #     # print()
        #
        #     g_plot_kwargs = dict(
        #         colormap = plt.get_cmap('richardson'),
        #         richardson_equator_magnitude = np.nanmax(np.abs(r.mesh.g)) / 10,
        #         shading = 'flat',
        #         y_unit = 'bohr_radius',
        #         y_label = '$r$',
        #         x_label = '$\ell$',
        #         target_dir = OUT_DIR,
        #     )
        #
        #     # transformed_g = r.mesh.gauge_transformation(r.mesh.g, r.spec.evolution_gauge)
        #     # si.plots.xyz_plot(f'{r.name}__g',
        #     #                   r.mesh.l_mesh, r.mesh.r_mesh, r.mesh.g,
        #     #                   **g_plot_kwargs
        #     #                   )
        #     # si.plots.xyz_plot(f'{r.name}__g_transformed',
        #     #                   r.mesh.l_mesh, r.mesh.r_mesh, transformed_g,
        #     #                   **g_plot_kwargs)
        #
        #     # r.mesh.plot_mesh(r.mesh.g, name = f'{r.name}__g')
        #     # r.mesh.plot_mesh(transformed_g, name = f'{r.name}__g_transformed')
        #     # print(transformed_g)
        #     # print()
        #     # print()
