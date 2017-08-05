import itertools
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

        sim.info().log()
        sim.run_simulation()
        sim.info().log()

        # sim.plot_test_state_overlaps_vs_time(target_dir = OUT_DIR)
        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR,
                                      img_format = 'png',
                                      fig_dpi_scale = 3, )


if __name__ == '__main__':
    with logman as logger:
        # efield = ion.SineWave(twopi / (300 * asec), amplitude = .1 * atomic_electric_field,
        #                       window = ion.RectangularTimeWindow(on_time = -300 * asec, off_time = 300 * asec))

        # efield = ion.Rectangle(-100 * asec, -10 * asec, .1 * atomic_electric_field)
        # efield += ion.Rectangle(10 * asec, 100 * asec, -.1 * atomic_electric_field)

        efield = ion.Rectangle(-100 * asec, 100 * asec, 1 * atomic_electric_field)

        r_points = (200, 201)
        l_points = (30, 31)

        spec_kwargs = dict(
            r_bound = 50 * bohr_radius,
            # r_points = 200, l_points = 30,
            # test_states = tuple(ion.HydrogenBoundState(n, l) for n in range(5) for l in range(n)),
            initial_state = ion.HydrogenBoundState(1, 0),
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_angular_momentum = 10,
            numeric_eigenstate_max_energy = 10 * eV,
            electric_potential = efield,
            time_initial = -150 * asec, time_final = 150 * asec,
            electric_potential_dc_correction = True,
            animators = [
                ion.animators.PolarAnimator(
                    postfix = 'g2',
                    length = 10,
                    fps = 30,
                    target_dir = OUT_DIR,
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                        which = 'g2'
                    ),
                    axman_lower_right = ion.animators.ElectricPotentialPlotAxis(
                        show_vector_potential = True
                    ),
                    axman_upper_right = ion.animators.WavefunctionStackplotAxis(
                    ),
                ),
                ion.animators.PolarAnimator(
                    postfix = 'g',
                    length = 10,
                    fps = 30,
                    target_dir = OUT_DIR,
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                        which = 'g',
                        colormap = plt.get_cmap('richardson'),
                        norm = si.vis.RichardsonNormalization(),
                    ),
                    axman_lower_right = ion.animators.ElectricPotentialPlotAxis(
                        show_vector_potential = True,
                    ),
                    axman_upper_right = ion.animators.WavefunctionStackplotAxis(
                    ),
                )
            ]
        )

        specs = []

        for rr, ll, gauge in itertools.product(r_points, l_points, ('LEN', 'VEL')):
            specs.append(ion.SphericalHarmonicSpecification(f'{gauge}__r={rr}_l={ll}',
                                                            r_points = rr, l_bound = ll,
                                                            evolution_gauge = gauge,
                                                            **spec_kwargs,
                                                            store_internal_energy_expectation_value = True,
                                                            ))

        si.utils.multi_map(run_sim, specs)
