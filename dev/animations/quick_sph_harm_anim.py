import logging
import os
from copy import deepcopy

import simulacra as si
from simulacra.units import *

import ionization as ion

import matplotlib.pyplot as plt


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

# def make_movie(spec):
#     with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.INFO,
#                              file_logs = True, file_name = spec.name, file_dir = OUT_DIR, file_mode = 'w', file_level = logging.DEBUG) as logger:
#         sim = spec.to_simulation()
#
#         logger.info(sim.info())
#         sim.run_simulation()
#         logger.info(sim.info())


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        anim_kwargs = dict(
                length = 10,
                target_dir = OUT_DIR,
        )

        epot_axman = ion.animators.ElectricPotentialAxis(
                show_electric_field = True,
                show_vector_potential = False,
                show_y_label = False,
                show_ticks_right = True,
        )

        test_state_axman = ion.animators.TestStateStackplot(
                states = None
        )

        animators = [
            ion.animators.PhiSliceAnimator(
                    postfix = 'g2',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                            shading = 'flat'
                    ),
                    axman_lower_right = deepcopy(epot_axman),
                    axman_upper_right = deepcopy(test_state_axman),
                    **anim_kwargs,
            ),
            ion.animators.PhiSliceAnimator(
                    postfix = 'g',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                            which = 'g',
                            colormap = plt.get_cmap('richardson'),
                            norm = si.plots.RichardsonNormalization(),
                            shading = 'flat'),
                    axman_lower_right = deepcopy(epot_axman),
                    axman_upper_right = deepcopy(test_state_axman),
                    **anim_kwargs,
            )
        ]

        sim = ion.SphericalHarmonicSpecification('sph_harm',
                                                 time_initial = 0 * asec, time_final = 100 * asec,
                                                 r_bound = 25 * bohr_radius, l_bound = 20,
                                                 test_states = tuple(ion.HydrogenBoundState(n, l) for n in range(5) for l in range(n)),
                                                 r_points = 100,
                                                 electric_potential = ion.Rectangle(start_time = 25 * asec, end_time = 75 * asec, amplitude = 1 * atomic_electric_field),
                                                 animators = animators).to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())
