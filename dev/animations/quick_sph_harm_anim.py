import logging
import os
from copy import deepcopy

import simulacra as si
from simulacra.units import *

import ionization as ion

import matplotlib.pyplot as plt


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        anim_kwargs = dict(
                length = 10,
                target_dir = OUT_DIR,
        )

        epot_axman = ion.animators.ElectricPotentialPlotAxis(
                show_electric_field = True,
                show_vector_potential = False,
                show_y_label = False,
                show_ticks_right = True,
        )

        test_state_axman = ion.animators.TestStateStackplotAxis(
                states = tuple(ion.HydrogenBoundState(n, l) for n in range(5) for l in range(n))[:8]
        )

        animators = [
            ion.animators.PolarAnimator(
                    postfix = 'g2',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                            shading = 'flat'
                    ),
                    axman_lower_right = deepcopy(epot_axman),
                    axman_upper_right = deepcopy(test_state_axman),
                    axman_colorbar = ion.animators.ColorBarAxis(),
                    **anim_kwargs,
            ),
            ion.animators.PolarAnimator(
                    postfix = 'g',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                            which = 'g',
                            colormap = plt.get_cmap('richardson'),
                            norm = si.plots.RichardsonNormalization(),
                            shading = 'flat'),
                    axman_lower_right = deepcopy(epot_axman),
                    axman_upper_right = deepcopy(test_state_axman),
                    axman_colorbar = None,
                    **anim_kwargs,
            ),
            ion.animators.PolarAnimator(
                    postfix = 'g_angmom',
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                            which = 'g',
                            colormap = plt.get_cmap('richardson'),
                            norm = si.plots.RichardsonNormalization(),
                            shading = 'flat'),
                    axman_lower_right = deepcopy(epot_axman),
                    axman_upper_right = ion.animators.AngularMomentumDecompositionAxis(maximum_l = 10),
                    axman_colorbar = None,
                    **anim_kwargs,
            ),
        ]

        sim = ion.SphericalHarmonicSpecification('sph_harm',
                                                 time_initial = 0 * asec, time_final = 100 * asec,
                                                 r_bound = 50 * bohr_radius, l_bound = 20,
                                                 r_points = 200,
                                                 electric_potential = ion.Rectangle(start_time = 25 * asec, end_time = 75 * asec, amplitude = 1 * atomic_electric_field),
                                                 use_numeric_eigenstates = True,
                                                 numeric_eigenstate_max_energy = 10 * eV,
                                                 numeric_eigenstate_max_angular_momentum = 5,
                                                 animators = animators).to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())
