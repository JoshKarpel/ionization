import logging
import os

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

        pot = ion.HarmonicOscillator.from_energy_spacing_and_mass(1 * eV)

        animators = [
            ion.animators.RectangleAnimator(
                postfix = 'psi2',
                axman_wavefunction = ion.animators.LineMeshAxis(
                    which = 'psi2'
                ),
                axman_lower = ion.animators.ElectricPotentialPlotAxis(),
                **anim_kwargs,
            ),
            ion.animators.RectangleSplitLowerAnimator(
                postfix = 'psi2_split',
                axman_wavefunction = ion.animators.LineMeshAxis(
                    which = 'psi2'
                ),
                axman_lower_left = ion.animators.ElectricPotentialPlotAxis(),
                axman_lower_right = ion.animators.WavefunctionStackplotAxis(
                    states = (ion.QHOState.from_potential(pot, mass = electron_mass, n = n) for n in range(4)),
                    show_norm = False,
                    legend_kwargs = {'fontsize': 12},
                ),
                **anim_kwargs,
            ),
        ]

        sim = ion.LineSpecification('line',
                                    time_initial = 0 * asec, time_final = 300 * asec,
                                    x_bound = 100 * bohr_radius, x_points = 2 ** 10,
                                    internal_potential = pot,
                                    initial_state = ion.QHOState.from_potential(pot, mass = electron_mass),
                                    test_states = (ion.QHOState.from_potential(pot, mass = electron_mass, n = n) for n in range(50)),
                                    electric_potential = ion.Rectangle(start_time = 100 * asec, end_time = 150 * asec, amplitude = .1 * atomic_electric_field),
                                    animators = animators).to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())
