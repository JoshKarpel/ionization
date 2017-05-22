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

        sim.plot_test_state_overlaps_vs_time(target_dir = OUT_DIR)
        # sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)

        return sim


if __name__ == '__main__':
    with logman as logger:
        efield = ion.SineWave(twopi / (300 * asec), amplitude = .1 * atomic_electric_field,
                              window = ion.RectangularTimeWindow(on_time = -300 * asec, off_time = 300 * asec))

        # efield = ion.Rectangle(-100 * asec, -10 * asec, .1 * atomic_electric_field)
        # efield += ion.Rectangle(10 * asec, 100 * asec, -.1 * atomic_electric_field)

        """
        This script tests all of the evolution methods on each mesh.
        """

        import itertools as it
        import logging
        import os

        import simulacra as si
        from simulacra.units import *

        import ionization as ion


        FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
        OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

        logman = si.utils.LogManager('simulacra', 'ionization',
                                     stdout_level = logging.INFO,
                                     file_logs = True, file_level = logging.WARN, file_dir = OUT_DIR, file_name = FILE_NAME)

        if __name__ == '__main__':
            with logman as logger:
                x_bound = 100 * bohr_radius
                spacing = 1 * eV
                amp = .001 * atomic_electric_field
                t_bound = 5
                max_n = 10

                potential = ion.HarmonicOscillator.from_energy_spacing_and_mass(spacing, electron_mass)
                efield = ion.SineWave.from_photon_energy(spacing, amplitude = amp)
                efield.window = ion.SymmetricExponentialTimeWindow(window_time = (t_bound - 1) * efield.period, window_width = .1 * efield.period)

                line_spec_base = dict(
                    x_bound = x_bound, x_points = 2 ** 10,
                    internal_potential = potential,
                    electric_potential = efield,
                    initial_state = ion.QHOState.from_potential(potential, electron_mass),
                    test_states = tuple(ion.QHOState.from_potential(potential, electron_mass, n) for n in range(max_n + 1)),
                    time_initial = -t_bound * efield.period, time_final = t_bound * efield.period,
                    time_step = 5 * asec,
                    electric_potential_dc_correction = True,
                )

                specs = []

                for method, equations, gauge in it.product(
                        # ('CN', 'SO',),
                        ('CN',),
                        ('HAM',),
                        ('LEN', 'VEL')):
                            specs.append(
                                ion.LineSpecification(f'{gauge}__t_bound={t_bound}',
                                                      **line_spec_base,
                                                      evolution_method = method, evolution_equations = equations, evolution_gauge = gauge,
                                                      )
                            )

                results = si.utils.multi_map(run_sim, specs)
