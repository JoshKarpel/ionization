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
                             stdout_level = logging.DEBUG)

if __name__ == '__main__':
    with logman as logger:
        efield = ion.SineWave.from_frequency(1 / (100 * asec), amplitude = .1 * atomic_electric_field)

        sim = ion.SphericalHarmonicSpecification('test',
                                                 r_bound = 50 * bohr_radius, r_points = 200,
                                                 l_bound = 50,
                                                 initial_state = ion.HydrogenBoundState(2, 0),
                                                 electric_potential = efield + efield,
                                                 time_initial = 0 * asec, time_final = 500 * asec, time_step = 1 * asec,
                                                 # test_states = tuple(ion.HydrogenBoundState(n, l) for n in range(5) for l in range(n)),
                                                 use_numeric_eigenstates = True,
                                                 numeric_eigenstate_max_energy = 10 * eV,
                                                 numeric_eigenstate_max_angular_momentum = 5,
                                                 ).to_simulation()

        path = sim.save(save_mesh = True, target_dir = OUT_DIR)

        # print(sim.spec.electric_potential.potentials)
        #
        # sim = si.Simulation.load(path)

        logger.info(sim.info())
        sim.run_simulation(progress_bar = False)
        logger.info(sim.info())

        sim.plot_wavefunction_vs_time(collapse_bound_state_angular_momenta = False, target_dir = OUT_DIR)
        # sim.plot_test_state_overlaps_vs_time(target_dir = OUT_DIR)
