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


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization',
                             stdout_level = logging.DEBUG)

if __name__ == '__main__':
    with logman as logger:
        sim = ion.SphericalHarmonicSpecification('velocity',
                                                 r_bound = 50 * bohr_radius,
                                                 r_points = 200, l_bound = 200,
                                                 evolution_gauge = 'VEL',
                                                 time_initial = 0, time_final = 500 * asec,
                                                 use_numeric_eigenstates = True,
                                                 numeric_eigenstate_max_angular_momentum = 10,
                                                 numeric_eigenstate_max_energy = 10 * eV,
                                                 electric_potential = ion.SineWave(1 / (100 * asec), amplitude = .1 * atomic_electric_field),
                                                 ).to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)

        # m = sim.mesh._get_interaction_hamiltonian_matrix_operators_without_field_LEN()
        # sim.mesh._make_split_operator_evolution_operators_LEN(m, 1)
        # m = sim.mesh._get_interaction_hamiltonian_matrix_operators_without_field_VEL()
        # print(m)
        #
        # mm = sim.mesh._make_split_operator_evolution_operators_VEL(m, 1)
