import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG, file_logs = False, file_mode = 'w', file_dir = OUT_DIR, file_name = 'log') as logger:
        bound = 100
        points_per_bohr_radius = 4
        l_max = 10
        energy = 20

        sim = ion.SphericalHarmonicSpecification('eig',
                                                 r_bound = bound * bohr_radius,
                                                 r_points = bound * points_per_bohr_radius,
                                                 l_bound = 50,
                                                 use_numeric_eigenstates = True,
                                                 numeric_eigenstate_max_energy = energy * eV,
                                                 numeric_eigenstate_max_angular_momentum = l_max,
                                                 ).to_simulation()

        print('max?:', ((bound * points_per_bohr_radius) - 2) * (l_max + 1))
        print('total:', len(sim.spec.test_states))
        bound = len(list(sim.bound_states))
        free = len(list(sim.free_states))
        print('{} free + {} bound = {} total'.format(free, bound, free + bound))
