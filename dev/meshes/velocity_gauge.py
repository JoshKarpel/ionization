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

@si.utils.memoize
def alpha(j):
    x = (j ** 2) + (2 * j)
    return (x + 1) / (x + 0.75)

if __name__ == '__main__':
    with logman as logger:
        sim = ion.SphericalHarmonicSpecification('velocity',
                                                 r_bound = 1 * bohr_radius, r_points = 4, l_bound = 4,
                                                 ).to_simulation()

        m = sim.mesh._get_interaction_hamiltonian_matrix_operators_without_field_VEL()
        print(m)

        mm = sim.mesh._make_split_operator_evolution_operators_VEL(m, 1)
