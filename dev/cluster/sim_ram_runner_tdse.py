import time

time.sleep(5)

import sys
import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    time.sleep(5)

    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        # R, ppR, L, T = sys.argv[1:]
        R = 100
        ppR = 10
        L = 200
        T = 1

        R = int(R)
        ppR = int(ppR)
        L = int(L)
        T = int(T)

        sim = ion.mesh.SphericalHarmonicSpecification(
            f'ram_test__R={R}_ppR={ppR}_L={L}',
            r_bound = R * bohr_radius,
            r_points = R * ppR, l_bound = L,
            # use_numeric_eigenstates = True,
            # numeric_eigenstate_max_angular_momentum = 5,
            # numeric_eigenstate_max_energy = 20 * eV,
            time_initial = 0,
            time_final = T * asec,
            time_step = 1 * asec,
            store_data_every = 1,
            evolution_method = ion.mesh.SplitInteractionOperator(),
        ).to_sim()
        sim.info().log()

        time.sleep(5)

        sim.run()

        sim.info().log()

        time.sleep(5)
