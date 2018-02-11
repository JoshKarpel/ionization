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

time.sleep(5)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        R, ppR, L, T = sys.argv[1:]

        R = int(R)
        ppR = int(ppR)
        L = int(L)
        T = int(T)

        sim = ion.SphericalHarmonicSpecification(
            f'ram_test__R={R}_ppR={ppR}_L={L}',
            r_bound = R * bohr_radius,
            r_points = R * ppR, l_bound = L,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_angular_momentum = 20,
            numeric_eigenstate_max_energy = 50 * eV,
            time_initial = 0, time_final = T * asec, time_step = 1 * asec,
            # electric_potential = ion.SineWave.from_photon_energy(10 * eV, amplitude = 1 * atomic_electric_field),
            store_data_every = 1,
        ).to_sim()
        sim.info().log()

        time.sleep(5)

        sim.run()

        sim.info().log()

        time.sleep(5)
