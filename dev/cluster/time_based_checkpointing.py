import logging
import os
import datetime

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        sim = ion.SphericalHarmonicSpecification(
            'info_test',
            r_bound = 100 * bohr_radius,
            r_points = 100 * 8, l_bound = 100,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_angular_momentum = 10,
            numeric_eigenstate_max_energy = 20 * eV,
            time_initial = 0, time_final = 1000 * asec, time_step = 1 * asec,
            electric_potential = ion.SineWave.from_photon_energy(10 * eV, amplitude = 1 * atomic_electric_field),
            store_data_every = -1,
            checkpoints = True,
            checkpoint_dir = OUT_DIR,
            checkpoint_every = datetime.timedelta(seconds = 5),
        ).to_sim()

        sim.info().log()
        sim.run()
        sim.info().log()
