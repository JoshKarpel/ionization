import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)


def run(spec):
    with log as logger:
        sim = spec.to_simulation()

        sim.run()

        sim.save(target_dir = OUT_DIR)


if __name__ == '__main__':
    with log as logger:
        specs = []

        for store in [1000, 100, 50, 20, 10, 5, 2, 1]:
            specs.append(ion.SphericalHarmonicSpecification('store={}'.format(store),
                                                            time_initial = 0 * asec, time_final = 10000 * asec, time_step = 1 * asec,
                                                            store_data_every = store,
                                                            r_bound = 100 * bohr_radius, r_points = 100 * 4, l_bound = 100,
                                                            use_numeric_eigenstates = True, numeric_eigenstate_max_energy = 50 * eV, numeric_eigenstate_max_angular_momentum = 20,
                                                            ))

        si.utils.multi_map(run, specs, processes = 2)
