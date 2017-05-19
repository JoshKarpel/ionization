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
                             stdout_level = logging.INFO)


def run_sim(spec):
    with logman as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        # sim.plot_test_state_overlaps_vs_time(target_dir = OUT_DIR)
        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)

        return sim


if __name__ == '__main__':
    with logman as logger:
        efield = ion.SineWave(1 / (100 * asec), amplitude = .1 * atomic_electric_field,
                              window = ion.SymmetricExponentialTimeWindow(300 * asec))

        spec_kwargs = dict(
            r_bound = 50 * bohr_radius,
            r_points = 200, l_bound = 50,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_angular_momentum = 10,
            numeric_eigenstate_max_energy = 10 * eV,
            electric_potential = efield,
            time_initial = -500 * asec, time_final = 500 * asec,
        )

        specs = []

        for gauge in ('LEN', 'VEL'):
            specs.append(ion.SphericalHarmonicSpecification(gauge,
                                                            evolution_gauge = gauge,
                                                            **spec_kwargs,
                                                            ))

        results = si.utils.multi_map(run_sim, specs)

        for r in results:
            logger.info(r.info())
