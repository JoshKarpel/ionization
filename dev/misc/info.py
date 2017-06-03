import os
import itertools

import numpy as np

import simulacra as si
from simulacra.units import *
import ionization as ion

import matplotlib as mpl


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization') as logger:
        pot = ion.Rectangle(start_time = 10 * asec, end_time = 50 * asec)
        pot += ion.SineWave.from_photon_energy(1 * eV, window = ion.SymmetricExponentialTimeWindow())
        pot += ion.SincPulse(window = ion.RectangularTimeWindow())

        spec = ion.SphericalHarmonicSpecification('test',
                                                  electric_potential = pot,
                                                  mask = ion.RadialCosineMask(),
                                                  use_numeric_eigenstates = True,
                                                  numeric_eigenstate_max_energy = 10 * eV,
                                                  numeric_eigenstate_max_angular_momentum = 5)
        print()
        logger.info(spec.info())
        print()

        sim = spec.to_simulation()

        print()
        logger.info(sim.info())
