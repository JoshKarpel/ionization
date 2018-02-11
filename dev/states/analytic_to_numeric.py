import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        bound = 100
        points_per_bohr_radius = 4

        sim = ion.SphericalHarmonicSpecification('test',
                                                 r_bound = bound * bohr_radius,
                                                 r_points = bound * points_per_bohr_radius,
                                                 l_bound = 50,
                                                 initial_state = ion.HydrogenBoundState(1, 0),
                                                 use_numeric_eigenstates = True,
                                                 numeric_eigenstate_max_energy = 10 * eV,
                                                 numeric_eigenstate_max_angular_momentum = 5,
                                                 ).to_sim()

        print(sim.info())
        print()

        # for k, v in sim.mesh.analytic_to_numeric.items():
        #     print(f'{k} -> {v}')

        print()
        a_state = ion.HydrogenBoundState(1, 0)
        print(a_state)
        print(sim.mesh.get_g_for_state(a_state))
        print((a_state.radial_function(sim.mesh.r) * sim.mesh.g_factor)[:30])

        dipole_moment = np.abs(sim.mesh.z_dipole_moment_inner_product(mesh_a = sim.mesh.get_g_for_state(ion.HydrogenBoundState(2, 1))))
        print(dipole_moment)

        print()

        n_state = sim.mesh.analytic_to_numeric[a_state]
        print(n_state)
        print(sim.mesh.get_g_for_state(n_state))
        print((n_state.g * sim.mesh.g_factor)[:30])
        print(np.abs((n_state.g * sim.mesh.g_factor)[:30]))
