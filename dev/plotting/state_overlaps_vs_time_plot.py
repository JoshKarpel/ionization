import os
import logging

import simulacra as si
from simulacra.units import *
import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO) as logger:
        sim = ion.SphericalHarmonicSpecification(
                'test',
                r_bound = 50 * bohr_radius,
                r_points = 200,
                l_bound = 50,
                use_numeric_eigenstates = True,
                numeric_eigenstate_max_angular_momentum = 10,
                numeric_eigenstate_max_energy = 10 * eV,
                time_initial = -250 * asec, time_final = 250 * asec,
                electric_potential = ion.Rectangle(start_time = -200 * asec, end_time = 200 * asec, amplitude = .1 * atomic_electric_field)
        ).to_simulation()

        sim.info().log()
        sim.run(progress_bar = True)
        sim.info().log()

        PLOT_KWARGS = dict(
                target_dir = OUT_DIR,
        )

        sim.plot_state_overlaps_vs_time(name_postfix = 'None',
                                        states = None,
                                        **PLOT_KWARGS)

        sim.plot_state_overlaps_vs_time(name_postfix = 'Lambda',
                                        states = lambda state: state.bound,
                                        **PLOT_KWARGS)

        sim.plot_state_overlaps_vs_time(name_postfix = 'List',
                                        states = set(ion.HydrogenBoundState(n, l) for n in range(5) for l in range(n)),
                                        **PLOT_KWARGS)
