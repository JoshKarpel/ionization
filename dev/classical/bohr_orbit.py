import os
import logging

import numpy as np

import simulacra as si
from simulacra.units import *
import ionization as ion
import ionization.classical as cla


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO) as logger:
        initial_position = [(0, bohr_radius)]
        initial_velocity = [(alpha * c, 0)]
        test_mass = electron_mass_reduced
        test_charge = electron_charge

        force = cla.CoulombForce(position = [0, 0], charge = proton_charge)
        # force = cla.NoForce()

        spec = cla.ClassicalSpecification(
            'classical',
            initial_position,
            initial_velocity,
            test_mass = test_mass,
            test_charge = test_charge,
            force = force,
            time_initial = 0,
            time_final = 1000 * asec,
            time_step = .1 * asec,
            evolution_method = 'VV',
        )
        print(spec.info())

        sim = spec.to_simulation()
        print(sim.info())

        print('force at t= 0', force(0, position = np.array(initial_position), test_charge = test_charge) / N)

        sim.run_simulation(progress_bar = True)
        print(sim.info())

        sim.plot_particle_paths_2d(**PLOT_KWARGS)
