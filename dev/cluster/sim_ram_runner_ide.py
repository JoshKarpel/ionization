import time

time.sleep(5)

import sys
import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.integrodiff as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

time.sleep(5)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        dt, = sys.argv[1:]
        dt = float(dt) * asec

        pw = 100 * asec

        pulse = ion.SincPulse(pulse_width = pw * asec)

        test_width = 1 * bohr_radius
        test_charge = 1 * electron_charge
        test_mass = 1 * electron_mass_reduced

        steps = pw * 20 / dt

        sim = ide.IntegroDifferentialEquationSpecification(
            f'ram_test__dt={uround(dt, asec, 5)}as',
            time_initial = -pw * 10,
            time_final = pw * 10,
            time_step = dt,
            prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge),
            kernel = ide.gaussian_kernel_LEN,
            kernel_kwargs = {'tau_alpha': ide.gaussian_tau_alpha_LEN(test_width, test_mass)},
        ).to_simulation()
        sim.info().log()

        time.sleep(5)

        sim.run_simulation()

        sim.info().log()

        time.sleep(5)

        sim.save(target_dir = OUT_DIR)
