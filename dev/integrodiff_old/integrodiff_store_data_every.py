import logging
import os

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.ide as ide


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        # electric_field = ion.Rectangle(start_time = -500 * asec, end_time = 500 * asec, amplitude = 1 * atomic_electric_field)
        electric_field = ion.SincPulse(pulse_width = 100 * asec, fluence = 1 * Jcm2)

        q = electron_charge
        m = electron_mass_reduced
        L = bohr_radius

        tau_alpha = 4 * m * (L ** 2) / hbar
        prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

        dt = 1
        t_bound = 1000

        # sde = 50

        for sde in (1, -1, 2, 5, 25, 50):
            sim = ide.IntegroDifferentialEquationSpecification(f'sde={sde}',
                                                               time_initial = -t_bound * asec, time_final = t_bound * asec, time_step = dt * asec,
                                                               integral_prefactor = prefactor,
                                                               electric_potential = electric_field,
                                                               kernel = ide.gaussian_kernel_LEN, kernel_kwargs = dict(tau_alpha = tau_alpha),
                                                               store_data_every = sde,
                                                               ).to_simulation()

            sim.info().log()
            sim.run_simulation(progress_bar = True)
            sim.info().log()

            sim.save(target_dir = OUT_DIR)

            sim.plot_b2_vs_time(target_dir = OUT_DIR, img_format = 'png', fig_dpi_scale = 3)

        for sde in (1, -1, 2, 5, 25, 50):
            sim = ide.AdaptiveIntegroDifferentialEquationSpecification(f'a_sde={sde}',
                                                                       time_initial = -t_bound * asec, time_final = t_bound * asec,
                                                                       prefactor = prefactor,
                                                                       electric_potential = electric_field,
                                                                       kernel = ide.gaussian_kernel_LEN, kernel_kwargs = dict(tau_alpha = tau_alpha),
                                                                       store_data_every = sde,
                                                                       ).to_simulation()

            sim.info().log()
            sim.run_simulation(progress_bar = True)
            sim.info().log()

            sim.save(target_dir = OUT_DIR)

            sim.plot_b2_vs_time(target_dir = OUT_DIR, img_format = 'png', fig_dpi_scale = 3)
