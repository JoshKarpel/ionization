import logging
import os
import functools

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 5,
)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG) as logger:
        pulse_width = 1000 * asec
        fluence = 20 * Jcm2
        phase = 0

        test_width = 1 * bohr_radius
        test_charge = 1 * electron_charge
        test_mass = 1 * electron_mass
        potential_depth = 36.831335 * eV

        internal_potential = ion.FiniteSquareWell(potential_depth = potential_depth, width = test_width)
        bound_state = ion.FiniteSquareWellState.from_potential(internal_potential, mass = electron_mass)
        # pulse = ion.GaussianPulse(pulse_width = pulse_width, fluence = fluence, phase = phase)
        pulse = ion.SincPulse(pulse_width = pulse_width, fluence = fluence, phase = phase)

        t_bound = 10 * pulse_width

        prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge)
        tau_alpha = ide.gaussian_tau_alpha_LEN(test_width, test_mass)
        kernel = functools.partial(ide.gaussian_kernel_LEN, tau_alpha = tau_alpha)
        omega_alpha = 1 / (2 * tau_alpha)

        sim = ide.DeltaKickSpecification(
            'delta_kick',
            time_initial = -t_bound,
            time_final = t_bound,
            test_width = test_width,
            test_mass = test_mass,
            test_charge = test_charge,
            test_energy = bound_state.energy,
            electric_potential = pulse,
            b_initial = 1,
            integral_prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge),
            kernel = ide.gaussian_kernel_LEN, kernel_kwargs = {'tau_alpha': ide.gaussian_tau_alpha_LEN(test_width, test_mass)},
        ).to_simulation()

        sim.run_simulation()

        for k in sim.kicks:
            print(k.time / asec, k.amplitude / (atomic_electric_field * atomic_time))

        # print(sim.data_times)
        # print(sim.a2)

        sim.plot_wavefunction_vs_time(**PLOT_KWARGS)

        sim.info().log()
