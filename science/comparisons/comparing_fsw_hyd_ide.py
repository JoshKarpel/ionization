import logging
import os

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.integrodiff as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 2,
)


def run(spec):
    with logman as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        return sim


if __name__ == '__main__':
    with logman as logger:
        pw = 200 * asec
        flu = 1 * Jcm2
        phase = 0
        t_bound = 10

        efield = ion.SincPulse(pulse_width = pw, fluence = flu, phase = phase,
                               window = ion.SymmetricExponentialTimeWindow(window_time = (t_bound - 2) * pw, window_width = .2 * pw))

        test_width = 1 * bohr_radius
        test_charge = 1 * electron_charge
        test_mass = 1 * electron_mass
        potential_depth = 36.831335 * eV

        internal_potential = ion.FiniteSquareWell(potential_depth = potential_depth, width = test_width)

        shared_kwargs = dict(
            test_width = test_width,
            test_charge = test_charge,
            test_mass = test_mass,
            potential_depth = potential_depth,
            electric_potential = efield,
            time_initial = -t_bound * pw,
            time_final = t_bound * pw,
            time_step = 1 * asec,
            electric_potential_dc_correction = True,
            x_bound = 200 * bohr_radius,
            x_points = 2 ** 12,
            r_bound = 200 * bohr_radius,
            r_points = 1000,
            l_bound = 100,
            mask = ion.RadialCosineMask(inner_radius = 180 * bohr_radius, outer_radius = 200 * bohr_radius),
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 10 * eV,
            numeric_eigenstate_max_angular_momentum = 10,
            time_step_minimum = .05 * asec,
            time_step_maximum = 10 * asec,
            epsilon = 1e-3,
            analytic_eigenstate_type = ion.FiniteSquareWellState,
        )

        specs = [
            ion.LineSpecification(
                'line_len',
                internal_potential = internal_potential,
                initial_state = ion.FiniteSquareWellState.from_potential(internal_potential, mass = test_mass),
                store_data_every = 5,
                evolution_gauge = 'LEN',
                **shared_kwargs,
            ),
            ion.LineSpecification(
                'line_vel',
                internal_potential = internal_potential,
                initial_state = ion.FiniteSquareWellState.from_potential(internal_potential, mass = test_mass),
                store_data_every = 5,
                evolution_gauge = 'VEL',
                **shared_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                'hyd_len',
                store_data_every = 5,
                evolution_gauge = 'LEN',
                **shared_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                'hyd_vel',
                store_data_every = 5,
                evolution_gauge = 'VEL',
                **shared_kwargs,
            ),
            ide.IntegroDifferentialEquationSpecification(
                'ide_len',
                prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge),
                kernel = ide.gaussian_kernel_LEN,
                kernel_kwargs = {'tau_alpha': ide.gaussian_tau_alpha_LEN(test_width, test_mass)},
                evolution_gauge = 'LEN',
                evolution_method = 'ARK4',
                **shared_kwargs,
            ),
            ide.IntegroDifferentialEquationSpecification(
                'ide_vel',
                prefactor = ide.gaussian_prefactor_VEL(test_width, test_charge, test_mass),
                kernel = ide.gaussian_kernel_VEL,
                kernel_kwargs = {'tau_alpha': ide.gaussian_tau_alpha_VEL(test_width, test_mass),
                                 'width': test_width},
                evolution_gauge = 'VEL',
                evolution_method = 'ARK4',
                **shared_kwargs,
            )
        ]

        results = si.utils.multi_map(run, specs, processes = 3)

        for r in results:
            r.plot_wavefunction_vs_time(**PLOT_KWARGS)
            print(r.info())
