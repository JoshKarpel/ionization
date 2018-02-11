import logging
import os
import functools
import itertools

from tqdm import tqdm

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
    fig_dpi_scale = 6,
)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.WARNING) as logger:
        t_bound_per_pw = 10

        # pulse_width = 200 * asec
        # fluence = .1 * Jcm2
        # phase = pi / 2

        pulse_widths = np.array([50, 100, 200, 400, 800]) * asec
        fluences = np.array([.01, .05, .1, .2, .3]) * Jcm2
        phases = np.linspace(0, pi, 100)

        test_width = 1 * bohr_radius
        test_charge = 1 * electron_charge
        test_mass = 1 * electron_mass
        potential_depth = 36.831335 * eV

        internal_potential = ion.FiniteSquareWell(potential_depth = potential_depth, width = test_width)
        bound_state = ion.FiniteSquareWellState.from_potential(internal_potential, mass = electron_mass)

        prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge)
        tau_alpha = ide.gaussian_tau_alpha_LEN(test_width, test_mass)
        kernel = functools.partial(ide.gaussian_kernel_LEN, tau_alpha = tau_alpha)
        omega_alpha = 1 / (2 * tau_alpha)

        for pulse_width, fluence in tqdm(itertools.product(pulse_widths, fluences)):
            # logger.info('Generating Specifications...')
            specs = []
            for phase in phases:
                pulse = ion.SincPulse(pulse_width = pulse_width, fluence = fluence, phase = phase)

                specs.append(ide.DeltaKickSpecification(
                    f'delta_kick__pw={uround(pulse_width, asec)}as_flu={uround(fluence, Jcm2)}jcm2_cep={uround(phase, pi)}pi',
                    time_initial = -t_bound_per_pw * pulse_width,
                    time_final = t_bound_per_pw * pulse_width,
                    test_width = test_width,
                    test_mass = test_mass,
                    test_charge = test_charge,
                    test_energy = bound_state.energy,
                    electric_potential = pulse,
                    b_initial = 1,
                    integral_prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge),
                    kernel = ide.gaussian_kernel_LEN, kernel_kwargs = {'tau_alpha': ide.gaussian_tau_alpha_LEN(test_width, test_mass)},
                ))

            # logger.info('Generating Simulations...')
            sims = []
            for spec in specs:
                sims.append(spec.to_simulation())

            # logger.info('Running Simulations...')
            for sim in sims:
                sim.run()

            for log in [True, False]:
                if log:
                    postfix = '__log'
                else:
                    postfix = ''

                si.vis.xy_plot(
                    f'phase_scan__pw={uround(pulse_width, asec)}as_flu={uround(fluence, Jcm2)}jcm2' + postfix,
                    [sim.spec.electric_potential.phase for sim in sims],
                    [sim.b2[-1] for sim in sims],
                    x_label = r'\varphi', x_unit = 'rad',
                    y_label = r'$\left| a \right|^2$',
                    y_log_axis = log,
                    title = rf'$ \tau = {uround(pulse_width, asec)} \; \mathrm{{as}}, \, H = {uround(fluence, Jcm2)} \; \mathrm{{J/cm^2}} $',
                    **PLOT_KWARGS,
                )
