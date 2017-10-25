import logging
import os
import itertools

import numpy as np

import simulacra as si
from simulacra.units import *
import ionization as ion
from ionization import integrodiff as ide


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG)

PLOT_KWARGS = (
    dict(
            target_dir = OUT_DIR,
            img_format = 'png',
            fig_dpi_scale = 3,
    ),
    dict(
            target_dir = OUT_DIR,
    )
)


def run(spec):
    with logman as logger:
        sim = spec.to_simulation()

        logger.debug(sim.info())
        sim.run_simulation()
        logger.debug(sim.info())

        for kwargs in PLOT_KWARGS:
            sim.plot_b2_vs_time(name_postfix = f'{uround(sim.spec.time_step, asec, 3)}', **kwargs)

        return sim


if __name__ == '__main__':
    with logman as logger:
        specs = []

        pw = 100 * asec
        # flu = .1 * Jcm2

        # t_bound = 10
        # electric_pot = ion.SincPulse(pulse_width = pw, fluence = flu,
        #                              window = ion.SymmetricExponentialTimeWindow((t_bound - 2) * pw, .2 * pw))

        t_bound = 3
        electric_pot = ion.Rectangle(start_time = -5 * pw, end_time = 5 * pw, amplitude = 1 * atomic_electric_field,
                                     window = ion.SymmetricExponentialTimeWindow(pw / 2, .1 * pw, window_center = -1 * pw))
        electric_pot += ion.Rectangle(start_time = 0 * pw, end_time = 5 * pw, amplitude = -1 * atomic_electric_field,
                                      window = ion.SymmetricExponentialTimeWindow(pw / 2, .1 * pw, window_center = 1 * pw))

        q = electron_charge
        m = electron_mass_reduced
        L = bohr_radius

        spec_kwargs = dict(
            time_initial = -t_bound * pw, time_final = t_bound * pw, time_step = 1 * asec,
            prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2),
            electric_potential = electric_pot,
            evolution_method = 'FE',
            evolution_gauge = 'LEN',
            kernel = ide.gaussian_kernel_LEN, kernel_kwargs = dict(tau_alpha = 4 * m * (L ** 2) / hbar),
            # electric_potential_dc_correction = True,
        )

        for energy in [0, (hbar ** 2) / (2 * m * (L ** 2))]:
            specs.append(ide.IntegroDifferentialEquationSpecification(
                f'E={uround(energy, eV)}eV',
                test_energy = energy,
                **spec_kwargs,
            ))

        si.utils.multi_map(run, specs)
