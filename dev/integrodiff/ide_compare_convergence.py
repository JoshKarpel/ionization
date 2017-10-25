import logging
import os
import itertools

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.integrodiff as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def run(spec):
    with LOGMAN as logger:
        sim = spec.to_simulation()

        sim.run_simulation()

        sim.plot_b2_vs_time(**PLOT_KWARGS)

        return sim


if __name__ == '__main__':
    with LOGMAN as logger:
        dts = np.array([1]) * asec
        # methods = ['FE', 'BE', 'TRAP', 'RK4']
        methods = ['BE']

        tb = 4
        pulse = ion.GaussianPulse.from_number_of_cycles(
            pulse_width = 100 * asec,
            fluence = 1 * Jcm2,
            number_of_cycles = 2,
        )

        shared_spec_kwargs = dict(
            time_initial = -tb * pulse.pulse_width,
            time_final = tb * pulse.pulse_width,
            electric_potential = pulse,
            kernel = ide.hydrogen_kernel_LEN,
            kernel_kwargs = {'omega_b': ion.HydrogenBoundState(1, 0).energy / hbar},
        )

        specs = []
        for method, dt in itertools.product(methods, dts):
            specs.append(
                ide.IntegroDifferentialEquationSpecification(
                    f'{method}_dt={uround(dt, asec)}as',
                    evolution_method = method,
                    time_step = dt,
                    **shared_spec_kwargs
                )
            )

        results = si.utils.multi_map(run, specs, processes = 2)
        results = {(r.spec.evolution_method, r.spec.time_step): r for r in results}

