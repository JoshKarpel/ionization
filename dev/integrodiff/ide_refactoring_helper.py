import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, 'SIMLIB')

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def run(spec):
    with LOGMAN as logger:
        sim = spec.to_simulation()

        sim.run_simulation()

        return sim


def run_from_lib(spec):
    with LOGMAN as logger:
        sim = si.utils.find_or_init_sim(spec, search_dir = SIM_LIB)

        if sim.status != si.Status.FINISHED:
            sim.run_simulation()
            sim.save(target_dir = SIM_LIB)

        return sim


if __name__ == '__main__':
    with LOGMAN as logger:
        dt = 1 * asec
        methods = [
            ion.ide.ForwardEulerMethod(),
            ion.ide.BackwardEulerMethod(),
            ion.ide.TrapezoidMethod(),
            ion.ide.RungeKuttaFourMethod(),
        ]

        pw = 100 * asec
        flu = 2 * Jcm2
        cep = 0
        tb = 4
        pulse = ion.potentials.GaussianPulse.from_number_of_cycles(
            pulse_width = pw,
            fluence = flu,
            phase = cep,
            number_of_cycles = 3,
        )

        shared_spec_kwargs = dict(
            time_initial = -tb * pulse.pulse_width,
            time_final = tb * pulse.pulse_width,
            time_step = dt,
            electric_potential = pulse,
            kernel = ion.ide.LengthGaugeHydrogenKernel(),
        )

        specs = []
        for method in methods:
            spec = ion.ide.IntegroDifferentialEquationSpecification(
                f'{method.__class__.__name__}',
                evolution_method = method,
                **shared_spec_kwargs
            )

            specs.append(spec)

        results = si.utils.multi_map(run, specs, processes = 3)
        for r in results:
            print(r.info())
            print()

        si.vis.xxyy_plot(
            'method_comparison',
            [
                *[r.data_times for r in results]
            ],
            [
                *[r.b2 for r in results]
            ],
            line_labels = [r.spec.evolution_method for r in results],
            x_unit = 'asec',
            x_label = r'$t$',
            **PLOT_KWARGS,
        )

        method_to_final_b2 = {r.spec.evolution_method.__class__: r.b2[-1] for r in results}
        expected = {
            ion.ide.ForwardEulerMethod: 0.11392725334866653,
            ion.ide.BackwardEulerMethod: 0.10407548854993905,
            ion.ide.TrapezoidMethod: 0.10891475259299933,
            ion.ide.RungeKuttaFourMethod: 0.10898054046019617
        }

        summary = 'Results:\n'
        summary += '\n'.join(f'{method.__name__}: {final_b2} | {expected[method]}' for method, final_b2 in method_to_final_b2.items())
        print(summary)

        for method in methods:
            np.testing.assert_allclose(method_to_final_b2[method.__class__], expected[method.__class__])
