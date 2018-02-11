import logging
import os
import itertools

import numpy as np

import simulacra as si
import simulacra.units as u

import ionization as ion
import ionization.ide as ide

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
        sim = spec.to_sim()

        sim.run()

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
        dt = 1 * u.asec
        methods = [
            ide.ForwardEulerMethod(),
            ide.BackwardEulerMethod(),
            ide.TrapezoidMethod(),
            ide.RungeKuttaFourMethod(),
        ]
        kernels = [
            ide.LengthGaugeHydrogenKernel(),
            ide.ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction(),
        ]

        pw = 100 * u.asec
        flu = 2 * u.Jcm2
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
        )

        specs = []
        for method, kernel in itertools.product(methods, kernels):
            spec = ide.IntegroDifferentialEquationSpecification(
                f'{method.__class__.__name__}',
                evolution_method = method,
                kernel = kernel,
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

        method_to_final_b2 = {(r.spec.evolution_method.__class__, r.spec.kernel.__class__): r.b2[-1] for r in results}
        expected = {
            (ide.ForwardEulerMethod, ide.LengthGaugeHydrogenKernel): 0.11392725334866653,
            (ide.ForwardEulerMethod, ide.ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction): 0.23874503549999987,
            (ide.BackwardEulerMethod, ide.LengthGaugeHydrogenKernel): 0.10407548854993905,
            (ide.BackwardEulerMethod, ide.ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction): 0.22880463049055752,
            (ide.TrapezoidMethod, ide.LengthGaugeHydrogenKernel): 0.10891475259299933,
            (ide.TrapezoidMethod, ide.ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction): 0.23370674421984355,
            (ide.RungeKuttaFourMethod, ide.LengthGaugeHydrogenKernel): 0.10898054046019617,
            (ide.RungeKuttaFourMethod, ide.ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction): 0.23337076999249678,
        }

        summary = '\nResults:\n'
        summary += '\n'.join(f'{method.__name__} + {kernel.__name__}: {final_b2} | {expected[method, kernel]}' for (method, kernel), final_b2 in method_to_final_b2.items())
        print(summary)

        for method, kernel in itertools.product(methods, kernels):
            key = (method.__class__, kernel.__class__)
            np.testing.assert_allclose(
                method_to_final_b2[key],
                expected[key]
            )

        print('\nAll good!')
