import logging
import os
import itertools

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 5,
)


def run_spec(spec):
    with LOGMAN as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        sim.plot_wavefunction_vs_time(show_vector_potential = False, **PLOT_KWARGS)

    return sim


if __name__ == '__main__':
    with LOGMAN as logger:
        num_random_pulses = 6

        pw = 200 * asec
        flu = 1 * Jcm2

        r_bound = 100

        pulse_bound = 10
        time_bound = 12

        window = ion.SymmetricExponentialTimeWindow(window_time = pulse_bound * pw, window_width = .2 * pw)

        cosine_pulse = ion.potentials.SincPulse(pulse_width = pw, fluence = flu, phase = 0, window = window)
        sine_pulse = ion.potentials.SincPulse(pulse_width = pw, fluence = flu, phase = pi / 2, window = window)

        shared_kwargs = dict(
            r_bound = r_bound * bohr_radius,
            r_points = r_bound * 4,
            l_bound = 100,
            time_initial = -time_bound * pw,
            time_final = time_bound * pw,
            time_step = 1 * asec,
            mask = ion.RadialCosineMask(inner_radius = (r_bound * .8) * bohr_radius, outer_radius = r_bound * bohr_radius),
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 10 * eV,
            numeric_eigenstate_max_angular_momentum = 10,
            electric_potential_dc_correction = True,
            store_data_every = 20,
        )

        specs = [
            ion.SphericalHarmonicSpecification(
                'cosine',
                electric_potential = cosine_pulse,
                **shared_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                'sine',
                electric_potential = sine_pulse,
                **shared_kwargs,
            )
        ]

        cosine_sim = specs[0].clone().to_simulation()
        times = cosine_sim.times

        print(specs)

        rand_pulses = []
        for i in range(num_random_pulses):
            rand_pulses.append(ion.GenericElectricPotential.from_pulse(
                cosine_pulse,
                times,
                phase_function = lambda f: si.math.rand_phase(f.shape),
                window = window,
            ))

        ion.potentials.plot_electric_field_amplitude_vs_time(
            'random_pulses',
            times,
            *rand_pulses,
            **PLOT_KWARGS,
        )

        all_pulses = list(itertools.chain([cosine_pulse, sine_pulse], rand_pulses))

        correlations = []
        for pulse in all_pulses:
            field = pulse.get_electric_field_amplitude(times)
            print(field)
            correlations.append(np.correlate(field, field, 'same'))

        print(correlations)

        si.vis.xy_plot(
            'correlations',
            times,
            *correlations,
            line_labels = ['cosine', 'sine', *range(num_random_pulses)],
            line_kwargs = [None, {'linestyle': '--'}],
            x_unit = 'asec', x_label = r'$\Delta t$',
            **PLOT_KWARGS
        )

        si.vis.xy_plot(
            'correlations_zoom',
            times,
            *correlations,
            line_labels = ['cosine', 'sine', *range(num_random_pulses)],
            line_kwargs = [None, {'linestyle': '--'}],
            x_unit = 'asec', x_label = r'$\Delta t$',
            x_lower_limit = -300 * asec, x_upper_limit = 300 * asec,
            **PLOT_KWARGS
        )

        # for name, pulse in enumerate(rand_pulses):
        #     specs.append(
        #         ion.SphericalHarmonicSpecification(
        #             f'rand_{name}',
        #             electric_potential = pulse,
        #             **shared_kwargs,
        #         )
        #     )
        #
        # for spec in specs:
        #     print(spec.info())
        #
        # results = si.utils.multi_map(run_spec, specs, processes = 7)
        #
        # for r in results:
        #     print(r.info())
        #
        # print('\n' * 3)
        #
        # for r in results:
        #     print(r.name, r.state_overlaps_vs_time[r.spec.initial_state][-1])
        #
        # si.vis.xxyy_plot(
        #     'comparison',
        #     [r.data_times for r in results],
        #     [r.state_overlaps_vs_time[r.spec.initial_state] for r in results],
        #     line_labels = [r.name for r in results],
        #     x_label = r'$t$', x_unit = 'asec',
        #     y_label = 'Initial State Overlap',
        #     legend_on_right = True,
        #     **PLOT_KWARGS,
        # )
        #
        #
