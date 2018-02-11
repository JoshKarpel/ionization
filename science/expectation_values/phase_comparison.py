import os

import numpy as np

import simulacra as si
import simulacra.cluster as clu
from simulacra.units import *

import ionization as ion
import ionization.cluster as iclu

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization')

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def run(spec):
    with LOGMAN as logger:
        sim = spec.to_simulation()

        sim.info().log()
        sim.run()
        sim.info().log()

        sim.plot_wavefunction_vs_time(show_vector_potential = False, **PLOT_KWARGS)
        sim.plot_radial_position_expectation_value_vs_time(**PLOT_KWARGS)
        sim.plot_dipole_moment_expectation_value_vs_time(**PLOT_KWARGS)
        sim.plot_energy_expectation_value_vs_time(**PLOT_KWARGS)

        return sim


if __name__ == '__main__':
    with LOGMAN as logger:
        initial_state = ion.HydrogenBoundState(1, 0)

        pulse_width = 92 * asec
        fluence = 1 * Jcm2
        phases = [0, pi / 4, pi / 2, 3 * pi / 4, pi]

        specs = []
        for phase in phases:
            pulse = ion.SincPulse(pulse_width = pulse_width, fluence = fluence, phase = phase,
                                  window = ion.SymmetricExponentialTimeWindow(window_time = 9 * pulse_width, window_width = .2 * pulse_width))

            specs.append(ion.SphericalHarmonicSpecification(
                # fr'cep={uround(phase, pi)}pi',
                f'pw={uround(pulse_width, asec)}as_flu={uround(fluence, Jcm2)}jcm2_cep={uround(phase, pi)}pi',
                r_bound = 100 * bohr_radius,
                r_points = 400,
                l_bound = 100,
                time_initial = -10 * pulse_width,
                time_final = 10 * pulse_width,
                time_step = 1 * asec,
                use_numeric_eigenstates = True,
                numeric_eigenstate_max_energy = 30 * eV,
                numeric_eigenstate_max_angular_momentum = 10,
                electric_potential = pulse,
                electric_potential_dc_correction = True,
                initial_state = initial_state,
                # animators = [
                #     ion.animators.PolarAnimator(
                #         postfix = '__g_wavefunction',
                #         axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                #             which = 'g',
                #             colormap = plt.get_cmap('richardson'),
                #             norm = si.vis.RichardsonNormalization(),
                #             plot_limit = 30 * bohr_radius,
                #         ),
                #         axman_lower_right = ion.animators.ElectricPotentialPlotAxis(
                #             show_electric_field = True,
                #             show_vector_potential = False,
                #             show_y_label = False,
                #             show_ticks_right = True,
                #         ),
                #         axman_upper_right = ion.animators.WavefunctionStackplotAxis(states = [initial_state]),
                #         axman_colorbar = None,
                #         target_dir = OUT_DIR,
                #         fig_dpi_scale = 2,
                #         length = 30,
                #         fps = 30,
                #     ),
                # ]
            ))

        results = si.utils.multi_map(run, specs, processes = 2)
        tex_labels = [r'$' + r.name[r.name.find('cep'):].replace('cep', r'\varphi').replace('pi', r'\pi') + '$' for r in results]

        si.vis.xxyy_plot(
            f'comparison__pw={uround(pulse_width, asec)}as_flu={uround(fluence, Jcm2)}jcm2__initial_state_overlap',
            [*[r.data_times for r in results]],
            [*[r.state_overlaps_vs_time[r.spec.initial_state] for r in results]],
            line_labels = tex_labels,
            x_label = '$t$', x_unit = 'asec',
            y_label = r'Initial State Overlap',
            title = 'Initial State Overlap',
            **PLOT_KWARGS,
        )

        si.vis.xxyy_plot(
            f'comparison__pw={uround(pulse_width, asec)}as_flu={uround(fluence, Jcm2)}jcm2__bound_state_overlap',
            [*[r.data_times for r in results]],
            [*[sum(r.state_overlaps_vs_time[state] for state in r.bound_states) for r in results]],
            line_labels = tex_labels,
            x_label = '$t$', x_unit = 'asec',
            y_label = r'Initial State Overlap',
            title = 'Initial State Overlap',
            **PLOT_KWARGS,
        )

        si.vis.xxyy_plot(
            f'comparison__pw={uround(pulse_width, asec)}as_flu={uround(fluence, Jcm2)}jcm2__radial_position',
            [*[r.data_times for r in results]],
            [*[r.radial_position_expectation_value_vs_time for r in results]],
            line_labels = tex_labels,
            x_label = '$t$', x_unit = 'asec',
            y_label = r'$ \left\langle r(t) \right\rangle $', y_unit = bohr_radius,
            title = 'Radial Position',
            **PLOT_KWARGS,
        )

        si.vis.xxyy_plot(
            f'comparison__pw={uround(pulse_width, asec)}as_flu={uround(fluence, Jcm2)}jcm2__dipole_moment',
            [*[r.data_times for r in results]],
            [*[r.electric_dipole_moment_expectation_value_vs_time for r in results]],
            line_labels = tex_labels,
            x_label = '$t$', x_unit = 'asec',
            y_label = r'$ \left\langle d(t) \right\rangle $', y_unit = atomic_electric_dipole_moment,
            title = 'Dipole Moment',
            **PLOT_KWARGS,
        )

        si.vis.xxyy_plot(
            f'comparison__pw={uround(pulse_width, asec)}as_flu={uround(fluence, Jcm2)}jcm2__internal_energy',
            [*[r.data_times for r in results]],
            [*[r.internal_energy_expectation_value_vs_time for r in results]],
            line_labels = tex_labels,
            x_label = '$t$', x_unit = 'asec',
            y_label = r'$ \left\langle E(t) \right\rangle $', y_unit = eV,
            title = 'Internal Energy',
            **PLOT_KWARGS,
        )

        si.vis.xxyy_plot(
            f'comparison__pw={uround(pulse_width, asec)}as_flu={uround(fluence, Jcm2)}jcm2__total_energy',
            [*[r.data_times for r in results]],
            [*[r.total_energy_expectation_value_vs_time for r in results]],
            line_labels = tex_labels,
            x_label = '$t$', x_unit = 'asec',
            y_label = r'$ \left\langle E(t) \right\rangle $', y_unit = eV,
            title = 'Total Energy',
            **PLOT_KWARGS,
        )

