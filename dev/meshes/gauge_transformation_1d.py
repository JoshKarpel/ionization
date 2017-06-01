"""
This script tests all of the evolution methods on each mesh.
"""

import itertools as it
import logging
import os

import numpy as np
import scipy.sparse as sparse

import simulacra as si
from simulacra.units import *

import ionization as ion

import matplotlib.pyplot as plt


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization',
                             stdout_level = logging.INFO)

plt_kwargs = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 3,
)


def plot_g_1d(name, g, x, **kwargs):
    g_real = np.real(g)
    g_imag = np.imag(g)
    g_abs = np.abs(g)
    norm = np.nanmax(g_abs)

    si.vis.xy_plot(name,
                   x,
                   g_real / norm, g_imag / norm, g_abs / norm,
                   line_labels = ('Real g', 'Imag g', 'Abs g'),
                   line_kwargs = (None, None, {'linestyle': '--'}),
                   x_unit = 'bohr_radius',
                   y_lower_limit = -1, y_upper_limit = 1,
                   **kwargs)


GAUGE_TO_OPP = {
    'LEN': 'VEL',
    'VEL': 'LEN',
}


def wrapped_plot_g_1d(sim):
    if sim.time_index % (sim.time_steps // 6) == 0 or sim.time_index == sim.time_steps - 1:
        print(f'index {sim.time_index}')
        plot_g_1d(f'{sim.time_index}_g__{sim.spec.evolution_gauge}', sim.mesh.g, sim.mesh.x_mesh, **plt_kwargs)
        plot_g_1d(f'{sim.time_index}_g__{GAUGE_TO_OPP[sim.spec.evolution_gauge]}_from_{sim.spec.evolution_gauge}', sim.mesh.gauge_transformation(leaving_gauge = sim.spec.evolution_gauge), sim.mesh.x_mesh, **plt_kwargs)


def run_sim(spec):
    with logman as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation(callback = wrapped_plot_g_1d, progress_bar = True)
        logger.info(sim.info())

        sim.plot_state_overlaps_vs_time(
            **plt_kwargs,
        )

        # plot_g_1d(f'{sim.name}__g', sim.mesh.g, sim.mesh.x_mesh,
        #           **plt_kwargs, )
        #
        # g_transformed = sim.mesh.gauge_transformation(leaving_gauge = sim.spec.evolution_gauge)
        #
        # plot_g_1d(f'{sim.name}__g_transformed', g_transformed, sim.mesh.x_mesh,
        #           **plt_kwargs, )

        # sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)


if __name__ == '__main__':
    with logman as logger:
        x_bound = 50 * bohr_radius
        spacing = 1 * eV
        amp = .001 * atomic_electric_field
        t_bound = 2
        max_n = 10

        potential = ion.HarmonicOscillator.from_energy_spacing_and_mass(spacing, electron_mass)

        efield = ion.SineWave.from_photon_energy(spacing, amplitude = amp)
        efield.window = ion.SymmetricExponentialTimeWindow(window_time = (t_bound - 1) * efield.period, window_width = .1 * efield.period)

        # efield = ion.Rectangle(amplitude = 1 * atomic_electric_field, start_time = 50 * asec, end_time = 250 * asec)

        line_spec_base = dict(
            x_bound = x_bound, x_points = 2 ** 10,
            internal_potential = potential,
            electric_potential = efield,
            test_charge = electron_charge,
            initial_state = ion.QHOState.from_potential(potential, electron_mass),
            test_states = tuple(ion.QHOState.from_potential(potential, electron_mass, n) for n in range(max_n + 1)),
            time_initial = -t_bound * efield.period, time_final = t_bound * efield.period,
            # time_initial = 0, time_final = 300 * asec,
            time_step = 5 * asec,
            electric_potential_dc_correction = True,
            evolution_algorithm = 'CN',
            animators = [
                ion.animators.RectangleAnimator(
                    # length = 10,
                    length = 30,
                    fps = 30,
                    target_dir = OUT_DIR,
                    axman_wavefunction = ion.animators.LineMeshAxis(
                        norm = si.vis.AbsoluteRenormalize(),
                    ),
                    axman_lower = ion.animators.ElectricPotentialPlotAxis(
                        show_vector_potential = True,
                    )
                )
            ]
        )

        specs = []

        for gauge in ('LEN', 'VEL'):
            specs.append(
                ion.LineSpecification(f'{gauge}',
                                      **line_spec_base,
                                      evolution_gauge = gauge,
                                      )
            )

    results = si.utils.multi_map(run_sim, specs)
