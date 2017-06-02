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


def plot_g_1d(name, g, sim, **kwargs):
    si.vis.xyz_plot(name,
                    sim.mesh.l_mesh, sim.mesh.r_mesh, g,
                    colormap = plt.get_cmap('richardson'),
                    norm = si.vis.RichardsonNormalization(),
                    **kwargs
                    )


GAUGE_TO_OPP = {
    'LEN': 'VEL',
    'VEL': 'LEN',
}


def wrapped_plot_g_1d(sim):
    if sim.time_index % (sim.time_steps // 6) == 0 or sim.time_index == sim.time_steps - 1:
        print(f'index {sim.time_index}')
        plot_g_1d(f'{sim.time_index}_g__{sim.spec.evolution_gauge}', sim.mesh.g, sim, **plt_kwargs)
        plot_g_1d(f'{sim.time_index}_g__{GAUGE_TO_OPP[sim.spec.evolution_gauge]}_from_{sim.spec.evolution_gauge}', sim.mesh.gauge_transformation(leaving_gauge = sim.spec.evolution_gauge), sim, **plt_kwargs)


def run_sim(spec):
    with logman as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation(callback = wrapped_plot_g_1d, progress_bar = True)
        logger.info(sim.info())

        sim.plot_state_overlaps_vs_time(
            **plt_kwargs,
        )


if __name__ == '__main__':
    with logman as logger:
        efield = ion.Rectangle(amplitude = .5 * atomic_electric_field, start_time = 50 * asec, end_time = 250 * asec)

        spec_base = dict(
            r_bound = 50 * bohr_radius, r_points = 200, l_bound = 50,
            electric_potential = efield,
            test_charge = electron_charge,
            time_initial = 0, time_final = 300 * asec,
            time_step = 1 * asec,
            electric_potential_dc_correction = True,
            animators = [
                ion.animators.PolarAnimator(
                    # length = 10,
                    length = 30,
                    fps = 30,
                    target_dir = OUT_DIR,
                    axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                        norm = si.vis.AbsoluteRenormalize(),
                    ),
                )
            ]
        )

        specs = []

        for gauge in ('LEN', 'VEL'):
            specs.append(
                ion.SphericalHarmonicSpecification(f'{gauge}',
                                                   **spec_base,
                                                   evolution_gauge = gauge,
                                                   )
            )

    results = si.utils.multi_map(run_sim, specs)
