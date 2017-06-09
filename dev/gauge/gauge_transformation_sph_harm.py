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

PLT_KWARGS = dict(
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
        # print(f'index {sim.time_index}')
        plot_g_1d(f'{sim.time_index}_g__{sim.spec.evolution_gauge}',
                  sim.mesh.g,
                  sim,
                  **PLT_KWARGS)
        # plot_g_1d(f'{sim.time_index}_g__{GAUGE_TO_OPP[sim.spec.evolution_gauge]}_from_{sim.spec.evolution_gauge}',
        #           sim.mesh.gauge_transformation(leaving_gauge = sim.spec.evolution_gauge),
        #           sim,
        #           **PLT_KWARGS)


def run_sim(spec):
    with logman as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())

        sim.run_simulation(callback = wrapped_plot_g_1d)

        logger.info(sim.info())

        sim.plot_wavefunction_vs_time(
            **PLT_KWARGS,
        )

        return sim

if __name__ == '__main__':
    with logman as logger:
        photon_energy = 1 * eV
        amp = .01 * atomic_electric_field
        t_bound = 3

        efield = ion.SineWave.from_photon_energy(photon_energy, amplitude = amp)
        efield.window = ion.SymmetricExponentialTimeWindow(window_time = (t_bound - 1) * efield.period, window_width = .1 * efield.period)

        r_bound = 50
        ppbr = 4

        spec_base = dict(
            r_bound = r_bound * bohr_radius, r_points = ppbr * r_bound, l_bound = 50,
            electric_potential = efield,
            time_initial = -t_bound * efield.period, time_final = t_bound * efield.period,
            time_step = 5 * asec,
            electric_potential_dc_correction = True,
            store_electric_dipole_moment_expectation_value = True,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 10 * eV,
            numeric_eigenstate_max_angular_momentum = 10,
            # animators = [
            #     ion.animators.PolarAnimator(
            #         length = 30,
            #         fps = 30,
            #         target_dir = OUT_DIR,
            #         axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
            #             norm = si.vis.AbsoluteRenormalize(),
            #         ),
            #     )
            # ]
        )

        specs = []
        for gauge in ('LEN', 'VEL'):
            specs.append(ion.SphericalHarmonicSpecification(
                f'{gauge}',
                **spec_base,
                evolution_gauge = gauge,
            ))

        results = si.utils.multi_map(run_sim, specs, processes = 2)

        si.vis.xxyy_plot(
            'dipole_moment',
            (r.data_times for r in results),
            (r.electric_dipole_moment_expectation_value_vs_time for r in results),
            line_labels = (r.name for r in results),
            line_kwargs = ({'linestyle': '-'}, {'linestyle': '--'}),
            x_label = r'Time $t$', x_unit = 'asec',
            y_label = r'Dipole Moment $d$', y_unit = 'atomic_electric_dipole_moment',
            **PLT_KWARGS,
        )
