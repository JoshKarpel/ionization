import itertools
import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization',
                             stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 5,
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
    if sim.time_index % (sim.time_steps // 20) == 0 or sim.time_index == sim.time_steps - 1 or sim.time_index < 20:
        sim.mesh.plot_g_repr(name_postfix = f'_{sim.time_index}',
                             **PLOT_KWARGS)
        sim.mesh.plot_g_repr(name_postfix = f'__TEST__{sim.time_index}',
                             norm = si.vis.RichardsonNormalization(),
                             **PLOT_KWARGS)
        # print(f'index {sim.time_index}')
        # plot_g_1d(f'{sim.time_index}_g__{sim.spec.evolution_gauge}',
        #           sim.mesh.g,
        #           sim,
        #           **PLOT_KWARGS)
        # plot_g_1d(f'{sim.time_index}_g__{GAUGE_TO_OPP[sim.spec.evolution_gauge]}_from_{sim.spec.evolution_gauge}',
        #           sim.mesh.gauge_transformation(leaving_gauge = sim.spec.evolution_gauge),
        #           sim,
        #           **PLOT_KWARGS)


def run_sim(spec):
    with logman as logger:
        sim = spec.to_sim()

        sim.info().log()
        sim.run(callback = wrapped_plot_g_1d)
        sim.info().log()

        sim.plot_wavefunction_vs_time(
            **PLOT_KWARGS,
        )

        with open(os.path.join(OUT_DIR, f'{sim.file_name}.txt'), mode = 'w') as f:
            f.write(str(sim.info()))

        return sim


if __name__ == '__main__':
    with logman as logger:
        photon_energy = 1 * eV
        amp = .01 * atomic_electric_field
        t_bound = 2

        efield = ion.SineWave.from_photon_energy(photon_energy, amplitude = amp)
        efield.window = ion.SymmetricExponentialTimeWindow(window_time = (t_bound - 1) * efield.period_carrier, window_width = .1 * efield.period_carrier)

        spec_base = dict(
            r_bound = 50 * bohr_radius,
            electric_potential = efield,
            time_initial = -t_bound * efield.period_carrier, time_final = t_bound * efield.period_carrier,
            time_step = 1 * asec,
            electric_potential_dc_correction = True,
            store_electric_dipole_moment_expectation_value = True,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 10 * eV,
            numeric_eigenstate_max_angular_momentum = 5,
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
            for r_points, l_bound in itertools.product([200, 201], [30, 31]):
                specs.append(ion.SphericalHarmonicSpecification(
                    f'{gauge}__R={r_points}_L={l_bound}',
                    r_points = r_points,
                    l_bound = l_bound,
                    evolution_gauge = gauge,
                    **spec_base,
                ))

        results = si.utils.multi_map(run_sim, specs, processes = 3)

        si.vis.xxyy_plot(
            'dipole_moment',
            (r.data_times for r in results),
            (r.electric_dipole_moment_expectation_value_vs_time for r in results),
            line_labels = (r.name for r in results),
            line_kwargs = ({'linestyle': '-'}, {'linestyle': '--'}),
            x_label = r'Time $t$', x_unit = 'asec',
            y_label = r'Dipole Moment $d$', y_unit = 'atomic_electric_dipole_moment',
            **PLOT_KWARGS,
        )
