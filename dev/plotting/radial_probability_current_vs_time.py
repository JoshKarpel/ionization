import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *
import ionization as ion

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

if __name__ == '__main__':
    pulse = ion.Rectangle(start_time = 100 * asec, end_time = 300 * asec, amplitude = 2 * atomic_electric_field)

    sim = ion.SphericalHarmonicSpecification(
        'radial_current',
        r_bound = 50 * bohr_radius, r_points = 400, l_bound = 10,
        time_initial = 0 * asec, time_final = 500 * asec, time_step = 1 * asec,
        electric_potential = pulse,
        use_numeric_eigenstates = True,
        numeric_eigenstate_max_energy = 10 * eV,
        numeric_eigenstate_max_angular_momentum = 10,
        mask = ion.RadialCosineMask(inner_radius = 40 * bohr_radius, outer_radius = 50 * bohr_radius),
        # animators = [
        #     ion.animators.PolarAnimator(
        #         axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(),
        #         target_dir = OUT_DIR,
        #         fig_dpi_scale = 1,
        #         length = 10,
        #     )
        # ],
    ).to_simulation()

    total_radial_current_vs_time = np.empty((sim.time_steps, sim.spec.r_points), dtype = np.float64)

    def store_total_radial_current(sim):
        total_radial_current = np.sum(sim.mesh.get_radial_probability_current_mesh(), axis = 0)
        total_radial_current_vs_time[sim.time_index] = total_radial_current

    sim.run_simulation(progress_bar = True, callback = store_total_radial_current)

    z_lim = np.max(np.abs(total_radial_current_vs_time))
    t_mesh, r_mesh = np.meshgrid(sim.times, sim.mesh.r, indexing = 'ij')
    si.vis.xyz_plot(
        'radial_current_vs_time',
        t_mesh,
        r_mesh,
        total_radial_current_vs_time,
        x_label = r'Time $t$', x_unit = 'asec',
        y_label = r'Radius $r$', y_unit = 'bohr_radius',
        y_upper_limit = 5 * bohr_radius,
        z_log_axis = True,
        z_lower_limit = -z_lim, z_upper_limit = z_lim,
        colormap = plt.get_cmap('seismic'),
        title = r'Radial Probability Current vs. Time',
        **PLOT_KWARGS,
    )
