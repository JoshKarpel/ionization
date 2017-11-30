import logging
import os

import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.integrodiff as ide

from bauer_1999 import BauerGaussianPulse

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def plot_length_gauge_hydrogen_bessel_kernel_heatmap(times):
    t_mesh, tp_mesh = np.meshgrid(times, times, indexing = 'ij')

    hydrogen_kernel_length_gauge_norm = ide.hydrogen_kernel_LEN(0)
    si.vis.xyz_plot(
        'kernel_heatmap_len',
        t_mesh,
        tp_mesh,
        ide.hydrogen_kernel_LEN(t_mesh - tp_mesh) / hydrogen_kernel_length_gauge_norm,
        x_label = r"$t$",
        x_unit = 'asec',
        y_label = r"$t'$",
        y_unit = 'asec',
        title = "Length-Gauge Hydrogen-Bessel Kernel $K_b^L(t-t')$",
        colormap = plt.get_cmap('richardson'),
        **PLOT_KWARGS,
    )


if __name__ == '__main__':
    with LOGMAN as logger:
        pulse = BauerGaussianPulse()

        times = np.linspace(-1 * pulse.pulse_center, 3 * pulse.pulse_center, 1000)

        vector_potential = pulse.get_vector_potential_amplitude_numeric_cumulative(times)
        vector_potential = interp.CubicSpline(x = times, y = vector_potential, bc_type = 'clamped')

        quiver_motion = (electron_charge / electron_mass) * integ.cumtrapz(y = vector_potential(times), x = times, initial = 0)
        quiver_motion = interp.CubicSpline(x = times, y = quiver_motion, bc_type = 'natural')

        si.vis.xy_plot(
            'fields',
            times,
            pulse.get_electric_field_amplitude(times) / atomic_electric_field,
            # pulse.get_vector_potential_amplitude_numeric_cumulative(times) * proton_charge / atomic_momentum,
            vector_potential(times) * proton_charge / atomic_momentum,
            quiver_motion(times) / bohr_radius,
            line_labels = [r'$\mathcal{E}(t)$', r'$e \, \mathcal{A}(t, t_0)$', r'$\alpha(t,t_0)$'],
            x_unit = 'asec',
            x_label = r'$t$',
            y_label = 'Field Amplitudes',
            **PLOT_KWARGS
        )

        plot_length_gauge_hydrogen_bessel_kernel_heatmap(times)
