import logging
import os
import functools

from tqdm import tqdm

import numpy as np

import simulacra as si
import simulacra.units as u

import ionization as ion
import ionization.integrodiff as ide

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def theta_k_integral(k, alpha):
    kalpha = k * alpha

    pre = 2 / (kalpha ** 3)
    cos = 2 * kalpha * np.cos(kalpha)
    sin = ((kalpha ** 2) - 2) * np.sin(kalpha)

    return np.where(
        (k != 0) * (alpha != 0),
        pre * (cos + sin),
        2 / 3
    )


def kernel_integrand(k, time_difference, alpha):
    exp = np.exp(-1j * u.hbar * time_difference * (k ** 2) / (2 * u.electron_mass))
    frac = k ** 4 / ((1 + ((u.bohr_radius * k) ** 2)) ** 4)
    other = theta_k_integral(k, alpha)

    return exp * frac * other


def kernel(time_difference, alpha, omega_b = ion.HydrogenBoundState(1, 0).energy / u.hbar):
    prefactor = 8 * u.pi * (u.bohr_radius ** 3) * np.exp(1j * omega_b * time_difference)

    result, *errs = si.math.complex_quad(
        kernel_integrand,
        0,
        20 / u.bohr_radius,
        args = (time_difference, alpha),
    )

    return prefactor * result


if __name__ == '__main__':
    with LOGMAN as logger:
        tds = np.linspace(0, 500, 100) * u.asec
        alphas = np.linspace(0, 10, 100) * u.bohr_radius

        ks = np.linspace(0, 20, 1000) / u.bohr_radius

        si.vis.xy_plot(
            'theta_k_integral',
            ks,
            [theta_k_integral(k, 0) for k in ks],
            [theta_k_integral(k, .1 * u.bohr_radius) for k in ks],
            [theta_k_integral(k, .2 * u.bohr_radius) for k in ks],
            [theta_k_integral(k, 1 * u.bohr_radius) for k in ks],
            line_labels = ['0', '.1', '.2', '1'],
            x_unit = 'per_bohr_radius',
            x_label = 'k',
            **PLOT_KWARGS,
        )

        kern_integrand = kernel_integrand(ks, time_difference = 0, alpha = .1 * u.bohr_radius)

        si.vis.xy_plot(
            'integrand',
            ks,
            np.abs(kern_integrand),
            np.real(kern_integrand),
            np.imag(kern_integrand),
            x_unit = 'per_bohr_radius',
            **PLOT_KWARGS,
        )

        td_mesh, alpha_mesh = np.meshgrid(tds, alphas, indexing = 'ij')
        kernel_mesh = np.empty_like(td_mesh, dtype = np.complex128)
        for i, td in enumerate(tqdm(tds)):
            for j, alpha in enumerate(alphas):
                kernel_mesh[i, j] = kernel(td, alpha)

        kernel_norm = kernel_mesh[0, 0]
        print(kernel_norm)

        si.vis.xyz_plot(
            'kernel_vs_td_and_alpha',
            td_mesh,
            alpha_mesh,
            kernel_mesh / kernel_norm,
            x_label = r"$t-t'$",
            x_unit = 'asec',
            y_label = r"$\alpha(t, t')$",
            y_unit = 'bohr_radius',
            title = "Velocity-Gauge Hydrogen-Bessel Kernel $K_b^V(t,t')$",
            colormap = plt.get_cmap('richardson'),
            richardson_equator_magnitude = .5,
            **PLOT_KWARGS,
        )
