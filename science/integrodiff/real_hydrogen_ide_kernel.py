import logging
import os
import functools

import numpy as np
import scipy.integrate as integ

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.integrodiff as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def gamma(k):
    return 1 / (bohr_radius * k)


N = (1 / twopi) ** 1.5


def imag_exp(k, td):
    return np.exp(-1j * td * (hbar * (k ** 2) / (2 * electron_mass_reduced)))


def abs_matrix_element_squared_from_hecht(k):
    prefactor = 512 * (pi ** 2) * (pi / 2) * (bohr_radius ** 5) * (N ** 2)

    g = gamma(k)

    first = (g ** 11) / ((1 + (g ** 2)) ** 5)
    second = np.exp(-4 * g * np.arctan2(1, g)) / (1 - np.exp(-twopi * g))

    return prefactor * first * second


def integrand_from_hecht(k, td = 0):
    return (k ** 2) * abs_matrix_element_squared_from_hecht(k) * imag_exp(k, td)


def integrand_from_bessels(k, td = 0):
    return (24 * (bohr_radius ** 7) * (k ** 4) / ((1 + ((k * bohr_radius) ** 2)) ** 6)) * imag_exp(k, td)


@np.vectorize
def integrate(integrand, td):
    # f = lambda k: integrand(k * twopi / bohr_radius, td)
    # return si.math.complex_quad(f, 0, np.inf)[0]
    k = (twopi / bohr_radius) * np.linspace(0, 1, 10000)[1:]
    return integ.simps(y = integrand(k, td), x = k)


if __name__ == '__main__':
    with LOGMAN as logger:
        k = (twopi / bohr_radius) * np.linspace(0, .5, 1000)[1:]
        # tds = np.array([0, 50, 100, 200, 300, 400, 600, 800, 1000]) * asec
        #
        # for td in tds:
        #     si.vis.xy_plot(
        #         f'integrands_vs_k__td={uround(td, asec)}',
        #         k,
        #         np.abs(integrand_from_bessels(k, td)),
        #         np.real(integrand_from_bessels(k, td)),
        #         np.imag(integrand_from_bessels(k, td)),
        #         np.abs(integrand_from_hecht(k, td)),
        #         np.real(integrand_from_hecht(k, td)),
        #         np.imag(integrand_from_hecht(k, td)),
        #         line_labels = [
        #             'Bessel ABS',
        #             'Bessel RE',
        #             'Bessel IM',
        #             'Hecht ABS',
        #             'Hecht RE',
        #             'Hecht IM',
        #         ],
        #         line_kwargs = [
        #             {'color': 'C0', 'linestyle': '-'},
        #             {'color': 'C0', 'linestyle': '--'},
        #             {'color': 'C0', 'linestyle': ':'},
        #             {'color': 'C1', 'linestyle': '-'},
        #             {'color': 'C1', 'linestyle': '--'},
        #             {'color': 'C1', 'linestyle': ':'},
        #         ],
        #         x_unit = twopi / bohr_radius,
        #         x_label = r'$k$',
        #         x_lower_limit = 0,
        #         y_unit = bohr_radius ** 2,
        #         y_label = r'$ \left| \left< k | \hat{z} | b \right> \right|^2 $',
        #         legend_kwargs = {'loc': 'upper right'},
        #         font_size_legend = 8,
        #         **PLOT_KWARGS
        #     )

        tds = np.linspace(0, 10000, 10000) * asec

        print(9 * pi / 32)

        print(integrate(integrand_from_bessels, 0 * asec) / (bohr_radius ** 2))
        print(integrate(integrand_from_hecht, 0 * asec) / (bohr_radius ** 2))

        omega_b = ion.HydrogenBoundState(1, 0).energy / hbar
        kernel_from_bessels = integrate(integrand_from_bessels, tds) * np.exp(1j * omega_b * tds)
        kernel_from_hecht = integrate(integrand_from_hecht, tds) * np.exp(1j * omega_b * tds)

        # si.vis.xy_plot(
        #     f'kernel',
        #     tds,
        #     np.abs(kernel_from_bessels),
        #     np.real(kernel_from_bessels),
        #     np.imag(kernel_from_bessels),
        #     np.abs(kernel_from_hecht),
        #     np.real(kernel_from_hecht),
        #     np.imag(kernel_from_hecht),
        #     line_labels = [
        #         'Bessel ABS',
        #         'Bessel RE',
        #         'Bessel IM',
        #         'Hecht ABS',
        #         'Hecht RE',
        #         'Hecht IM',
        #     ],
        #     line_kwargs = [
        #         {'color': 'C0', 'linestyle': '-'},
        #         {'color': 'C0', 'linestyle': '--'},
        #         {'color': 'C0', 'linestyle': ':'},
        #         {'color': 'C1', 'linestyle': '-'},
        #         {'color': 'C1', 'linestyle': '--'},
        #         {'color': 'C1', 'linestyle': ':'},
        #     ],
        #     x_unit = asec,
        #     x_label = r"$t-t'$",
        #     x_lower_limit = 0,
        #     x_upper_limit = 1 * fsec,
        #     y_unit = bohr_radius ** 2,
        #     y_label = r"$ K_b(t-t') $",
        #     legend_kwargs = {'loc': 'upper right'},
        #     font_size_legend = 8,
        #     **PLOT_KWARGS
        # )

        print(integ.simps(y = kernel_from_bessels, x = tds) / ((bohr_radius ** 2) * asec))
        print(integ.simps(y = kernel_from_hecht, x = tds) / ((bohr_radius ** 2) * asec))
