"""
Bauer1999, Bauer2016 reference a 2.4 E^2 ionization rate. Can we get that?
"""

import logging
import os
import functools
import datetime

import numpy as np
import scipy.integrate as integ

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.integrodiff as ide

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, 'SIMLIB')

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

ANIMATOR_KWARGS = dict(
    target_dir = OUT_DIR,
    fig_dpi_scale = 1,
    length = 30,
    fps = 30,
)


def analytic():
    omega_b = ion.HydrogenBoundState(1, 0).energy / hbar
    m = electron_mass
    a = bohr_radius

    thingy = (a ** 2) * m * omega_b

    prefactor = 1j * 3 * (a ** 4) * m * pi / (16 * (((2 * (a ** 2) * m * omega_b) + hbar) ** 6))
    first = 96 * (thingy ** 5)
    second = 336 * (thingy ** 4) * hbar
    third = 560 * (thingy ** 3) * (hbar ** 2)
    fourth = 840 * (thingy ** 2) * (hbar ** 3)
    fifth = -210 * thingy * (hbar ** 4)
    sixth = -7 * (hbar ** 5)
    seventh = 512 * np.sqrt(2) * (a ** 3) * omega_b * np.sqrt(-(m ** 3) * omega_b * (hbar ** 7))

    # print(prefactor * first)
    # print(prefactor * second)
    # print(prefactor * third)
    # print(prefactor * fourth)
    # print(prefactor * fifth)
    # print(prefactor * sixth)
    # print(prefactor * seventh)

    return prefactor * (first + second + third + fourth + fifth + sixth + seventh)


def HCC_kernel_STATIC(td, efield_0):
    H_kernel = ide.hydrogen_kernel_LEN(td)

    pre = (electron_charge ** 2) / (6 * electron_mass * hbar)
    extra = np.exp(-1j * pre * (efield_0 ** 2) * (td ** 3))

    return H_kernel * extra


def HCC_kernel_SINE(td, efield_0, omega):
    H_kernel = ide.hydrogen_kernel_LEN(td)

    pre = (electron_charge ** 2) / (2 * electron_mass * hbar)
    first = ((efield_0 / omega) ** 2) * td
    second = -((efield_0 ** 2) / (omega ** 3)) * np.sin(omega * td)

    extra = np.exp(-1j * pre * (first + second))

    return H_kernel * extra


if __name__ == '__main__':
    with LOGMAN as logger:
        omega_b = ion.HydrogenBoundState(1, 0).energy / hbar
        tau_alpha = ide.gaussian_tau_alpha_LEN(bohr_radius, electron_mass)
        # print(tau_alpha / asec)
        # print(tau_alpha / atomic_time)
        G_kernel = functools.partial(ide.gaussian_kernel_LEN, tau_alpha = ide.gaussian_tau_alpha_LEN(bohr_radius, electron_mass))
        #
        # k_integral = si.math.complex_quad(kernel, 0, np.inf)[0]
        # # k_integral = si.math.complex_quadrature(kernel, 0, 1000 * fsec)[0]
        # print(k_integral)
        # print(k_integral / tau_alpha)
        #
        # print(kernel(0))
        # print(kernel(1000 * fsec))
        #
        time_diffs = np.linspace(0, 10 * fsec, 1e6)
        print('dt', (time_diffs[1] - time_diffs[0]) / asec)

        KG_from_simps = integ.simps(y = G_kernel(time_diffs) * np.exp(1j * omega_b * time_diffs),
                                    x = time_diffs)
        # print(G_kernel(0))
        # print('KG', KG_from_simps)
        # print('KG / tau_alpha', KG_from_simps / tau_alpha)
        # KG_abs_per_tau_alpha = np.abs(KG_from_simps / tau_alpha)
        # print(KG_abs_per_tau_alpha)

        H_kernel = ide.hydrogen_kernel_LEN

        H_kernel_prefactor = (bohr_radius ** 2)
        assert H_kernel(0) == H_kernel_prefactor
        KH_from_simps = integ.simps(y = H_kernel(time_diffs),
                                    x = time_diffs)

        KH_per_atomic_time_per_prefactor = KH_from_simps / atomic_time / H_kernel_prefactor
        KH_abs_per_atomic_time_per_prefactor = np.abs(KH_per_atomic_time_per_prefactor)
        print('KH / atomic time / kernel prefactor', KH_per_atomic_time_per_prefactor)
        print('Abs(KH) / atomic time / kernel prefactor', KH_abs_per_atomic_time_per_prefactor)

        overall_prefactor = -(electron_charge ** 4) / (16 * (pi ** 2) * (hbar ** 2) * (epsilon_0 ** 2) * (bohr_radius ** 4))
        gamma = overall_prefactor * KH_abs_per_atomic_time_per_prefactor * atomic_time * H_kernel_prefactor
        print('gamma', gamma)

        Gamma = 2 * gamma
        print('Gamma * atomic time', Gamma * atomic_time)

        si.vis.xy_plot(
            'visualize',
            time_diffs,
            np.real(H_kernel(time_diffs) / H_kernel_prefactor),
            np.real(G_kernel(time_diffs) * np.exp(1j * omega_b * time_diffs)),
            line_labels = [
                'H',
                'G',
            ],
            x_unit = 'fsec',
            # y_lower_limit = -1,
            # y_upper_limit = 1,
            **PLOT_KWARGS
        )

        # WITH EXTRA C-C TERM STATIC (SEE NOTEBOOK #3 P.103)

        # plot_times = np.linspace(0, 1, 1000) * fsec
        #
        # si.vis.xy_plot(
        #     'visualize_with_cc',
        #     plot_times,
        #     np.real(H_kernel(plot_times) / H_kernel_prefactor),
        #     np.real(G_kernel(plot_times) * np.exp(1j * omega_b * plot_times)),
        #     np.real(HCC_kernel(plot_times, 0) / H_kernel_prefactor),
        #     np.real(HCC_kernel(plot_times, .01 * atomic_electric_field) / H_kernel_prefactor),
        #     np.real(HCC_kernel(plot_times, .1 * atomic_electric_field) / H_kernel_prefactor),
        #     np.real(HCC_kernel(plot_times, .5 * atomic_electric_field) / H_kernel_prefactor),
        #     line_labels = [
        #         'H',
        #         'G',
        #         'HCC, 0',
        #         'HCC, 0.01',
        #         'HCC, 0.1',
        #         'HCC, 0.5',
        #     ],
        #     x_unit = 'fsec',
        #     # y_lower_limit = -1,
        #     # y_upper_limit = 1,
        #     legend_on_right = True,
        #     **PLOT_KWARGS
        # )


        print('-' * 40)

        efield_0s = np.array([0, .01, .1, .3, .5, 1, 2]) * atomic_electric_field

        for efield_0 in efield_0s:
            KHCC_from_simps = integ.simps(y = HCC_kernel_STATIC(time_diffs, efield_0 = efield_0),
                                          x = time_diffs)
            KHCC_per_atomic_time_per_prefactor = KHCC_from_simps / atomic_time / H_kernel_prefactor
            KHCC_abs_per_atomic_time_per_prefactor = np.abs(KHCC_per_atomic_time_per_prefactor)
            KHCC_re_per_atomic_time_per_prefactor = np.real(KHCC_per_atomic_time_per_prefactor)

            print('efield_0', uround(efield_0, atomic_electric_field))
            print('KHCC / atomic time / kernel prefactor', KHCC_per_atomic_time_per_prefactor)
            print('Abs(KHCC) / atomic time / kernel prefactor', KHCC_abs_per_atomic_time_per_prefactor)
            print('Re(KHCC) / atomic time / kernel prefactor', KHCC_re_per_atomic_time_per_prefactor)

            gamma = overall_prefactor * KHCC_re_per_atomic_time_per_prefactor * atomic_time * H_kernel_prefactor
            # print('gamma', gamma)

            Gamma = 2 * gamma
            print('Gamma * atomic time', Gamma * atomic_time)
            print()

        # WITH EXTRA C-C TERM SINE WAVE (SEE NOTEBOOK #3 P.104)
        print('-' * 40)

        efield_0s = np.array([0, .01, .1, .3, .5, 1, 2]) * atomic_electric_field
        omega = twopi * c / (800 * nm)

        for efield_0 in efield_0s:
            KHCC_from_simps = integ.simps(y = HCC_kernel_SINE(time_diffs, efield_0 = efield_0, omega = omega),
                                          x = time_diffs)
            KHCC_per_atomic_time_per_prefactor = KHCC_from_simps / atomic_time / H_kernel_prefactor
            KHCC_abs_per_atomic_time_per_prefactor = np.abs(KHCC_per_atomic_time_per_prefactor)
            KHCC_re_per_atomic_time_per_prefactor = np.real(KHCC_per_atomic_time_per_prefactor)

            print('efield_0', uround(efield_0, atomic_electric_field))
            print('KHCC / atomic time / kernel prefactor', KHCC_per_atomic_time_per_prefactor)
            print('Abs(KHCC) / atomic time / kernel prefactor', KHCC_abs_per_atomic_time_per_prefactor)
            print('Re(KHCC) / atomic time / kernel prefactor', KHCC_re_per_atomic_time_per_prefactor)

            gamma = overall_prefactor * KHCC_re_per_atomic_time_per_prefactor * atomic_time * H_kernel_prefactor
            # print('gamma', gamma)

            Gamma = 2 * gamma
            print('Gamma * atomic time', Gamma * atomic_time)
            print()
