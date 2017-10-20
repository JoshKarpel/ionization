import logging
import os
import functools

import numpy as np
import scipy.optimize as optim

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


def kernel(td, omega):
    return ide.hydrogen_kernel_LEN(td) * np.exp(1j * omega * td)


def alpha(eta):
    return (eta * electron_charge / hbar) ** 2


def analytic_b2(td, eta = 1 * atomic_electric_field * atomic_time, omega = ion.HydrogenBoundState(1, 0).energy / hbar):
    return np.abs((1 + (alpha(eta) * kernel(td, omega))) / ((1 + (alpha(eta) * kernel(0, omega))) ** 2)) ** 2


def limit(eta):
    return np.abs(1 / ((1 + (alpha(eta) * kernel(0, omega))) ** 2)) ** 2


def analytic_b2_m_lim(td, eta, omega):
    return analytic_b2(td, eta, omega) - limit(eta)


if __name__ == '__main__':
    with LOGMAN as logger:
        pts = 500
        omega = ion.HydrogenBoundState(1, 0).energy / hbar
        eta = 1 * atomic_electric_field * atomic_time

        td = np.linspace(0, 500, pts) * asec

        k = kernel(td, omega)
        norm = np.abs(kernel(0, omega))
        tau_alpha = 4 * atomic_time
        orbit_time = twopi * atomic_time

        ## FIND ROOTS
        imag_zero_at = optim.brentq(lambda td: np.imag(kernel(td * asec, omega)) / norm, 80, 110) * asec
        print(f'first imag zero of kernel is at {uround(imag_zero_at, asec)} as')
        print(f'ratio to tau alpha: {imag_zero_at / tau_alpha:.6f}')

        print()

        real_zero_at = optim.brentq(lambda td: np.real(kernel(td * asec, omega)) / norm, 140, 160) * asec
        print(f'first real zero of kernel is at {uround(real_zero_at, asec)} as')
        print(f'ratio to orbit time: {real_zero_at / orbit_time:.6f}')

        print()

        ## MAKE PLOT
        si.vis.xy_plot(
            'kernel',
            td,
            np.abs(k),
            np.real(k),
            np.imag(k),
            line_labels = [
                'A',
                'R',
                'I',
            ],
            x_label = r"$ t - t' $",
            x_unit = 'asec',
            y_label = r"$ K(t-t') $",
            vlines = [
                imag_zero_at,
                real_zero_at,
                44.005 * asec,
            ],
            **PLOT_KWARGS,
        )

        #######

        max_ionization_at = optim.minimize_scalar(
            lambda td: functools.partial(analytic_b2, eta = eta, omega = omega)(td * asec),
            bracket = [80, 100],
        ).x * asec
        print(f'max ionization at td = {uround(max_ionization_at, asec)} as')

        zero_diff_at = optim.brentq(
            lambda td: functools.partial(analytic_b2_m_lim, eta = eta, omega = omega)(td * asec),
            10, 100
        ) * asec
        print(f'zero diff ionization at td = {uround(zero_diff_at, asec)} as')
        print(kernel(0, omega))
        print(kernel(40 * asec, omega))
        print(kernel(zero_diff_at, omega))
        akv = alpha(eta) * kernel(zero_diff_at, omega)
        print(f'alpha * kernel at zero diff ionization = {akv}')
        print('should be close to zero', (akv.real ** 2) + (2 * akv.real) + (akv.imag ** 2))

        si.vis.xy_plot(
            'sine_ionization_variation',
            td,
            analytic_b2_m_lim(td, eta, omega),
            x_label = r"$ t - t' $",
            x_unit = 'asec',
            y_label = r'$\left| a(t_f) \right|^2$',
            vlines = [max_ionization_at, zero_diff_at],
            hlines = [limit(eta)],
            **PLOT_KWARGS,
        )
