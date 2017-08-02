import logging
import os
import functools
import collections

from tqdm import tqdm

import numpy as np
import scipy.integrate as integrate

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.integrodiff as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

time_field_unit = atomic_electric_field * atomic_time


def single_kick(beta):
    return (1 + beta) ** 2


def double_kick_no_interference(beta):
    return (1 + beta) ** 4


def double_kick(beta, delta_t, omega_alpha, tau_alpha):
    """previous a is 1 + beta"""
    kernel = ide.gaussian_kernel_LEN(delta_t, tau_alpha = tau_alpha)

    pre = (1 + beta) ** 2
    kernel_squared = (beta * np.abs(kernel)) ** 2
    interference = -2 * beta * (1 + beta) * np.real(kernel * np.exp(1j * omega_alpha * delta_t))

    return pre * (pre + kernel_squared + interference)


def double_kick_v2(beta, delta_t, omega_alpha, tau_alpha):
    """previous a is 1 + beta / 2"""
    kernel = ide.gaussian_kernel_LEN(delta_t, tau_alpha = tau_alpha)

    first = (1 + beta) ** 4
    second = (beta ** 2) * ((1 + (beta / 2)) ** 2) * (np.abs(kernel) ** 2)
    third = 2 * beta * (1 + (beta / 2)) * ((1 + beta) ** 2) * np.real(kernel * np.exp(1j * omega_alpha * delta_t))

    return first + second - third

def double_kick_v3(beta, delta_t, omega_alpha, tau_alpha):
    """previous a is 1"""
    kernel = ide.gaussian_kernel_LEN(delta_t, tau_alpha = tau_alpha)

    first = (1 + beta) ** 4
    second = (beta ** 2) * (np.abs(kernel) ** 2)
    third = 2 * beta * ((1 + beta) ** 2) * np.real(kernel * np.exp(1j * omega_alpha * delta_t))

    return first + second - third


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG) as logger:
        delta_t = np.linspace(0, 500, 500) * asec
        omega_alpha = ion.HydrogenBoundState(1).energy / hbar
        eta = .2 * time_field_unit

        tau_alpha = ide.gaussian_tau_alpha_LEN(bohr_radius, electron_mass)
        B = ide.gaussian_prefactor_LEN(bohr_radius, electron_charge)
        beta = B * (eta ** 2)

        dbk = double_kick(beta, delta_t, omega_alpha, tau_alpha)

        si.vis.xy_plot(
            'delta_t_scan',
            delta_t,
            np.ones_like(delta_t) * double_kick_no_interference(beta),
            np.ones_like(delta_t) * single_kick(beta),
            double_kick(beta, delta_t, omega_alpha, tau_alpha),
            double_kick_v2(beta, delta_t, omega_alpha, tau_alpha),
            double_kick_v3(beta, delta_t, omega_alpha, tau_alpha),
            np.ones_like(delta_t) * single_kick(beta * (1.5 ** 2)),
            line_labels = [
                'double kick (no int.)',
                'single kick',
                'double kick',
                'double kick v2',
                'double kick v3',
                'single stronger kick'
            ],
            x_label = r'$\Delta t$', x_unit = 'asec',
            y_label = r'$ \left| a_{\alpha} \right|^2 $',
            legend_on_right = True,
            **PLOT_KWARGS,
        )

        # find minimum
        min_delta_t = delta_t[np.argmin(dbk)]
        print(min_delta_t / asec)

