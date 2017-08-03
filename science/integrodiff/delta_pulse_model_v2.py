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


def get_cosine_and_sine_etas(pulse_width = 200 * asec, fluence = 1 * Jcm2):
    cos_pulse = ion.SincPulse(pulse_width = pulse_width, fluence = fluence, phase = 0)
    sin_pulse = ion.SincPulse(pulse_width = pulse_width, fluence = fluence, phase = pi / 2)

    times = np.linspace(-35 * pulse_width, 35 * pulse_width, 10000)
    cos_pulse = ion.DC_correct_electric_potential(cos_pulse, times)
    sin_pulse = ion.DC_correct_electric_potential(sin_pulse, times)

    cos_kicks = ide.decompose_potential_into_kicks__amplitude(cos_pulse, times)
    sin_kicks = ide.decompose_potential_into_kicks__amplitude(sin_pulse, times)

    max_cos_kick = max(k.amplitude for k in cos_kicks)
    max_sin_kick = max(k.amplitude for k in sin_kicks)

    return max_cos_kick, max_sin_kick


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG) as logger:
        delta_t = np.linspace(0, 500, 500) * asec
        omega_alpha = ion.HydrogenBoundState(1).energy / hbar
        eta = .2 * time_field_unit

        tau_alpha = ide.gaussian_tau_alpha_LEN(bohr_radius, electron_mass)
        B = ide.gaussian_prefactor_LEN(bohr_radius, electron_charge)

        cos_eta, sin_eta = get_cosine_and_sine_etas(pulse_width = 200 * asec, fluence = .001 * Jcm2)

        print(cos_eta / time_field_unit, sin_eta / time_field_unit)

        cos_beta = B * (cos_eta ** 2)
        sin_beta = B * (sin_eta ** 2)

        si.vis.xy_plot(
            'delta_t_scan',
            delta_t,
            np.ones_like(delta_t) * double_kick_no_interference(sin_beta),
            np.ones_like(delta_t) * single_kick(sin_beta),
            double_kick(sin_beta, delta_t, omega_alpha, tau_alpha),
            double_kick_v2(sin_beta, delta_t, omega_alpha, tau_alpha),
            double_kick_v3(sin_beta, delta_t, omega_alpha, tau_alpha),
            np.ones_like(delta_t) * single_kick(cos_beta),
            line_labels = [
                'double sine kick (no int.)',
                'single sine kick',
                'sine kick',
                'sine kick v2',
                'sine kick v3',
                'cosine kick'
            ],
            x_label = r'$\Delta t$', x_unit = 'asec',
            y_label = r'$ \left| a_{\alpha} \right|^2 $',
            legend_on_right = True,
            **PLOT_KWARGS,
        )

        # find minimum
        dbk = double_kick(sin_beta, delta_t, omega_alpha, tau_alpha)
        min_delta_t = delta_t[np.argmin(dbk)]
        print(min_delta_t / asec)
