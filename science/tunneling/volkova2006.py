import logging
import os

from tqdm import tqdm
from mpmath import mpf

import numpy as np
import scipy.integrate as integ

import simulacra as si
from simulacra.units import *
import ionization as ion

# import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, 'simlib')

logman = si.utils.LogManager('simulacra', 'ionization',
                             stdout_level = logging.INFO)

PLT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 5,
)


def instantaneous_tunneling_rate(electric_field_amplitude, ionization_potential = -rydberg):
    # f = np.abs(electric_field_amplitude / atomic_electric_field)
    #
    # return (4 / f) * (electron_mass_reduced * (proton_charge ** 4) / (hbar ** 3)) * np.exp(-(2 / 3) / f)

    amplitude_scaled = np.abs(electric_field_amplitude / atomic_electric_field)
    potential_scaled = np.abs(ionization_potential / hartree)

    f = amplitude_scaled / ((2 * potential_scaled) ** 1.5)

    # return (4 / f) * np.exp(-2 / (3 * f)) / atomic_time
    return 4 * np.sqrt(4 / (3 * f)) * np.exp(-2 / (3 * f)) / atomic_time

    # e_a = (electron_mass_reduced ** 2) * (proton_charge ** 5) / (((4 * pi * epsilon_0) ** 3) * (hbar ** 4))
    # w_a = (electron_mass_reduced * (proton_charge ** 4)) / (((4 * pi * epsilon_0) ** 2) * (hbar ** 3))
    # f = e_a / np.abs(electric_field_amplitude)

    # return 4 * w_a * f * np.exp(-2 * f / 3)


if __name__ == '__main__':
    with logman as logger:
        intensities = np.linspace((1 / 35 ** 2) * atomic_intensity, (1 / 10 ** 2) * atomic_intensity, 1e5)
        efields = np.sqrt(2 * intensities / (c * epsilon_0))

        tunneling_rates = instantaneous_tunneling_rate(efields)

        si.vis.xy_plot(
            'rate_vs_field',
            efields,
            tunneling_rates,
            x_label = fr'Electric Field $ {ion.LATEX_EFIELD}_0 $', x_unit = 'atomic_electric_field',
            y_label = fr'Tunneling Rate ($\mathrm{{s^{{-1}}}}$)',
            y_log_axis = True,
            ** PLT_KWARGS,
        )

        si.vis.xy_plot(
            'rate_vs_intensity',
            intensities,
            tunneling_rates,
            x_label = 'Intensity $P$', x_unit = 'atomic_intensity',
            y_label = fr'Tunneling Rate ($\mathrm{{s^{{-1}}}}$)',
            y_log_axis = True,
            **PLT_KWARGS,
        )

        si.vis.xy_plot(
            'rate_vs_inv_sqrt_intensity',
            1 / np.sqrt(intensities / atomic_intensity),
            tunneling_rates,
            x_label = '$ 1 / \sqrt{P / P_{\mathrm{atomic}}} $',
            y_label = fr'Tunneling Rate ($\mathrm{{s^{{-1}}}}$)',
            y_log_axis = True,
            **PLT_KWARGS,
        )
