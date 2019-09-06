import logging
import os

from tqdm import tqdm

import numpy as np
import scipy.integrate as integ

import simulacra as si
import simulacra.units as u

import ionization as ion

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def integrand_factory(c):
    def integrand(x):
        return ((x ** 4) / ((1 + (x ** 2)) ** 6)) * np.exp(-1j * c * (x ** 2))

    return integrand


def integral(c, d):
    result, *errors = si.math.complex_quad(integrand_factory(c), 0, 10)

    return result * np.exp(-1j * d)


def vector_potential_phase_factor(pulse, tp, t):
    prefactor = (u.electron_charge ** 2) / (2 * u.electron_mass * u.hbar)
    times = np.linspace(tp, t, 1e4)
    vp = pulse.get_vector_potential_amplitude_numeric_cumulative(times=times)

    result = integ.simps(y=vp ** 2, x=times)

    return result


def plot_integral_vs_c_and_d():
    c_vals = np.linspace(0, 10, 50)
    d_vals = np.linspace(0, 10, 50)

    c_mesh, f_mesh = np.meshgrid(c_vals, d_vals, indexing="ij")
    integral_mesh = np.empty_like(c_mesh, dtype=np.complex128)

    for i, c in enumerate(tqdm(c_vals)):
        for j, d in enumerate(d_vals):
            integral_mesh[i, j] = integral(c, d)

    integral_norm = integral(0, 0)

    si.vis.xyz_plot(
        "integral_vs_c_and_d",
        c_mesh,
        f_mesh,
        integral_mesh / integral_norm,
        x_label="c",
        y_label="d",
        colormap=plt.get_cmap("richardson"),
        **PLOT_KWARGS
    )


if __name__ == "__main__":
    with LOGMAN as logger:
        pulse = ion.potentials.GaussianPulse(pulse_width=100 * u.asec)
        times = np.linspace(-1000, 1000, 1000) * u.asec

        # phase_factor = np.

        # t_mesh, tp_mesh = np.meshgrid(times, times, indexing = 'ij')
        # phase_factor_mesh = np.empty_like(t_mesh, )
        # for i, t in enumerate(times):
        #     for j, tp in enumerate(times):

        plot_integral_vs_c_and_d()
