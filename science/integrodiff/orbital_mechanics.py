import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=5)

time_field_unit = atomic_electric_field * atomic_time


# def new_orbit_period(eta):
#     f1 = (hbar ** 3) / (coulomb_constant * (proton_charge ** 2) * (bohr_radius ** 3) * (electron_mass_reduced ** 2))
#     f2 = 1 / ((hartree + (((hbar + (bohr_radius * proton_charge * eta)) ** 2) / (electron_mass_reduced * (bohr_radius ** 2)))) ** 3)
#
#     print(hbar, bohr_radius * proton_charge * eta)
#     print(electron_mass_reduced * (bohr_radius ** 2))
#     print('comp', hartree, (((hbar + (bohr_radius * proton_charge * eta)) ** 2) / (electron_mass_reduced * (bohr_radius ** 2))))
#     print(f1, f2)
#     return (pi / np.sqrt(2)) * np.sqrt(f1 * f2)


def new_energy(eta):
    # print('second_term (hartrees):', (.5 * ((hbar + (bohr_radius * proton_charge * eta)) ** 2) / (electron_mass_reduced * (bohr_radius ** 2))) / hartree)
    # print('total (hartrees):', (-hartree + (.5 * ((hbar + (bohr_radius * proton_charge * eta)) ** 2) / (electron_mass_reduced * (bohr_radius ** 2)))) / hartree)
    return -hartree + (
        0.5
        * ((hbar + (bohr_radius * proton_charge * eta)) ** 2)
        / (electron_mass_reduced * (bohr_radius ** 2))
    )


def new_semimajor(eta):
    # print('new semimajor', -coulomb_constant * (proton_charge ** 2) / (2 * new_energy(eta)) / bohr_radius)
    return -coulomb_constant * (proton_charge ** 2) / (2 * new_energy(eta))


def new_orbit_period(eta):
    pre = (
        4
        * (pi ** 2)
        * electron_mass_reduced
        / (coulomb_constant * (proton_charge ** 2))
    )

    return np.sqrt(pre * (new_semimajor(eta) ** 3))


def delta(eta):
    return (0.5 * (proton_charge ** 2) / electron_mass) * (eta ** 2)


def B(eta):
    return proton_charge * alpha * c * eta


def orbit_ratio(eta, theta, alpha):
    num = -rydberg
    den = -rydberg + delta(eta) - B(eta) * np.sin(theta) * np.cos(alpha)

    return (num / den) ** 1.5


def orbit_ratio_avg(eta, thetas, alphas):
    theta_mesh, alpha_mesh = np.meshgrid(thetas, alphas, indexing="ij")

    num = -rydberg
    den = -rydberg + delta(eta) - B(eta) * np.sin(theta_mesh) * np.cos(alpha_mesh)
    orbit_ratios = (num / den) ** 1.5

    return np.mean(orbit_ratios)


if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra", "ionization", stdout_level=logging.DEBUG
    ) as logger:
        # initial_orbit_period = h / hartree
        # # initial_orbit_period = twopi * (hbar ** 3) / ((coulomb_constant ** 2) * electron_mass * (electron_charge ** 4))
        # print(initial_orbit_period / asec)
        #
        # print('new', new_orbit_period(0 * time_field_unit) / asec)
        # print('new', new_orbit_period(.1 * time_field_unit) / asec)
        # print('new', new_orbit_period(.2 * time_field_unit) / asec)
        # print('new', new_orbit_period(.3 * time_field_unit) / asec)
        # print('new', new_orbit_period(.4 * time_field_unit) / asec)
        # # print('new', new_orbit_period(.5 * time_field_unit) / asec)
        # # print('new', new_orbit_period(1 * time_field_unit) / asec)

        eta = 0.01 * time_field_unit
        d = delta(eta)
        b = B(eta)
        print(d / eV, d / rydberg)
        print(b / eV, b / rydberg)
        print(b / d)

        thetas = np.linspace(0, pi, 200)
        alphas = np.linspace(0, twopi, 400)

        theta_mesh, alpha_mesh = np.meshgrid(thetas, alphas, indexing="ij")

        etas = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4]) * time_field_unit
        for eta in etas:
            si.vis.xyz_plot(
                f"eta={eta / time_field_unit:3f}au",
                theta_mesh,
                alpha_mesh,
                orbit_ratio(eta, theta_mesh, alpha_mesh),
                x_label=r"$\theta$",
                x_unit="rad",
                y_label=r"$\alpha$",
                y_unit="rad",
                z_label=fr"$T' / T$ at $\eta = {eta / time_field_unit:3f}$ a.u.",
                **PLOT_KWARGS,
            )

        ###### averaging
        n_points = 500
        thetas = np.linspace(0, pi, n_points)  # denser meshes
        alphas = np.linspace(0, twopi, n_points)

        etas = np.linspace(0, 0.4, 100) * time_field_unit

        avgs = list(orbit_ratio_avg(eta, thetas, alphas) for eta in etas)

        si.vis.xy_plot(
            "avg_ratio_vs_eta",
            etas,
            avgs,
            x_label=r"$\eta$ (a.u.)",
            x_unit=time_field_unit,
            y_label=r"avg. $T'/T$",
            **PLOT_KWARGS,
        )
