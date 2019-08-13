import logging
import os
import functools

import numpy as np
import sympy as sym
import scipy.special as special

import simulacra as si
from simulacra.units import *
import ionization as ion
import ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

logman = si.utils.LogManager(
    "simulacra", "ionization", stdout_logs=True, stdout_level=logging.DEBUG
)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def run(spec):
    with logman as logger:
        sim = spec.to_sim()

        logger.debug(sim.info())
        sim.run()
        logger.debug(sim.info())

        sim.plot_b2_vs_time(**PLOT_KWARGS)
        # sim.plot_a2_vs_time(name_postfix = f'{uround(sim.spec.time_step, asec, 3)}', **PLOT_KWARGS)

        return sim


if __name__ == "__main__":
    sym.init_printing(use_unicode=True, wrap_line=False, no_global=True)

    with logman as logger:
        # wavenumber, a, m, t, hb = sym.symbols('wavenumber a m t hb')
        k = sym.Symbol("wavenumber", real=True)
        a = sym.Symbol("a", real=True, positive=True)
        m = sym.Symbol("m", real=True, positive=True)
        t = sym.Symbol("t", real=True, positive=True)
        hb = sym.Symbol("hb", real=True, positive=True)

        integrand = ((k ** 4) / ((1 + ((a * k) ** 2)) ** 6)) * (
            sym.exp(-sym.I * hb * (k ** 2) * t / (2 * m))
        )
        print(integrand)

        kernel_hydrogen_func = sym.integrate(integrand, (k, -sym.oo, sym.oo))
        print(kernel_hydrogen_func)

        kernel_hydrogen_func = sym.lambdify(
            (a, m, hb, t),
            kernel_hydrogen_func,
            modules=["numpy", {"erfc": special.erfc}],
        )
        print(kernel_hydrogen_func)

        td = np.linspace(0, 1000 * asec, 1000000)

        test_mass = 1 * electron_mass_reduced
        test_charge = 1 * electron_charge
        test_width = 1 * bohr_radius

        hyd_kern_prefactor = 128 * (bohr_radius ** 7) / (3 * (pi ** 2))
        gau_kern_prefactor = np.sqrt(pi) * (test_width ** 2)

        kernel_gaussian = gau_kern_prefactor * ide.gaussian_kernel_LEN(
            td, tau_alpha=ide.gaussian_tau_alpha_LEN(test_width, test_mass)
        )
        # kernel_gaussian = ide.gaussian_kernel_LEN(td, tau_alpha = ide.gaussian_tau_alpha_LEN(test_width, test_mass))
        # kernel_hydrogen = kernel_hydrogen(bohr_radius, electron_mass, hbar, td)
        kernel_hydrogen_func = functools.partial(
            kernel_hydrogen_func, bohr_radius, electron_mass, hbar
        )
        kernel_hydrogen_func.__name__ = "HydrogenKernel"
        kernel_hydrogen = hyd_kern_prefactor * kernel_hydrogen_func(td)
        # kernel_hydrogen = kernel_hydrogen_func(td)
        # kernel_hydrogen = kernel_hydrogen / np.abs(kernel_hydrogen[1])  # normalize for ease of use

        kernel_hydrogen_func_shifted = lambda t: kernel_hydrogen_func(t + 0.01 * asec)
        kernel_hydrogen_func_shifted.__name__ = "HydrogenKernelShifted"

        si.vis.xy_plot(
            "kernel_comparison",
            td,
            np.abs(kernel_gaussian),
            np.real(kernel_gaussian),
            np.imag(kernel_gaussian),
            np.abs(kernel_hydrogen),
            np.real(kernel_hydrogen),
            np.imag(kernel_hydrogen),
            line_labels=[
                r"Abs Gaussian",
                r"Re Gaussian",
                r"Im Gaussian",
                r"Abs Hydrogen",
                r"Re Hydrogen",
                r"Im Hydrogen",
            ],
            line_kwargs=[
                {"linestyle": "-", "color": "C0"},
                {"linestyle": "-", "color": "C1"},
                {"linestyle": "-", "color": "C2"},
                {"linestyle": "--", "color": "C0"},
                {"linestyle": "--", "color": "C1"},
                {"linestyle": "--", "color": "C2"},
            ],
            x_label=r"$t-t'$",
            x_unit="asec",
            y_label=r"$K(t-t')$",
            legend_on_right=True,
            vlines=[93 * asec, 150 * asec],
            **PLOT_KWARGS,
        )

        pulse = ion.SincPulse(pulse_width=200 * asec)

        shared_kwargs = dict(
            time_initial=-pulse.pulse_width * 10,
            time_final=pulse.pulse_width * 10,
            time_step=0.1 * asec,
            electric_potential=pulse,
            electric_potential_dc_correction=True,
        )

        specs = [
            ide.IntegroDifferentialEquationSpecification(
                "gaussian",
                kernel=ide.gaussian_kernel_LEN,
                kernel_kwargs={
                    "tau_alpha": ide.gaussian_tau_alpha_LEN(test_width, test_mass)
                },
                integral_prefactor=ide.gaussian_prefactor_LEN(test_width, test_charge),
                **shared_kwargs,
            ),
            ide.IntegroDifferentialEquationSpecification(
                "hydrogen",
                kernel=kernel_hydrogen_func_shifted,
                integral_prefactor=-((proton_charge / hbar) ** 2) * hyd_kern_prefactor,
                **shared_kwargs,
            ),
        ]

        results = list(run(spec) for spec in specs)
