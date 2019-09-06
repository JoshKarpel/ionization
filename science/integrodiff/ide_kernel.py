import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *

import ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=5)

if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra", "ionization", stdout_logs=True, stdout_level=logging.DEBUG
    ) as logger:
        tau_alpha = 4 * electron_mass * (bohr_radius ** 2) / hbar
        print(f"tau alpha: {tau_alpha / asec:3f}")

        time_diffs = np.linspace(0, 10 * tau_alpha, 1000)
        kernel = ide.gaussian_kernel_LEN(time_diffs, tau_alpha=tau_alpha)

        comp_cosh = 1 / np.cosh(time_diffs / tau_alpha)

        si.vis.xy_plot(
            "kernel_vs_time_diff",
            time_diffs,
            np.abs(kernel),
            np.real(kernel),
            np.imag(kernel),
            comp_cosh,
            line_labels=[
                r"$ \left|K\right| $",
                r"$ \mathrm{Re} \, K $",
                r"$ \mathrm{Im} \, K $",
                "$ 1 / \cosh(t-t') $",
            ],
            x_label=r"$ t - t' $",
            x_unit="asec",
            y_label=r"$ f(t-t') $",
            **PLOT_KWARGS,
        )
