import logging
import os

from tqdm import tqdm

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion
import ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

logman = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(
    target_dir=OUT_DIR,
    # img_format = 'png',
    # fig_dpi_scale = 3,
)


def find_time_between_maxima(pulse):
    """For a sine-like pulse!"""
    times = np.linspace(-5 * pulse.pulse_width, 5 * pulse.pulse_width, 10000)
    field = pulse.get_electric_field_amplitude(times)

    return np.abs(2 * times[np.argmax(np.abs(field))])


if __name__ == "__main__":
    with logman as logger:
        p = ion.potentials.SincPulse(pulse_width=200 * asec, phase=pi / 2)
        times = np.linspace(-100 * asec, 100 * asec, 1000)
        si.vis.xy_plot(
            "test",
            times,
            p.get_electric_field_amplitude(times),
            x_unit="asec",
            **PLOT_KWARGS,
        )

        tau_alpha = ide.gaussian_tau_alpha_LEN(
            test_width=1 * bohr_radius, test_mass=electron_mass
        )
        dt = np.linspace(0, 3 * tau_alpha, 1000)
        k = ide.gaussian_kernel_LEN(time_difference=dt, tau_alpha=tau_alpha)

        si.vis.xy_plot(
            "kernel",
            dt,
            np.abs(k),
            np.real(k),
            np.imag(k),
            x_unit="asec",
            **PLOT_KWARGS,
        )
        imag_min_t = dt[np.argmin(np.imag(k))]
        print(tau_alpha / asec)
        print(imag_min_t / asec)
        print(imag_min_t / tau_alpha)

        pulse_widths = np.linspace(50, 200, 1000) * asec
        time_between_maxima = np.empty_like(pulse_widths)

        for ii, pulse_width in enumerate(tqdm(pulse_widths)):
            pulse = ion.potentials.SincPulse(pulse_width=pulse_width, phase=pi / 2)
            time_between_maxima[ii] = find_time_between_maxima(pulse)

        si.vis.xy_plot(
            "time_between_maxima",
            pulse_widths,
            time_between_maxima,
            pulse_widths,
            pulse_widths / 2,
            line_kwargs=[
                None,
                {"linestyle": "--", "color": "black"},
                {"linestyle": "--", "color": "black"},
            ],
            hlines=[tau_alpha, imag_min_t],
            hline_kwargs=[
                {"linestyle": "--", "color": "C3"},
                {"linestyle": "--", "color": "C4"},
            ],
            x_label=r"$\tau$",
            x_unit="asec",
            y_label=r"$\Delta t$",
            y_unit="asec",
            y_pad=0,
            title=r"Time Between Sine-Like Pulse Maxima vs. $\tau$",
            **PLOT_KWARGS,
        )
