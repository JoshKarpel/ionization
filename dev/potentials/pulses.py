import logging
import os
import itertools

from tqdm import tqdm

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=5)

T_PLOT = 10
T_BOUND = 13


def compare_pulse_types(pulse_types, pulse_widths, fluences, phases, omega_mins):
    total = len(pulse_widths) * len(fluences) * len(phases) * len(omega_mins)
    for pw, flu, phase, omega_min in tqdm(
        itertools.product(pulse_widths, fluences, phases, omega_mins), total=total
    ):
        name = f"PulseComparison___PW={pw / asec:3f}as__FLU={flu / Jcm2:3f}jcm2__PHA={phase / pi:3f}pi__OmegaMin=2pi_x_{omega_min / twopi * THz:.3f}THz"

        kwargs = dict(pulse_width=pw, fluence=flu, phase=phase, omega_min=omega_min)

        pulses = list(
            pulse_type.from_omega_min(
                **kwargs,
                window=ion.potentials.LogisticWindow(
                    window_time=10 * pw, window_width=0.2 * pw
                ),
            )
            for pulse_type in pulse_types
        )
        times = np.linspace(-T_BOUND * pw, T_BOUND * pw, 10000)

        si.vis.xy_plot(
            name,
            times,
            *[
                pulse.get_electric_field_amplitude(times)
                / np.max(pulse.get_electric_field_amplitude(times))
                for pulse in pulses
            ],
            line_labels=[pulse.__class__.__name__ for pulse in pulses],
            x_label=r"$t$ ($\tau$)",
            x_unit=pw,
            y_label=fr"$ {ion.vis.LATEX_EFIELD}(t) $ (fraction of max)",
            x_lower_limit=-T_PLOT * pw,
            x_upper_limit=T_PLOT * pw,
            y_lower_limit=-1,
            y_upper_limit=1,
            **PLOT_KWARGS,
        )


def compare_omega_mins(pulse_types, pulse_widths, fluences, phases, omega_mins):
    total = len(pulse_types) * len(pulse_widths) * len(fluences) * len(phases)
    for pulse_type, pw, flu, phase in tqdm(
        itertools.product(pulse_types, pulse_widths, fluences, phases), total=total
    ):
        name = f"OmegaMinComparison___PulseType={pulse_type.__name__}__PW={pw / asec:3f}as__FLU={flu / Jcm2:3f}jcm2__PHA={phase / pi:3f}pi"

        kwargs = dict(pulse_width=pw, fluence=flu, phase=phase)

        pulses = list(
            pulse_type.from_omega_min(
                **kwargs,
                omega_min=omega_min,
                window=ion.potentials.LogisticWindow(
                    window_time=10 * pw, window_width=0.2 * pw
                ),
            )
            for omega_min in omega_mins
        )
        times = np.linspace(-T_BOUND * pw, T_BOUND * pw, 10000)

        si.vis.xy_plot(
            name,
            times,
            *[
                pulse.get_electric_field_amplitude(times)
                / np.max(pulse.get_electric_field_amplitude(times))
                for pulse in pulses
            ],
            line_labels=[
                rf'$ \omega_{{\min}} = 2\pi \, \times \, {omega_min / twopi * THz:.3f} \; \mathrm{{THz}} $'
                for omega_min in omega_mins
            ],
            x_label=r"$t$ ($\tau$)",
            x_unit=pw,
            y_label=fr"$ {ion.vis.LATEX_EFIELD}(t) $ (fraction of max)",
            x_lower_limit=-T_PLOT * pw,
            x_upper_limit=T_PLOT * pw,
            y_lower_limit=-1,
            y_upper_limit=1,
            legend_on_right=True,
            **PLOT_KWARGS,
        )


def compare_omega_mins_movie(pulse_types, pulse_widths, fluences, phases, omega_mins):
    total = len(pulse_types) * len(pulse_widths) * len(fluences) * len(phases)
    for pulse_type, pw, flu, phase in tqdm(
        itertools.product(pulse_types, pulse_widths, fluences, phases), total=total
    ):
        kwargs = dict(pulse_width=pw, fluence=flu, phase=phase)

        times = np.linspace(-T_BOUND * pw, T_BOUND * pw, 10000)

        def get_field_by_omega_min(times, f_min):
            pulse = pulse_type.from_omega_min(
                **kwargs,
                omega_min=twopi * f_min,
                window=ion.potentials.LogisticWindow(
                    window_time=10 * pw, window_width=0.2 * pw
                ),
            )
            field = pulse.get_electric_field_amplitude(times)
            return field / np.max(field)

        def get_dc_corrected_field_by_omega_min(times, f_min):
            pulse = pulse_type.from_omega_min(
                **kwargs,
                omega_min=twopi * f_min,
                window=ion.potentials.LogisticWindow(
                    window_time=10 * pw, window_width=0.2 * pw
                ),
            )
            pulse = ion.DC_correct_electric_potential(pulse, times)
            field = pulse.get_electric_field_amplitude(times)
            return field / np.max(field)

        si.vis.xyt_plot(
            f"OmegaMinComparison_Movie___PulseType={pulse_type.__name__}__PW={pw / asec:3f}as__FLU={flu / Jcm2:3f}jcm2__PHA={phase / pi:3f}pi",
            times,
            omega_mins / twopi,
            get_dc_corrected_field_by_omega_min,
            get_field_by_omega_min,
            line_labels=["DC-Corrected", "Base Pulse"],
            line_kwargs=[{"color": "C0"}, {"color": "C1", "linestyle": "--"}],
            x_label=r"$t$ ($\tau$)",
            x_unit=pw,
            y_label=fr"$ {ion.vis.LATEX_EFIELD}(t) $ (fraction of max)",
            x_lower_limit=-T_PLOT * pw,
            x_upper_limit=T_PLOT * pw,
            y_lower_limit=-1,
            y_upper_limit=1,
            t_unit="THz",
            t_fmt_string=r"$\omega_{{\min}} = 2\pi \, \times \, {} \; {}$",
            length=10,
            **PLOT_KWARGS,
        )


if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra", "ionization", stdout_logs=True, stdout_level=logging.INFO
    ):
        pulse_types = [ion.potentials.SincPulse, ion.potentials.GaussianPulse]
        pulse_widths = np.array([50, 100, 200, 400, 800, 1600]) * asec
        fluences = np.array([1]) * Jcm2
        phases = [0, pi / 2]
        omega_mins = np.array([10, 50, 100, 300, 500, 800]) * twopi * THz

        compare_pulse_types(pulse_types, pulse_widths, fluences, phases, omega_mins)
        compare_omega_mins(pulse_types, pulse_widths, fluences, phases, omega_mins)
        compare_omega_mins_movie(
            pulse_types,
            pulse_widths,
            fluences,
            phases,
            np.linspace(10, 1000, 600) * twopi * THz,
        )
