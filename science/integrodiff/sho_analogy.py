import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

time_field_unit = atomic_electric_field * atomic_time
hydrogen_omega = np.abs(ion.HydrogenBoundState(1).energy / hbar)


def new_amplitude(
    initial_amplitude,
    initial_phase,
    kick_amplitude,
    test_mass=electron_mass,
    test_charge=proton_charge,
    omega=hydrogen_omega,
):
    amp = test_charge * kick_amplitude / (test_mass * omega)
    return np.sqrt(
        (initial_amplitude ** 2)
        + (amp ** 2)
        + (2 * initial_amplitude * amp * np.cos(initial_phase))
    )


def new_phase(
    initial_amplitude,
    initial_phase,
    kick_amplitude,
    test_mass=electron_mass,
    test_charge=proton_charge,
    omega=hydrogen_omega,
):
    amp = test_charge * kick_amplitude / (test_mass * omega)
    num = initial_amplitude * np.sin(initial_phase)
    den = initial_amplitude * np.cos(initial_phase) + amp
    return np.arctan2(num, den)


if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra", "ionization", stdout_level=logging.DEBUG
    ) as logger:
        initial_amplitude = 1 * bohr_radius
        initial_phase = 0
        kick_amplitude = 0.3 * time_field_unit

        print(initial_amplitude)
        print(new_amplitude(initial_amplitude, initial_phase, kick_amplitude))
        print(new_phase(initial_amplitude, initial_phase, kick_amplitude))

        initial_phases = np.linspace(0, twopi, 100)

        si.vis.xy_plot(
            "new_amplitude_vs_initial_phase",
            initial_phases,
            new_amplitude(initial_amplitude, initial_phases, kick_amplitude),
            x_label=r"$\phi_x$",
            x_unit="rad",
            y_label=r"$x_1$",
            y_unit="bohr_radius",
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            "new_phase_vs_initial_phase",
            initial_phases,
            new_phase(initial_amplitude, initial_phases, kick_amplitude),
            x_label=r"$\phi_x$",
            x_unit="rad",
            y_label=r"$\phi$",
            y_unit="rad",
            **PLOT_KWARGS,
        )

        new_phases = new_phase(initial_amplitude, initial_phases, kick_amplitude)
        phase_to_pi = np.abs(pi - (new_phases % pi))
        time_to_zero = np.abs(phase_to_pi / hydrogen_omega)

        si.vis.xy_plot(
            "time_to_zero",
            initial_phases,
            time_to_zero,
            x_label=r"$\phi_x$",
            x_unit="rad",
            y_label=r"$\Delta t$",
            y_unit="asec",
            **PLOT_KWARGS,
        )

        print(np.mean(time_to_zero) / asec)
