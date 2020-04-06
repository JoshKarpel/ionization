import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def power_exclusion():
    pw = 3400 * asec
    exclusions = [0.5, 1, 2, 3]
    phase = 0

    plot_bound = 5

    pulses_by_exclusion = [
        ion.potentials.GaussianPulse.from_power_exclusion(
            pulse_width=pw,
            exclusion=exclusion,
            phase=phase,
            window=ion.potentials.LogisticWindow(
                window_time=3 * pw, window_width=0.1 * pw
            ),
        )
        for exclusion in exclusions
    ]

    ti_sapph = ion.potentials.GaussianPulse.from_omega_carrier(
        pulse_width=3400 * asec, omega_carrier=twopi * c / (800 * nm)
    )

    times = np.linspace(-plot_bound * pw, plot_bound * pw, 1e3)

    si.vis.xy_plot(
        "power_exclusion_comparison",
        times,
        *[pulse.get_electric_field_amplitude(times) for pulse in pulses_by_exclusion],
        # ti_sapph.get_electric_field_amplitude(times),
        line_labels=[
            rf"$N_{{\sigma}} = {exclusion}$" for exclusion in exclusions
        ],  # + ['tisapph'],
        x_unit="asec",
        x_label=r"Time $ t $",
        y_unit="atomic_electric_field",
        y_label=rf"$ {ion.vis.LATEX_EFIELD}(t) $",
        legend_on_right=True,
        **PLOT_KWARGS,
    )

    si.vis.xy_plot(
        "power_exclusion_comparison__dc_corrected",
        times,
        *[
            ion.DC_correct_electric_potential(
                pulse, times
            ).get_electric_field_amplitude(times)
            for pulse in pulses_by_exclusion
        ],
        # ti_sapph.get_electric_field_amplitude(times),
        line_labels=[
            rf"$N_{{\sigma}} = {exclusion}$" for exclusion in exclusions
        ],  # + ['tisapph'],
        x_unit="asec",
        x_label=r"Time $ t $",
        y_unit="atomic_electric_field",
        y_label=rf"$ {ion.vis.LATEX_EFIELD}(t) $",
        legend_on_right=True,
        **PLOT_KWARGS,
    )

    for pulse, exclusion in zip(pulses_by_exclusion, exclusions):
        # print(pulse.info())
        print()
        print(f"N_sigma: {exclusion}")
        print(f"Numeric Fluence: {pulse.get_fluence_numeric(times)/ Jcm2:.3f} J/cm^2")
        print(
            f"Vector Potential at End: {proton_charge * pulse.get_vector_potential_amplitude_numeric(times)/atomic_momentum:3f} a.u."
        )
        print()


def number_of_cycles():
    pw = 200 * asec
    num_cycles = [3, 4]
    phase = 0

    plot_bound = 5

    pulses_by_num_cycles = [
        ion.potentials.GaussianPulse.from_number_of_cycles(
            pulse_width=pw,
            number_of_cycles=num_cyc,
            number_of_pulse_widths=3,
            phase=phase,
            window=ion.potentials.LogisticWindow(
                window_time=3 * pw, window_width=0.1 * pw
            ),
        )
        for num_cyc in num_cycles
    ]

    times = np.linspace(-plot_bound * pw, plot_bound * pw, 1e3)

    si.vis.xy_plot(
        "number_of_cycles_comparison",
        times,
        *[pulse.get_electric_field_amplitude(times) for pulse in pulses_by_num_cycles],
        line_labels=[rf"$N_c = {num_cyc}$" for num_cyc in num_cycles],
        x_unit="asec",
        x_label=r"Time $ t $",
        y_unit="atomic_electric_field",
        y_label=rf"$ {ion.vis.LATEX_EFIELD}(t) $",
        legend_on_right=True,
        **PLOT_KWARGS,
    )

    si.vis.xy_plot(
        "number_of_cycles_comparison__dc_corrected",
        times,
        *[
            ion.DC_correct_electric_potential(
                pulse, times
            ).get_electric_field_amplitude(times)
            for pulse in pulses_by_num_cycles
        ],
        line_labels=[rf"$N_c = {num_cyc}$" for num_cyc in num_cycles],
        x_unit="asec",
        x_label=r"Time $ t $",
        y_unit="atomic_electric_field",
        y_label=rf"$ {ion.vis.LATEX_EFIELD}(t) $",
        legend_on_right=True,
        **PLOT_KWARGS,
    )

    for pulse, num_cyc in zip(pulses_by_num_cycles, num_cycles):
        # print(pulse.info())
        print()
        print(f"N_cycles: {num_cyc}")
        print(f"Numeric Fluence: {pulse.get_fluence_numeric(times)/ Jcm2:.3f} J/cm^2")
        print(
            f"Vector Potential at End: {proton_charge * pulse.get_vector_potential_amplitude_numeric(times)/ atomic_momentum:.3f} a.u."
        )
        print()


def number_of_cycles_fluence_and_vp():
    pw = 200 * asec
    num_cycles = np.linspace(0.5, 10, 100)
    phase = 0

    plot_bound = 5

    pulses_by_num_cycles = [
        ion.potentials.GaussianPulse.from_number_of_cycles(
            pulse_width=pw,
            number_of_cycles=num_cyc,
            number_of_pulse_widths=3,
            phase=phase,
            window=ion.potentials.LogisticWindow(
                window_time=3 * pw, window_width=0.1 * pw
            ),
        )
        for num_cyc in num_cycles
    ]

    times = np.linspace(-plot_bound * pw, plot_bound * pw, 1e3)

    si.vis.xy_plot(
        "number_of_cycles__fluence_diff",
        num_cycles,
        [
            (1 * Jcm2) - pulse.get_fluence_numeric(times)
            for pulse in pulses_by_num_cycles
        ],
        x_label=r"Number of Cycles $N_c$",
        y_label=r"$ H - 1 \, \mathrm{J/cm^2} $",
        y_unit="Jcm2",
        title="Fluence Error",
        **PLOT_KWARGS,
    )

    si.vis.xy_plot(
        "number_of_cycles__final_vp",
        num_cycles,
        [
            proton_charge * pulse.get_vector_potential_amplitude_numeric(times)
            for pulse in pulses_by_num_cycles
        ],
        x_label=r"Number of Cycles $N_c$",
        y_label=rf"$ q \, {ion.LATEX_AFIELD}(t_{{\mathrm{{final}}}}) $",
        y_unit="atomic_momentum",
        title="Vector Potential Error",
        **PLOT_KWARGS,
    )

    si.vis.xy_plot(
        "number_of_cycles__fluence__frac_log",
        num_cycles,
        [
            np.abs(pulse.get_fluence_numeric(times) / (1 * Jcm2))
            for pulse in pulses_by_num_cycles
        ],
        x_label=r"Number of Cycles $N_c$",
        y_label=r"$ \left| H / 1 \mathrm{J/cm^2} \right| $",
        title="Fluence Error",
        y_log_axis=True,
        **PLOT_KWARGS,
    )

    si.vis.xy_plot(
        "number_of_cycles__final_vp__log",
        num_cycles,
        [
            np.abs(proton_charge * pulse.get_vector_potential_amplitude_numeric(times))
            for pulse in pulses_by_num_cycles
        ],
        x_label=r"Number of Cycles $N_c$",
        y_label=rf"$ \left| q \, {ion.LATEX_AFIELD}(t_{{\mathrm{{final}}}}) \right| $",
        y_unit="atomic_momentum",
        title="Vector Potential Error",
        y_log_axis=True,
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    with LOGMAN as logger:
        power_exclusion()
        number_of_cycles()
        number_of_cycles_fluence_and_vp()
