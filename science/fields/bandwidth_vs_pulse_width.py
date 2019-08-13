import logging
import os
import itertools

import numpy as np

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="pdf", fig_dpi_scale=6)


def bandwidth_plot():
    pulse_widths = np.linspace(0, 1100, 1001)[1:] * u.asec
    pulses = [potentials.SincPulse(pulse_width=pw) for pw in pulse_widths]

    f_min = np.array([p.frequency_min for p in pulses])
    f_carrier = np.array([p.frequency_carrier for p in pulses])
    f_max = np.array([p.frequency_max for p in pulses])

    figman = si.vis.xy_plot(
        "bandwidth",
        pulse_widths,
        f_carrier,
        x_label=r"Pulse Width $ \tau $",
        y_label=r"Frequency $ f $",
        x_unit="asec",
        y_unit="THz",
        title="Sinc Pulse Bandwidth",
        line_kwargs=[dict(color="black", linewidth=2)],
        font_size_axis_labels=12,
        save_on_exit=False,
        close_after_exit=False,
        **PLOT_KWARGS
    )
    ax_freq = figman.fig.axes[0]
    ax_energy = ax_freq.twinx()
    ax_energy.set_ylabel(r"Photon Energy $E$ (eV)", fontsize=12)

    ax_freq.fill_between(
        pulse_widths / u.asec,
        f_min / u.THz,
        f_max / u.THz,
        facecolor="grey",
        edgecolor="black",
        linewidth=2,
    )

    energies = [0] + [states.HydrogenBoundState(n).energy for n in range(1, 4)]
    transition_energies = set(
        abs(a - b) for a, b in itertools.product(energies, repeat=2)
    )

    print([e / u.eV for e in transition_energies])

    for transition_energy in transition_energies:
        ax_energy.axhline(
            transition_energy / u.eV,
            linewidth=1,
            linestyle="-",
            color="blue",
            alpha=0.5,
        )

    y_lim_freq = 20000 * u.THz
    y_lim_energy = u.h * y_lim_freq

    ax_freq.set_xlim(0, 1000)
    ax_freq.set_ylim(0, y_lim_freq / u.THz)
    ax_energy.set_ylim(0, y_lim_energy / u.eV)

    figman.save()
    figman.cleanup()


if __name__ == "__main__":
    with LOGMAN as logger:
        bandwidth_plot()
