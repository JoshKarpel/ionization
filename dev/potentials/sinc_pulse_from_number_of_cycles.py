#!/usr/bin/env python

import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

if __name__ == "__main__":
    with LOGMAN as logger:
        number_of_cycles = [0.51, 1, 2, 3]
        nc_pulses = [
            (
                nc,
                potentials.SincPulse.from_number_of_cycles(
                    pulse_width=100 * u.asec, number_of_cycles=nc, phase=u.pi / 2
                ),
            )
            for nc in number_of_cycles
        ]

        # note that you actually get twice as many carrier cycles as you specify in the "center"
        # because the center of the sinc is twice as wide as a pulse width (it's double-sided)

        tb = 1
        for nc, pulse in nc_pulses:
            print(pulse.info())
            times = np.linspace(-tb * pulse.pulse_width, tb * pulse.pulse_width, 10000)
            si.vis.xy_plot(
                f"Nc={nc}",
                times,
                pulse.amplitude * np.cos((pulse.omega_carrier * times) + pulse.phase),
                pulse.get_electric_field_amplitude(times),
                line_labels=["carrier", "pulse"],
                line_kwargs=[{"linestyle": "--"}, None],
                x_unit=pulse.pulse_width,
                y_unit=pulse.amplitude,
                **PLOT_KWARGS,
            )
