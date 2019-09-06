import logging
import os

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=5)

if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra", "ionization", stdout_level=logging.DEBUG
    ) as logger:
        pw = 200 * asec
        flu = 1 * Jcm2

        t_bound = 30
        plot_bound = 35
        times = np.linspace(-plot_bound * pw, plot_bound * pw, 2 ** 12)

        window = ion.potentials.LogisticWindow(
            window_time=t_bound * pw, window_width=0.2 * pw
        )

        base_pulse = ion.potentials.SincPulse(
            pulse_width=pw, fluence=flu, window=window
        )
        base_pulse = ion.DC_correct_electric_potential(base_pulse, times)

        replica_pulse = ion.GenericElectricPotential.from_pulse(
            base_pulse, times, phase_function=lambda f: 0, window=window
        )
        replica_pulse = ion.DC_correct_electric_potential(replica_pulse, times)

        rand_pulse = ion.GenericElectricPotential.from_pulse(
            base_pulse,
            times,
            phase_function=lambda f: si.math.rand_phase(f.shape),
            window=window,
        )

        print(base_pulse)
        print(replica_pulse)
        print(rand_pulse)

        # print(rand_pulse.complex_amplitude_vs_frequency)
        # si.vis.xy_plot(
        #     'spectra',
        #     rand_pulse.frequency,
        #     np.real(rand_pulse.complex_amplitude_vs_frequency),
        #     np.imag(rand_pulse.complex_amplitude_vs_frequency),
        #     np.abs(rand_pulse.complex_amplitude_vs_frequency),
        #     line_labels = ['real', 'imag'],
        #     x_unit = 'THz', x_lower_limit = -6000 * THz, x_upper_limit = 6000 * THz,
        #     **PLOT_KWARGS,
        # )

        ion.potentials.plot_electric_field_amplitude_vs_time(
            "pulses",
            times,
            base_pulse,
            replica_pulse,
            rand_pulse,
            line_labels=["base", "replica", "rand"],
            line_kwargs=[None, {"linestyle": "--"}],
            **PLOT_KWARGS,
        )

        for pulse in [base_pulse, replica_pulse, rand_pulse]:
            print(
                pulse.get_fluence_numeric(times) / Jcm2,
                pulse.get_vector_potential_amplitude_numeric(times),
                pulse,
            )

        rand_pulses = []
        for i in range(5):
            rand_pulses.append(
                ion.GenericElectricPotential.from_pulse(
                    base_pulse,
                    times,
                    phase_function=lambda f: si.math.rand_phase(f.shape),
                    window=window,
                )
            )

        ion.potentials.plot_electric_field_amplitude_vs_time(
            "lots_of_pulses", times, *rand_pulses, **PLOT_KWARGS
        )
