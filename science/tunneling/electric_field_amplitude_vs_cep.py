import logging
import os

from tqdm import tqdm

import numpy as np

import simulacra as si
from simulacra.units import *
import ionization as ion

# import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

log = si.utils.LogManager(
    "simulacra",
    "ionization",
    stdout_level=logging.INFO,
    file_logs=False,
    file_dir=OUT_DIR,
    file_level=logging.DEBUG,
)

PLOT_KWARGS = dict(
    target_dir=OUT_DIR,
    # img_format = 'png',
    # fig_dpi_scale = 6,
)

if __name__ == "__main__":
    with log as logger:
        pw = 200 * asec
        flu = 1 * Jcm2

        bound = 30
        times = np.linspace(-bound * pw, bound * pw, 1e4)
        len_times = len(times)

        ceps = np.linspace(0, twopi, 1e3)

        power_fraction.potentials._by_Pulse_type = dict()

        for pulse_type in (
            ion.potentials.SincPulse,
            ion.potentials.GaussianPulse,
            ion.potentials.SechPulse,
        ):
            power_fraction_vs_cep = np.zeros(len(ceps))

            pot_zero = pulse_type.from_omega_carrier(
                pulse_width=pw, fluence=flu, phase=0
            )

            field_zero = pot_zero.get_electric_field_amplitude(times)
            power_cut = np.nanmax(np.abs(field_zero)) / np.sqrt(2)

            for ii, cep in enumerate(tqdm(ceps)):
                pot = pulse_type(
                    pulse_width=pw,
                    fluence=flu,
                    phase=cep,
                    window=ion.potentials.LogisticWindow(
                        window_time=(bound - 2) * pw, window_width=0.2 * pw
                    ),
                )

                pot = ion.DC_correct_electric_potential(pot, times)

                field = pot.get_electric_field_amplitude(times)

                power_fraction_vs_cep[ii] = (
                    np.abs(field) > power_cut
                ).sum() / len_times

            power_fraction.potentials._by_Pulse_type[pulse_type] = (
                power_fraction_vs_cep / power_fraction_vs_cep[0]
            )  # normalize to cep = 0

        si.vis.xy_plot(
            # f'power_fractions_vs_cep',
            f"power_fractions_vs_cep__dc_corrected",
            ceps,
            *(v for k, v in power_fraction.potentials._by_Pulse_type.items()),
            line_labels=(k.__name__ for k in power_fraction.potentials._by_Pulse_type),
            x_label=r"Carrier-Envelope Phase $\varphi$",
            x_unit="rad",
            y_label=r"Rel. Fraction of Time at $>\frac{1}{2}$ Power",
            title=fr"Rel. Time-Power Fraction for T = ${bound}\tau$",
            **PLOT_KWARGS,
        )
