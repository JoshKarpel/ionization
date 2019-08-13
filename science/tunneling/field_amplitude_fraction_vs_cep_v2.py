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
        times = np.linspace(-bound * pw, bound * pw, 1e5)
        print(f"dt = {uround(times[1] - times[0], asec)} as")
        len_times = len(times)

        ceps = np.linspace(0, pi, 5e2)
        field_fraction_vs_cep = np.zeros(len(ceps))
        power_fraction_vs_cep = np.zeros(len(ceps))
        rms_field_vs_cep = np.zeros(len(ceps))
        rms_power_vs_cep = np.zeros(len(ceps))
        avg_abs_field_vs_cep = np.zeros(len(ceps))
        avg_abs_power_vs_cep = np.zeros(len(ceps))

        pot_zero = ion.SincPulse.from_omega_carrier(
            pulse_width=pw, fluence=flu, phase=0
        )

        field_zero = pot_zero.get_electric_field_amplitude(times)
        field_cut = np.max(np.abs(field_zero)) / 2
        power_cut = np.max(np.abs(field_zero)) / np.sqrt(2)

        for ii, cep in enumerate(tqdm(ceps)):
            pot = ion.SincPulse(
                pulse_width=pw,
                fluence=flu,
                phase=cep,
                window=ion.SymmetricExponentialTimeWindow(
                    window_time=(bound - 2) * pw, window_width=0.2 * pw
                ),
            )
            # pot = ion.DC_correct_electric_potential(pot, times)

            field = pot.get_electric_field_amplitude(times)

            field_fraction_vs_cep[ii] = (np.abs(field) > field_cut).sum() / len_times
            power_fraction_vs_cep[ii] = (np.abs(field) > power_cut).sum() / len_times
            rms_field_vs_cep[ii] = np.std(field)
            rms_power_vs_cep[ii] = np.std(np.abs(field) ** 2)
            avg_abs_field_vs_cep[ii] = np.mean(np.abs(field))
            avg_abs_power_vs_cep[ii] = np.mean(np.abs(field) ** 2)

        si.vis.xy_plot(
            f"relative_field_properties_vs_cep",
            ceps,
            field_fraction_vs_cep / field_fraction_vs_cep[0],
            power_fraction_vs_cep / power_fraction_vs_cep[0],
            rms_field_vs_cep / rms_field_vs_cep[0],
            rms_power_vs_cep / rms_power_vs_cep[0],
            avg_abs_field_vs_cep / avg_abs_field_vs_cep[0],
            avg_abs_power_vs_cep / avg_abs_power_vs_cep[0],
            line_labels=[
                rf"$ {ion.LATEX_EFIELD} $ Fraction",
                rf"$ \left|{ion.LATEX_EFIELD}\right|^2 $ Fraction",
                rf"RMS $ {ion.LATEX_EFIELD} $",
                rf"RMS $ \left|{ion.LATEX_EFIELD}\right|^2 $",
                rf"AVG $ \left|{ion.LATEX_EFIELD}\right| $",
                rf"AVG $ \left|{ion.LATEX_EFIELD}\right|^2 $",
            ],
            x_label=r"Carrier-Envelope Phase $ \varphi $",
            x_unit="rad",
            title=fr"Relative $ {ion.LATEX_EFIELD} $ Properties for T = $ {bound}\tau $",
            legend_on_right=True,
            **PLOT_KWARGS,
        )
