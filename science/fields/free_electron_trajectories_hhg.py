import logging
import os

from tqdm import tqdm

import numpy as np
import scipy.integrate as integ

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

SIM_LIB = os.path.join(OUT_DIR, "SIMLIB")

logman = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

if __name__ == "__main__":
    with logman as logger:
        t_bound = 5
        p_bound = 4

        phases = [0, pi / 4, pi / 2]

        for cep in tqdm(phases):
            wave = ion.SineWave.from_wavelength(
                wavelength=800 * nm, amplitude=0.0001 * atomic_electric_field, phase=cep
            )

            times = np.linspace(-t_bound * wave.period, t_bound * wave.period, 1e4)

            window = ion.SymmetricExponentialTimeWindow(
                window_time=p_bound * wave.period, window_width=0.2 * wave.period
            )
            wave = ion.SineWave.from_wavelength(
                wavelength=800 * nm,
                amplitude=0.0001 * atomic_electric_field,
                phase=cep,
                window=window,
            )
            # corrected_pulse = ion.DC_correct_electric_potential(wave, times)

            efield = wave.get_electric_field_amplitude(times)
            afield = wave.get_vector_potential_amplitude_numeric_cumulative(times)

            starts = range(0, len(times), 100)[10:20]

            sliced_times = list(times[start:] for start in starts)
            sliced_alphas = list(
                (proton_charge / electron_mass)
                * integ.cumtrapz(
                    y=integ.cumtrapz(y=efield[start:], x=times[start:], initial=0),
                    x=times[start:],
                    initial=0,
                )
                for start in starts
            )

            identifier = f"{wave.__class__.__name__}__lambda={uround(wave.wavelength, nm, 0)}nm_amp={uround(wave.amplitude, atomic_electric_field, 5)}aef_cep={uround(cep, pi, 2)}pi"

            si.vis.xxyy_plot(
                identifier,
                [times, times, *sliced_times],
                [
                    efield / np.max(np.abs(efield)),
                    afield / np.max(np.abs(afield)),
                    *(alpha / bohr_radius for alpha in sliced_alphas),
                ],
                line_labels=[
                    rf"$ {ion.LATEX_EFIELD}(t) $ (norm.)",
                    rf"$ e \, {ion.LATEX_AFIELD}(t) $ (norm.)",
                    rf"$ \alpha(t) $",
                ],
                line_kwargs=[
                    None,
                    None,
                    *({"color": "black", "alpha": 0.5} for _ in starts),
                ],
                x_label=r"Time $t$",
                x_unit="asec",
                title=rf"$ \lambda = {uround(wave.wavelength, nm, 0)} \, \mathrm{{nm}}, \; {ion.LATEX_EFIELD}_0 = {uround(wave.amplitude, atomic_electric_field, 5)} \, \mathrm{{a.u.}}, \; \varphi = {uround(cep, pi, 2)}\pi $",
                **PLOT_KWARGS,
            )
